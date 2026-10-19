"""Fresh portable Notepad++ install, real own-model readiness and panel smoke.

Runs only in an isolated Windows CI directory. It does not validate code quality,
Docker Desktop setup, upgrades, or the user's existing editor installation.
"""
import argparse
import ast
import ctypes
from ctypes import wintypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import zipfile

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.start_nppai import connection_environment,wait_ready
from scripts.smoke_release_package import inspect_package


def expand(archive,destination):
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            if not (destination/member.filename).resolve().is_relative_to(destination.resolve()):
                raise ValueError('Unsafe ZIP member')
        z.extractall(destination)


class WindowsUI:
    def __init__(self):
        self.api=ctypes.WinDLL('user32',use_last_error=True)
        self.callback=ctypes.WINFUNCTYPE(wintypes.BOOL,wintypes.HWND,wintypes.LPARAM)
        signatures={
            'EnumWindows':([self.callback,wintypes.LPARAM],wintypes.BOOL),
            'EnumChildWindows':([wintypes.HWND,self.callback,wintypes.LPARAM],wintypes.BOOL),
            'GetWindowThreadProcessId':([wintypes.HWND,ctypes.POINTER(wintypes.DWORD)],wintypes.DWORD),
            'GetClassNameW':([wintypes.HWND,wintypes.LPWSTR,ctypes.c_int],ctypes.c_int),
            'GetWindowTextW':([wintypes.HWND,wintypes.LPWSTR,ctypes.c_int],ctypes.c_int),
            'SetWindowTextW':([wintypes.HWND,wintypes.LPCWSTR],wintypes.BOOL),
            'GetMenu':([wintypes.HWND],wintypes.HMENU),
            'GetSubMenu':([wintypes.HMENU,ctypes.c_int],wintypes.HMENU),
            'GetMenuItemCount':([wintypes.HMENU],ctypes.c_int),
            'GetMenuStringW':([wintypes.HMENU,wintypes.UINT,wintypes.LPWSTR,ctypes.c_int,wintypes.UINT],ctypes.c_int),
            'GetMenuItemID':([wintypes.HMENU,ctypes.c_int],wintypes.UINT),
            'PostMessageW':([wintypes.HWND,wintypes.UINT,wintypes.WPARAM,wintypes.LPARAM],wintypes.BOOL),
            'GetDlgCtrlID':([wintypes.HWND],ctypes.c_int),
            'IsWindowUnicode':([wintypes.HWND],wintypes.BOOL),
            'IsWindowEnabled':([wintypes.HWND],wintypes.BOOL),
            'ShowWindow':([wintypes.HWND,ctypes.c_int],wintypes.BOOL),
            'SetForegroundWindow':([wintypes.HWND],wintypes.BOOL),
            'GetWindowRect':([wintypes.HWND,ctypes.POINTER(wintypes.RECT)],wintypes.BOOL),
            'SendMessageTimeoutW':([wintypes.HWND,wintypes.UINT,wintypes.WPARAM,wintypes.LPARAM,wintypes.UINT,wintypes.UINT,ctypes.POINTER(ctypes.c_size_t)],ctypes.c_size_t)}
        for name,(args,result) in signatures.items():
            fn=getattr(self.api,name);fn.argtypes=args;fn.restype=result

    def windows(self,parent=None):
        found=[]
        def collect(hwnd,_):found.append(hwnd);return True
        callback=self.callback(collect)
        if parent is None:self.api.EnumWindows(callback,0)
        else:self.api.EnumChildWindows(parent,callback,0)
        return found

    def pid(self,hwnd):
        result=wintypes.DWORD();self.api.GetWindowThreadProcessId(hwnd,ctypes.byref(result));return result.value

    def text(self,hwnd):
        # WM_GETTEXT/WM_SETTEXT are marshalled by Windows across processes;
        # GetWindowText cannot read another process's edit-control contents.
        text=ctypes.create_unicode_buffer(16384);result=ctypes.c_size_t()
        if not self.api.SendMessageTimeoutW(hwnd,0xD,len(text),ctypes.cast(text,ctypes.c_void_p).value,2,3000,ctypes.byref(result)):
            raise RuntimeError('Cross-process text read timed out')
        return text.value

    def set_text(self,hwnd,value):
        text=ctypes.create_unicode_buffer(value);result=ctypes.c_size_t()
        return bool(self.api.SendMessageTimeoutW(hwnd,0xC,0,ctypes.cast(text,ctypes.c_void_p).value,2,3000,ctypes.byref(result)) and result.value)

    def scroll_to_end(self,hwnd):
        result=ctypes.c_size_t()
        end=len(self.text(hwnd).encode('utf-16-le'))//2
        for message,wparam,lparam in ((0xB1,end,end),(0xB6,0,32767),(0xB7,0,0)):
            if not self.api.SendMessageTimeoutW(hwnd,message,wparam,lparam,2,3000,ctypes.byref(result)):
                raise RuntimeError('Cannot reveal actual response end')

    def cls(self,hwnd):
        text=ctypes.create_unicode_buffer(256);self.api.GetClassNameW(hwnd,text,len(text));return text.value

    def menu(self,menu,path=()):
        for index in range(self.api.GetMenuItemCount(menu)):
            text=ctypes.create_unicode_buffer(512);self.api.GetMenuStringW(menu,index,text,len(text),0x400)
            current=path+(text.value.replace('&',''),)
            child=self.api.GetSubMenu(menu,index)
            if child:yield from self.menu(child,current)
            else:yield current,self.api.GetMenuItemID(menu,index)


def wait_for(find,process,timeout=45):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        if process.poll() is not None:raise RuntimeError('Notepad++ exited before smoke completed')
        result=find()
        if result:return result
        time.sleep(0.25)
    raise TimeoutError('Expected Notepad++ UI did not become ready')


def verify_service(root):
    manifest=json.loads((root/'SOURCE_MANIFEST.json').read_bytes())
    for name,expected in manifest['files'].items():
        path=(root/name).resolve()
        if not path.is_relative_to(root.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            raise ValueError('Packaged service source changed: '+name)
    for name in ('server_backend.py','model_backends.py','scripts/start_nppai.py','scripts/check_dense_runtime_parity.py','requirements-inference.txt'):
        if name not in manifest['files']:raise ValueError('Missing service runtime: '+name)
    return manifest


class WindowRecording:
    """Sample the actual CI editor window; do not synthesize or annotate frames."""
    def __init__(self,ui,window,path):
        self.ui,self.window,self.path=ui,window,path
        self.frames=[];self.times=[]

    def capture(self):
        from PIL import ImageGrab
        rect=wintypes.RECT()
        if not self.ui.api.GetWindowRect(self.window,ctypes.byref(rect)):
            raise RuntimeError('Cannot locate demo editor window')
        frame=ImageGrab.grab(bbox=(rect.left,rect.top,rect.right,rect.bottom))
        if frame.width<100 or frame.height<100 or not any(high>low for low,high in frame.getextrema()):
            raise RuntimeError('Editor capture is empty or unavailable')
        self.frames.append(frame);self.times.append(time.monotonic())

    def save(self):
        if len(self.frames)<2:raise RuntimeError('No actual recording frames')
        durations=[max(100,round((b-a)*1000)) for a,b in zip(self.times,self.times[1:])]+[3000]
        self.frames[0].save(self.path,save_all=True,append_images=self.frames[1:],
                            duration=durations,loop=0,optimize=False)


def inspect_review_syntax(history):
    """Read the empty-document diff; parse model output without executing it."""
    marker=history.find('\n@@ ')
    if marker<0:raise ValueError('Review has no actual generated diff')
    lines=history[marker:].splitlines()[2:]
    if not lines or any(line and not line.startswith('+') for line in lines):
        raise ValueError('Expected an insertion into the empty demo document')
    code='\n'.join(line[1:] for line in lines if line.startswith('+'))+'\n'
    if not code.strip():raise ValueError('Actual demo proposal is empty')
    try:
        ast.parse(code)
        syntax={'passed':True,'execution':'not_run','functional_pass':'not_measured'}
    except SyntaxError as error:
        syntax={'passed':False,'error':error.msg,'line':error.lineno,
                'execution':'not_run','functional_pass':'not_measured'}
    return {'code':code,'syntax':syntax,'source':'actual_plugin_insertion_diff'}


def exercise_generation(ui,main,panel,controls,editor,output,record=False):
    ui.api.ShowWindow(main,3);ui.api.SetForegroundWindow(main)
    recording=WindowRecording(ui,main,output/'notepad-live-demo.gif') if record else None
    before=ui.text(controls[3]);events=[]
    def sample():
        events.append({'elapsed_seconds':round(time.monotonic()-start,3),
                       'status':ui.text(controls[4]),'history':ui.text(controls[3]),
                       'apply_enabled':bool(ui.api.IsWindowEnabled(controls[5]))})
        if recording:recording.capture()
    start=time.monotonic();time.sleep(1);sample()
    if not ui.api.PostMessageW(panel,0x111,2,controls[2]):
        raise RuntimeError('Cannot request actual generation')
    completed=False
    try:
        while time.monotonic()-start<90:
            if editor.poll() is not None:raise RuntimeError('Editor exited during generation')
            time.sleep(1);sample()
            if events[-1]['apply_enabled']:
                raise RuntimeError('Unvalidated generated output enabled Apply')
            if events[-1]['history']!=before and ui.api.IsWindowEnabled(controls[6]):
                completed=True;break
        if not completed:raise TimeoutError('Actual generation did not reach review')
        # A changed history alone could be a connection error. Require a real
        # response/repair review and corroborate the HTTP request in server.log.
        history=events[-1]['history']
        if 'Candidates validated:' not in history:
            raise RuntimeError('Generation did not reach the proposal/validation report')
        if not any('POST /api/v1/generate HTTP/1.1" 200' in line for line in (output/'server.log').read_text(encoding='utf-8').splitlines()):
            raise RuntimeError('No successful real model HTTP response observed')
        ui.scroll_to_end(controls[3]);time.sleep(1);sample()
        return {'completed':True,'task':ui.text(controls[1]),'events':events,
                'review':history,'status':events[-1]['status'],
                'model_http_response_200':True,'apply_disabled':True,
                'validation_configured':False,'functional_success_claimed':False,
                'recording_scrolls_history_to_end':True,
                'review_uses_crlf':'\n' not in history.replace('\r\n',''),
                'proposal':inspect_review_syntax(history),
                'recording':'notepad-live-demo.gif' if recording else None}
    finally:
        (output/'generation-events.json').write_text(json.dumps(events,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        if recording:
            recording.capture();recording.save()


def run(editor_zip,package,model,output,service_zip=None,exercise=False,record=False):
    if os.name!='nt':raise RuntimeError('Real portable editor smoke requires Windows')
    output.mkdir(parents=True,exist_ok=True)
    inspect_package(package,load_dll=True,current=True)
    with tempfile.TemporaryDirectory(prefix='nppai-clean-install-') as directory:
        root=Path(directory);editor_dir=root/'editor';editor_dir.mkdir();expand(editor_zip,editor_dir)
        service_root=ROOT
        if service_zip:
            service_root=root/'service';service_root.mkdir();expand(service_zip,service_root)
            verify_service(service_root)
        binary=editor_dir/'notepad++.exe'
        if not binary.is_file():raise ValueError('Portable editor layout differs')
        unpack=root/'package';unpack.mkdir();expand(package,unpack)
        dlls=list(unpack.rglob('NppAI.dll'))
        if len(dlls)!=1:raise ValueError('Expected one production plugin')
        destination=editor_dir/'plugins/NppAI'
        shutil.copytree(dlls[0].parent,destination)
        if not (destination/'sandbox_validate.py').is_file():raise ValueError('Current validator helper missing')
        with socket.socket() as socket_probe:
            socket_probe.bind(('127.0.0.1',0));port=socket_probe.getsockname()[1]
        environment=connection_environment(model,port)
        environment.pop('PYTHONPATH',None)
        # A missing Docker configuration must keep Apply disabled in this test.
        for key in ('NPPAI_VALIDATION_IMAGE','NPPAI_TEST_CASES'):
            environment.pop(key,None)
        document=root/'smoke.py';document.write_bytes(b'')
        ui=WindowsUI();editor=None
        with (output/'server.log').open('w',encoding='utf-8') as log:
            server=subprocess.Popen([sys.executable,'-m','uvicorn','server_backend:app','--host','127.0.0.1','--port',str(port)],cwd=service_root,env=environment,stdout=log,stderr=log)
            try:
                health=wait_ready(server,port,environment['NPPAI_API_KEY'])
                editor=subprocess.Popen([str(binary),'-multiInst','-nosession',str(document)],env=environment)
                main=wait_for(lambda:next((h for h in ui.windows() if ui.pid(h)==editor.pid and ui.cls(h)=='Notepad++'),None),editor)
                def command():
                    return next(((path,item) for path,item in ui.menu(ui.api.GetMenu(main))
                        if any('NppAI' in part for part in path) and any(label in path[-1] for label in ('Show AI Panel','Pokaż Panel AI'))),None)
                path,command_id=wait_for(command,editor)
                if not ui.api.PostMessageW(main,0x111,command_id,0):raise RuntimeError('Cannot open NppAI panel')
                panel=wait_for(lambda:next((h for h in ui.windows(main) if ui.cls(h)=='NppAI_PanelClass'),None),editor)
                controls={ui.api.GetDlgCtrlID(h):h for h in ui.windows(panel)}
                if not all(key in controls for key in (1,2,3,4,5,6)):raise RuntimeError('Panel controls missing')
                task='@python-signature silnia(n)\r\nWygeneruj kod na silnię. Zażółć gęślą jaźń.'
                if exercise:
                    task='@python-signature silnia(n)\r\nWygeneruj kod na silnię dla nieujemnej liczby całkowitej n. Zwróć wynik.'
                if not ui.api.IsWindowUnicode(controls[1]) or not ui.set_text(controls[1],task):raise RuntimeError('Unicode prompt field unavailable')
                if ui.text(controls[1])!=task:raise RuntimeError('Polish prompt round trip failed')
                button=ui.text(controls[2])
                if button not in ('Generate AI Code','Wygeneruj Kod AI'):raise RuntimeError('Unexpected generation button text: '+button)
                if ui.api.IsWindowEnabled(controls[5]):raise RuntimeError('Apply enabled without validation')
                demo=exercise_generation(ui,main,panel,controls,editor,output,record) if exercise else None
                if demo and record:
                    demo['recording_sha256']=hashlib.sha256((output/'notepad-live-demo.gif').read_bytes()).hexdigest()
                report={'editor_archive_sha256':hashlib.sha256(editor_zip.read_bytes()).hexdigest(),
                    'package_sha256':hashlib.sha256(package.read_bytes()).hexdigest(),
                    'model_sha256':hashlib.sha256(model.read_bytes()).hexdigest(),
                    'service_archive_sha256':hashlib.sha256(service_zip.read_bytes()).hexdigest() if service_zip else None,
                    'service_from_extracted_package':bool(service_zip),
                    'health':health,'plugin_menu':list(path),'panel_class':ui.cls(panel),
                    'polish_prompt_round_trip':True,'button_text':button,'apply_disabled_without_validation':True,
                    'scope':'Fresh portable install, real server/model readiness and real panel controls. No generated-code quality, Docker Desktop or upgrade verification.'}
                if demo:report['live_generation_demo']=demo
                (output/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
                print(json.dumps(report,indent=2,ensure_ascii=False))
            finally:
                if editor and editor.poll() is None:
                    editor.terminate();editor.wait(timeout=10)
                server.terminate()
                try:server.wait(timeout=10)
                except subprocess.TimeoutExpired:server.kill();server.wait(timeout=5)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('editor-zip','package','model','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--service-zip',type=Path)
    p.add_argument('--exercise-generation',action='store_true')
    p.add_argument('--record-gif',action='store_true')
    a=p.parse_args()
    if a.record_gif and not a.exercise_generation:p.error('--record-gif requires --exercise-generation')
    run(a.editor_zip,a.package,a.model,a.output,a.service_zip,a.exercise_generation,a.record_gif)
