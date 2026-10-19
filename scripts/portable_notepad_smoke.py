"""Fresh portable Notepad++ install, real own-model readiness and panel smoke.

Runs only in an isolated Windows CI directory. It does not validate code quality,
Docker Desktop setup, upgrades, or the user's existing editor installation.
"""
import argparse
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


def run(editor_zip,package,model,output):
    if os.name!='nt':raise RuntimeError('Real portable editor smoke requires Windows')
    output.mkdir(parents=True,exist_ok=True)
    inspect_package(package,load_dll=True,current=True)
    with tempfile.TemporaryDirectory(prefix='nppai-clean-install-') as directory:
        root=Path(directory);editor_dir=root/'editor';editor_dir.mkdir();expand(editor_zip,editor_dir)
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
        # A missing Docker configuration must keep Apply disabled in this test.
        for key in ('NPPAI_VALIDATION_IMAGE','NPPAI_TEST_CASES'):
            environment.pop(key,None)
        document=root/'smoke.py';document.write_bytes(b'')
        ui=WindowsUI();editor=None
        with (output/'server.log').open('w',encoding='utf-8') as log:
            server=subprocess.Popen([sys.executable,'-m','uvicorn','server_backend:app','--host','127.0.0.1','--port',str(port)],cwd=ROOT,env=environment,stdout=log,stderr=log)
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
                if not all(key in controls for key in (1,2,3,5,6)):raise RuntimeError('Panel controls missing')
                task='@python-signature silnia(n)\r\nWygeneruj kod na silnię. Zażółć gęślą jaźń.'
                if not ui.api.IsWindowUnicode(controls[1]) or not ui.set_text(controls[1],task):raise RuntimeError('Unicode prompt field unavailable')
                if ui.text(controls[1])!=task:raise RuntimeError('Polish prompt round trip failed')
                button=ui.text(controls[2])
                if button not in ('Generate AI Code','Wygeneruj Kod AI'):raise RuntimeError('Unexpected generation button text: '+button)
                if ui.api.IsWindowEnabled(controls[5]):raise RuntimeError('Apply enabled without validation')
                report={'editor_archive_sha256':hashlib.sha256(editor_zip.read_bytes()).hexdigest(),
                    'package_sha256':hashlib.sha256(package.read_bytes()).hexdigest(),
                    'model_sha256':hashlib.sha256(model.read_bytes()).hexdigest(),
                    'health':health,'plugin_menu':list(path),'panel_class':ui.cls(panel),
                    'polish_prompt_round_trip':True,'button_text':button,'apply_disabled_without_validation':True,
                    'scope':'Fresh portable install, real server/model readiness and real panel controls. No generated-code quality, Docker Desktop or upgrade verification.'}
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
    a=p.parse_args();run(a.editor_zip,a.package,a.model,a.output)
