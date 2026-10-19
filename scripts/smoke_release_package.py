"""Check an extracted release ZIP and load its production x64 DLL on Windows.

This verifies packaging and ABI exports, not model quality or GUI installation.
"""
import argparse
import ctypes
import os
from pathlib import Path
import tempfile
import zipfile
import json
import py_compile


def inspect_package(package, load_dll=False, current=False):
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        with zipfile.ZipFile(package) as archive:
            for item in archive.infolist():
                target = (root / item.filename).resolve()
                if not target.is_relative_to(root.resolve()):
                    raise ValueError("unsafe ZIP path")
            archive.extractall(root)
        dlls = list(root.rglob("NppAI.dll"))
        if len(dlls) != 1 or dlls[0].parent.name != "NppAI":
            raise ValueError("expected one NppAI/NppAI.dll")
        dll = dlls[0]
        for name in ("README.md", "model-format.md"):
            if not (dll.parent / name).is_file():
                raise ValueError("missing package documentation: " + name)
        if current:
            for name in ("sandbox_validate.py", "docs/notepad-setup-pl.md", "docs/release-readiness.md",
                         "examples/silnia-validation-cases.json"):
                if not (dll.parent/name).is_file():raise ValueError("missing current package dependency: "+name)
            py_compile.compile(str(dll.parent/'sandbox_validate.py'),cfile=str(root/'validator.pyc'),doraise=True)
            json.loads((dll.parent/'examples/silnia-validation-cases.json').read_text(encoding='utf-8'))
        if load_dll:
            if os.name != "nt" or ctypes.sizeof(ctypes.c_void_p) != 8:
                raise ValueError("DLL load requires 64-bit Windows Python")
            plugin = ctypes.WinDLL(str(dll))
            try:
                for name in ("setInfo", "getName", "getFuncsArray", "beNotified", "isUnicode", "messageProc"):
                    getattr(plugin, name)
                plugin.getName.restype = ctypes.c_wchar_p
                plugin.getName.argtypes = []
                if plugin.getName() != "NppAI":
                    raise ValueError("unexpected plugin name")
                plugin.isUnicode.restype = ctypes.c_int
                plugin.isUnicode.argtypes = []
                if not plugin.isUnicode():
                    raise ValueError("plugin does not declare Unicode support")
            finally:
                kernel = ctypes.WinDLL("kernel32", use_last_error=True)
                kernel.FreeLibrary.argtypes = [ctypes.c_void_p]
                kernel.FreeLibrary.restype = ctypes.c_int
                kernel.FreeLibrary(plugin._handle)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package")
    parser.add_argument("--load-dll", action="store_true")
    parser.add_argument("--current", action="store_true")
    args = parser.parse_args()
    inspect_package(args.package, args.load_dll,args.current)
    print("Release ZIP smoke test passed")
