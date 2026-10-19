import tempfile
from pathlib import Path
import unittest
import zipfile
import py_compile
from scripts.smoke_release_package import inspect_package


class CurrentPackageTests(unittest.TestCase):
    def write(self,path,helper='pass\n',missing=None):
        files={'NppAI.dll':b'fixture','README.md':b'doc','model-format.md':b'doc',
            'sandbox_validate.py':helper.encode(),'docs/notepad-setup-pl.md':b'doc',
            'docs/release-readiness.md':b'doc','examples/silnia-validation-cases.json':b'[]'}
        with zipfile.ZipFile(path,'w') as archive:
            for name,content in files.items():
                if name!=missing:archive.writestr('NppAI-x64/NppAI/'+name,content)

    def test_complete_package_and_missing_helper(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'package.zip';self.write(path)
            inspect_package(path,current=True)
            self.write(path,missing='sandbox_validate.py')
            with self.assertRaises(ValueError):inspect_package(path,current=True)

    def test_invalid_validator_is_not_accepted(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'package.zip';self.write(path,helper='def broken(')
            with self.assertRaises(py_compile.PyCompileError):inspect_package(path,current=True)


if __name__=='__main__':unittest.main()
