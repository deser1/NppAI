import hashlib
import importlib.util
import struct
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]

with mock.patch.dict("sys.modules", {"torch_directml": None}):
    spec = importlib.util.spec_from_file_location("train_nppai", ROOT / "train_nppai.py")
    train_nppai = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_nppai)


class ModelFormatV3Tests(unittest.TestCase):
    def setUp(self):
        self.model = SimpleNamespace(
            dim=16, hidden_dim=32, n_layers=2, max_seq_len=128, vocab_size=512
        )

    def test_dense_v3_header_layout_and_hash(self):
        payload = b"payload"
        header = train_nppai.build_v3_header(self.model, payload)
        self.assertEqual(len(header), 84)
        self.assertEqual(header[:8], b"NPPAI\0\0\0")
        self.assertEqual(struct.unpack_from("<I", header, 8)[0], 3)
        self.assertEqual(struct.unpack_from("<5i", header, 12), (16, 32, 2, 128, 512))
        self.assertEqual(struct.unpack_from("<3I", header, 32), (0, 0, 0))
        self.assertEqual(struct.unpack_from("<Q", header, 44)[0], len(payload))
        self.assertEqual(header[52:84], hashlib.sha256(payload).digest())

    def test_valid_moe_metadata_is_accepted(self):
        train_nppai.validate_model_format_metadata(
            train_nppai.MODEL_ARCH_MOE, 8, 2
        )

    def test_invalid_moe_metadata_is_rejected(self):
        for values in [(1, 0, 1), (1, 4, 0), (1, 4, 5), (7, 0, 0), (0, 1, 0)]:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    train_nppai.validate_model_format_metadata(*values)

    def test_moe_export_fails_closed_until_tensor_layout_exists(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "moe.nppai"
            with self.assertRaises(NotImplementedError):
                train_nppai.export_v3_metadata_model(
                    self.model, path, b"unsafe",
                    architecture=train_nppai.MODEL_ARCH_MOE,
                    num_experts=4, experts_per_token=2,
                )
            self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
