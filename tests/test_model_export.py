import hashlib
import struct
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
import model_format


class ModelFormatV3Tests(unittest.TestCase):
    def setUp(self):
        self.model = SimpleNamespace(
            dim=16, hidden_dim=32, n_layers=2, max_seq_len=128, vocab_size=512
        )

    def test_dense_v3_header_layout_and_hash(self):
        payload = b"payload"
        header = model_format.build_v3_header(self.model, payload)
        self.assertEqual(len(header), 84)
        self.assertEqual(header[:8], b"NPPAI\0\0\0")
        self.assertEqual(struct.unpack_from("<I", header, 8)[0], 3)
        self.assertEqual(struct.unpack_from("<5i", header, 12), (16, 32, 2, 128, 512))
        self.assertEqual(struct.unpack_from("<3I", header, 32), (0, 0, 0))
        self.assertEqual(struct.unpack_from("<Q", header, 44)[0], len(payload))
        self.assertEqual(header[52:84], hashlib.sha256(payload).digest())

    def test_valid_moe_metadata_is_accepted(self):
        model_format.validate_model_format_metadata(
            model_format.MODEL_ARCH_MOE, 8, 2
        )

    def test_invalid_moe_metadata_is_rejected(self):
        for values in [(1, 0, 1), (1, 4, 0), (1, 4, 5), (7, 0, 0), (0, 1, 0)]:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    model_format.validate_model_format_metadata(*values)

    def test_moe_export_fails_closed_until_tensor_layout_exists(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "moe.nppai"
            with self.assertRaises(NotImplementedError):
                model_format.export_v3_metadata_model(
                    self.model, path, b"unsafe",
                    architecture=model_format.MODEL_ARCH_MOE,
                    num_experts=4, experts_per_token=2,
                )
            self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
