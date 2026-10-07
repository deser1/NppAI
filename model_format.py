import hashlib
import struct

MODEL_ARCH_DENSE = 0
MODEL_ARCH_MOE = 1


def validate_model_format_metadata(architecture, num_experts=0, experts_per_token=0):
    if architecture == MODEL_ARCH_DENSE:
        if num_experts != 0 or experts_per_token != 0:
            raise ValueError("Dense v3 metadata requires zero expert fields")
        return
    if architecture == MODEL_ARCH_MOE:
        if num_experts < 1:
            raise ValueError("MoE metadata requires num_experts >= 1")
        if experts_per_token < 1 or experts_per_token > num_experts:
            raise ValueError("MoE metadata requires 1 <= experts_per_token <= num_experts")
        return
    raise ValueError(f"Unknown model architecture: {architecture}")


def build_v3_header(model, payload_bytes, architecture=MODEL_ARCH_DENSE,
                    num_experts=0, experts_per_token=0):
    validate_model_format_metadata(architecture, num_experts, experts_per_token)
    payload_sha256 = hashlib.sha256(payload_bytes).digest()
    return b"".join([
        b"NPPAI\0\0\0",
        struct.pack("<I", 3),
        struct.pack("<5i", model.dim, model.hidden_dim, model.n_layers,
                    model.max_seq_len, model.vocab_size),
        struct.pack("<3I", architecture, num_experts, experts_per_token),
        struct.pack("<Q", len(payload_bytes)),
        payload_sha256,
    ])


def export_v3_metadata_model(model, filepath, payload_bytes=b"",
                             architecture=MODEL_ARCH_DENSE,
                             num_experts=0, experts_per_token=0):
    """Export v3 only when the selected architecture has defined payload semantics."""
    validate_model_format_metadata(architecture, num_experts, experts_per_token)
    if architecture == MODEL_ARCH_MOE:
        raise NotImplementedError(
            "MoE v3 tensor serialization is not defined yet; refusing unsafe export"
        )
    header = build_v3_header(
        model, payload_bytes, architecture, num_experts, experts_per_token
    )
    with open(filepath, "wb") as f:
        f.write(header)
        f.write(payload_bytes)
