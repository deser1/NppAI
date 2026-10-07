# NppAI Model Format

NppAI uses a versioned binary model container. New exports use **format v2**.
The loader still accepts the original legacy v1 layout so existing local models
continue to work.

## Format v2

All integer fields are little-endian. The fixed 72-byte header is:

```text
offset  size  field
0       8     magic = "NPPAI\\0\\0\\0"
8       4     format_version = 2 (uint32)
12      4     dim (int32)
16      4     hidden_dim (int32)
20      4     n_layers (int32)
24      4     max_seq_len (int32)
28      4     vocab_size (int32)
32      8     payload_size_bytes (uint64)
40      32    SHA-256 of the complete tensor payload
72      ...   FP32 tensor payload
```

The tensor payload order is unchanged from v1 and matches the C++ loader and
`train_nppai.py` exporter.

## Format v3 architecture metadata

Format v3 extends the v2 header by inserting three little-endian `uint32`
fields after `vocab_size` and before `payload_size_bytes`:

```text
offset  size  field
0       8     magic = "NPPAI\\0\\0\\0"
8       4     format_version = 3
12      20    dim, hidden_dim, n_layers, max_seq_len, vocab_size
32      4     architecture (0 = dense, 1 = MoE)
36      4     num_experts
40      4     experts_per_token
44      8     payload_size_bytes
52      32    SHA-256 of the complete tensor payload
84      ...   tensor payload
```

Metadata invariants are:

- dense: `architecture=0`, `num_experts=0`, `experts_per_token=0`;
- MoE: `architecture=1`, `num_experts >= 1`, and
  `1 <= experts_per_token <= num_experts`;
- unknown architecture values are rejected;
- nonzero expert fields on a dense model are rejected.

The first MoE implementation will use `num_experts` as the number of FFN
experts per transformer layer and `experts_per_token` as deterministic top-k.
Router/expert tensor ordering is intentionally deferred until the routing and
inference implementation lands. A v3 file that declares MoE metadata must not
be interpreted as a dense v2 payload.

This separation makes the metadata contract explicit before MoE inference is
enabled and prevents partially implemented models from being silently loaded.

## Integrity and validation

Before allocating model tensors, the loader:

1. validates magic/version for v2;
2. validates dimension and layer limits;
3. computes the exact payload size implied by the dimensions;
4. requires both the physical file length and v2 `payload_size_bytes` to match;
5. computes SHA-256 over the payload and compares it with the digest in the header.

A truncated file, trailing data, unsupported v2 version, or payload whose bytes
do not match the SHA-256 digest is rejected.

The SHA-256 field is an integrity check, not a digital signature. It detects
corruption and accidental modification but does not establish who produced a
model. Authenticity would require a future signed-model mechanism.

## Legacy v1 compatibility

Legacy files contain five native 32-bit integer dimensions followed immediately
by the FP32 payload:

```text
dim
hidden_dim
n_layers
max_seq_len
vocab_size
payload...
```

Legacy v1 has no embedded checksum. Its dimensions and exact payload length are
still validated. New models should be exported as v2.

## Future evolution

A later format version may add an endianness marker, tensor metadata, explicit
quantization metadata, and cryptographic signatures. Unknown version numbers
must be rejected rather than guessed.
