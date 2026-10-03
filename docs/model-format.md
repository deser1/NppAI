# NppAI Model Format

The `.nppai` file currently starts with five 32-bit integer configuration values:

```text
dim
hidden_dim
n_layers
max_seq_len
vocab_size
```

They are followed by the model tensors in a fixed order matching the C++ loader and Python validation script.

## Compatibility

The format is currently an internal experimental format.

A future version should add:

- magic bytes;
- format version;
- endianness marker;
- tensor metadata;
- checksum/hash;
- explicit quantization metadata.

The loader currently validates header dimensions and requires the payload length
to exactly match the tensor layout implied by the header. Truncated payloads and
files with trailing data are rejected before tensor allocation begins.

Until then, model files should be treated as version-specific artifacts and not as a stable public interchange format.
