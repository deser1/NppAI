"""PyTorch-only attention ablation. Never export these as native NPPAI models."""


def configuration(arm, task):
    if arm not in ("legacy1", "absolute4", "rotary4") or task not in ("headers", "bodies"):
        raise ValueError("unknown preselected attention arm or task")
    return {"format": "torch-attention-probe-v1", "arm": arm, "task": task,
            "heads": 1 if arm == "legacy1" else 4,
            "position": "rotary" if arm == "rotary4" else "learned_absolute",
            "dimension": 64, "hidden_dimension": 128, "layers": 2,
            "context": 256, "vocabulary": 256}


def validate_configuration(config):
    if config != configuration(config.get("arm"), config.get("task")):
        raise ValueError("incompatible experimental checkpoint configuration")


def rotate_pairs(x, offset=0):
    import torch
    length, width = x.shape[-2:]
    if width % 2:
        raise ValueError("rotary head width must be even")
    frequency = 10000.0 ** (-torch.arange(0, width, 2, device=x.device, dtype=x.dtype) / width)
    angles = (torch.arange(length, device=x.device, dtype=x.dtype) + offset)[:, None] * frequency
    cosine, sine = angles.cos(), angles.sin()
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((even * cosine - odd * sine, even * sine + odd * cosine), dim=-1).flatten(-2)


def build_model(config):
    validate_configuration(config)
    import torch
    import torch.nn.functional as F
    from train_nppai import NppAIModel

    class MultiHeadProbe(NppAIModel):
        def forward(self, tokens):
            batch, length = tokens.shape
            x = self.tok_embeddings(tokens)
            if config["position"] == "learned_absolute":
                x = x + self.pos_embeddings(torch.arange(length, device=tokens.device))
            mask = torch.tril(torch.ones(length, length, device=tokens.device, dtype=torch.bool))
            for layer in self.layers:
                h = layer.rms_attn(x)
                q, k, v = [projection(h).reshape(batch, length, 4, 16).transpose(1, 2)
                           for projection in (layer.wq, layer.wk, layer.wv)]
                if config["position"] == "rotary":
                    q, k = rotate_pairs(q), rotate_pairs(k)
                scores = (q @ k.transpose(-2, -1)) / (16 ** 0.5)
                attention = F.softmax(scores.masked_fill(~mask, float("-inf")), dim=-1)
                combined = (attention @ v).transpose(1, 2).contiguous().reshape(batch, length, 64)
                x = x + layer.wo(combined)
                h = layer.rms_ffn(x)
                x = x + layer.w3(F.silu(layer.w1(h)) * layer.w2(h))
            return self.output(self.norm(x))

    model_type = NppAIModel if config["arm"] == "legacy1" else MultiHeadProbe
    model = model_type(256, 64, 128, 2, 256)
    if config["position"] == "rotary":
        # Retained for identical initialization order/storage; unused by forward.
        model.pos_embeddings.weight.requires_grad_(False)
    return model


def save_model(model, config, path):
    import torch
    validate_configuration(config)
    torch.save({"config": config, "state_dict": model.state_dict()}, path)


def load_model(path):
    import torch
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    model = build_model(config)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.eval(), config
