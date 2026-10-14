"""Fail closed on incompatible copying-stage checkpoints."""


def validate_initial_weights(weights):
    actual = (weights["dim"], weights["hidden"], weights["context"],
              len(weights["layers"]), len(weights["embedding"]))
    if actual != (64, 128, 256, 2, 256):
        raise ValueError("initial checkpoint must be dense byte 64/128, two layers, context 256")
