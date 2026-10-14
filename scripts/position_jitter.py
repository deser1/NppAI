"""Training-only whitespace offsets with an independent deterministic RNG."""
import hashlib
import json
import random


class PositionJitter:
    def __init__(self, seed=4201):
        self.seed = seed
        self._rng = random.Random(seed)
        self._trace = hashlib.sha256()
        self._views = 0

    def apply(self, record):
        leading = self._rng.randint(0, 32)
        trailing = self._rng.randint(0, 16)
        view = {**record, "prompt": " " * leading + record["prompt"] + "\n" * trailing}
        trace = {"id": view["id"], "prompt": view["prompt"], "answer": view["answer"]}
        self._trace.update(json.dumps(trace, sort_keys=True, ensure_ascii=False).encode("utf-8") + b"\n")
        self._views += 1
        return view

    def metadata(self):
        return {"seed": self.seed, "leading_spaces": [0, 32], "trailing_newlines": [0, 16],
                "training_views": self._views, "ordered_view_sha256": self._trace.hexdigest(),
                "method": "uniform independent offsets per sampled training view; separate RNG preserves source batch schedule"}
