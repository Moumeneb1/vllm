"""Load hash-checked public PCM audio and immutable benchmark case lists."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import soundfile as sf


def load_samples(config, normalize):
    path = Path(config["manifest"])
    assert hashlib.sha256(path.read_bytes()).hexdigest() == config["manifest_sha256"]
    manifest = json.loads(path.read_text())
    root = Path(os.environ.get("VOXTRAL_VALIDATION_AUDIO_ROOT", config["audio_root"]))
    samples = []
    for item in manifest["assets"]:
        audio_path = root / item["file"]
        with audio_path.open("rb") as stream:
            assert hashlib.file_digest(stream, "sha256").hexdigest() == item["sha256"]
        audio, rate = sf.read(audio_path, dtype="int16")
        assert rate == 16000 and audio.ndim == 1 and len(audio) == item["samples"]
        assert len(audio) / rate * 12.5 + 100 < config["max_model_len"]
        reference = item["reference"]
        samples.append(
            {
                "index": item["index"],
                "duration": len(audio) / rate,
                "reference": None if reference is None else normalize(reference),
                "pcm": audio.astype(np.dtype("<i2"), copy=False).tobytes(),
                "name": item["name"],
            }
        )
    assert all(sample["index"] == i for i, sample in enumerate(samples))
    for case in config["cases"]:
        assert case["mode"] in ("native", "graphs", "batched")
        assert 1 <= case["concurrency"] <= config["max_num_seqs"]
        assert len(case["indices"]) >= case["concurrency"]
        assert all(0 <= i < len(samples) for i in case["indices"])
        assert all(isinstance(n, int) and n > 0 for n in case["transport_samples"])
    return samples, manifest
