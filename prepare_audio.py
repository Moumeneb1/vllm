"""Rebuild exactly the benchmark WAVs from pinned public Parquet sources."""

import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import requests
import soundfile as sf


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, help="Optional existing Parquet cache")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    manifest_path = root / "audio_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    cache = args.source_dir or root / "source_data"
    cache.mkdir(exist_ok=True)
    rows = {}
    for name, source in manifest["sources"].items():
        path = cache / f"{name}.parquet"
        if not path.exists():
            api = f"https://huggingface.co/api/datasets/{source['repository']}/tree/{source['revision']}"
            response = requests.get(api, params={"recursive": "true"}, timeout=60)
            response.raise_for_status()
            candidates = [item for item in response.json() if item.get("lfs", {}).get("oid") == source["sha256"]]
            assert len(candidates) == 1, candidates
            url = f"https://huggingface.co/datasets/{source['repository']}/resolve/{source['revision']}/{candidates[0]['path']}"
            with requests.get(url, stream=True, timeout=300) as response:
                response.raise_for_status()
                with path.open("wb") as output:
                    for chunk in response.iter_content(1024 * 1024):
                        output.write(chunk)
        assert digest(path) == source["sha256"], path
        rows[name] = pq.read_table(path).to_pylist()
    audio_root = root / "public_audio"
    audio_root.mkdir(exist_ok=True)
    for item in manifest["assets"]:
        target = audio_root / item["file"]
        if not target.exists():
            provenance = item["provenance"]
            row = rows[provenance["dataset"]][provenance["row"]]
            audio, rate = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
            if "start_sample" in provenance:
                audio = audio[provenance["start_sample"] : provenance["end_sample"]]
            assert rate == 16000 and len(audio) == item["samples"]
            pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2")
            sf.write(target, pcm, rate, subtype="PCM_16")
        assert digest(target) == item["sha256"], target
    for name in ("benchmark", "quality", "focus"):
        config = json.loads((root / f"{name}_template.json").read_text())
        config.update(manifest=str(manifest_path), manifest_sha256=digest(manifest_path), audio_root=str(audio_root))
        (root / f"{name}.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"Verified {len(manifest['assets'])} public WAV files")


if __name__ == "__main__":
    main()
