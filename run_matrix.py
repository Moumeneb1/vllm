"""Reproduce the pinned three-version Realtime WebSocket comparison."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

BASE = "4233576700a4acc90da17bab2f8dc363f0b85c3f"
HEAD = "8349b6b3ed79c9c873d727f32ce71812c71c9a3f"


def source_manifest(source, expected, variant):
    source = source.resolve()
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=source, text=True
    ).strip()
    assert commit == expected, (commit, expected)
    subprocess.run(
        ["git", "diff", "--exit-code", "HEAD", "--", "vllm"], cwd=source, check=True
    )
    names = subprocess.check_output(
        ["git", "ls-files", "-z", "--", "vllm"], cwd=source
    ).split(b"\0")
    hashes = {
        name.decode(): hashlib.sha256((source / name.decode()).read_bytes()).hexdigest()
        for name in names
        if name.endswith(b".py")
    }
    process = subprocess.Popen(
        ["git", "archive", "--format=tar", "HEAD"], cwd=source, stdout=subprocess.PIPE
    )
    digest = hashlib.sha256()
    for chunk in iter(lambda: process.stdout.read(1024 * 1024), b""):
        digest.update(chunk)
    assert process.wait() == 0
    return {
        "commit": commit,
        "variant": variant,
        "archive_sha256": digest.hexdigest(),
        "source_files_sha256": hashes,
        "local_model_python_patch": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Official pinned local snapshot")
    parser.add_argument(
        "--source", required=True, type=Path, help="Refactored checkout"
    )
    parser.add_argument("--baseline-source", required=True, type=Path)
    parser.add_argument(
        "--prior-source", type=Path, help="Include the previous PR with graphs on"
    )
    parser.add_argument("--config", type=Path, default=Path("benchmark.json"))
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument(
        "--all-cases", action="store_true", help="Keep repeated WER cases together"
    )
    parser.add_argument(
        "--graphs-only-comparison",
        action="store_true",
        help="Compare prior and simplified graph paths",
    )
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    stage = output / "runtime"
    stage.mkdir(exist_ok=True)
    config = json.loads((root / args.config).read_text())
    versions = (
        []
        if args.graphs_only_comparison
        else [("native", args.baseline_source, BASE, "native")]
    )
    if args.graphs_only_comparison:
        assert args.prior_source is not None
    if args.prior_source is not None:
        versions.append(("published", args.prior_source, BASE, "graphs"))
    versions.append(("simplified", args.source, HEAD, "graphs"))
    manifests = {v: source_manifest(s, c, v) for v, s, c, _ in versions}
    for repeat in range(args.rounds):
        ordered = versions if repeat % 2 == 0 else list(reversed(versions))
        for variant, source, _, mode in ordered:
            current = dict(config)
            current["variant"] = variant
            current["cases"] = [
                dict(c, label=f"{c['pair']}:{variant}")
                for c in config["cases"]
                if c["mode"] == mode and (args.all_cases or c["repeat"] == repeat)
            ]
            assert current["cases"]
            destination = output / f"r{repeat}_{variant}"
            destination.mkdir(exist_ok=False)
            (destination / "source_manifest.json").write_text(
                json.dumps(manifests[variant], indent=2) + "\n"
            )
            config_path = destination / "config.json"
            config_path.write_text(json.dumps(current, indent=2) + "\n")
            environment = dict(
                os.environ,
                PYTHONPATH=str(source.resolve()) + os.pathsep + str(root),
                VOXTRAL_LOCAL_SOURCE=str(source.resolve()),
                VOXTRAL_NORMALIZER_ROOT=str(root),
                OMP_NUM_THREADS="1",
                VLLM_USE_RUST_FRONTEND="0",
                TOKENIZERS_PARALLELISM="false",
            )
            subprocess.run(
                [
                    sys.executable,
                    str(root / "serving.py"),
                    "--model",
                    args.model,
                    "--root",
                    str(destination),
                    "--stage",
                    str(stage),
                    "--config",
                    str(config_path),
                    "--mode",
                    mode,
                    "--output",
                    str(destination / "serving.json"),
                ],
                env=environment,
                check=True,
            )


if __name__ == "__main__":
    main()
