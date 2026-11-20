"""Exercise real Realtime WebSocket serving with public audio and pinned weights."""

import argparse
import asyncio
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import socket
import statistics
import subprocess
import sys
import time
import traceback

import aiohttp
from rapidfuzz.distance import Levenshtein
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
import websockets
import vllm

from matrix_support import load_samples


def percentile(values, fraction):
    ordered = sorted(values)
    point = (len(ordered) - 1) * fraction
    low = int(point)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (point - low)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("native", "batched", "graphs"), default="native"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    previous = Path(os.environ["VOXTRAL_NORMALIZER_ROOT"])
    config = json.loads(args.config.read_text())
    if args.root.name.startswith(("r0_", "r1_")):
        assert config["variant"] == args.root.name.split("_", 1)[1], "wrong variant configuration"
    assert os.environ["OMP_NUM_THREADS"] == str(config["omp_num_threads"])
    normalize = EnglishTextNormalizer(
        json.loads(
            (previous / "public_data/whisper_english_normalizer.json").read_text()
        )
    )
    samples, manifest = load_samples(config, normalize)
    source_root = Path(os.environ["VOXTRAL_LOCAL_SOURCE"])
    source_manifest = json.loads((args.root / "source_manifest.json").read_text())
    assert config["variant"] == source_manifest["variant"], "source/config variant mismatch"
    assert Path(vllm.__file__).is_relative_to(source_root)
    for filename, digest in source_manifest["source_files_sha256"].items():
        assert (
            hashlib.sha256((source_root / filename).read_bytes()).hexdigest() == digest
        )
    result = {
        "status": "running",
        "mode": args.mode,
        "phase": "validation",
        "validation_config": config,
        "job": os.environ.get("SLURM_JOB_ID"),
        "vllm_commit": source_manifest["commit"],
        "source_archive_sha256": source_manifest["archive_sha256"],
        "source_files_verified": True,
        "dataset_manifest_sha256": config["manifest_sha256"],
        "bank": [{k: v for k, v in item.items() if k != "pcm"} for item in samples],
        "scope": "Original BF16 weights and existing core graphs; variant source hashes identify native/current-PR/private-candidate frontend. No sampling hooks or model monkeypatching. Real WebSocket requests; paced cases separate.",
        "variant": config["variant"],
        "effective_source_manifest": source_manifest,
        "cases": [],
    }

    def save():
        temporary = args.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(result, indent=2) + "\n")
        temporary.replace(args.output)

    save()
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    address = f"http://127.0.0.1:{port}"
    compilation = {"cudagraph_mode": "FULL_AND_PIECEWISE", "cudagraph_mm_encoder": args.mode == "graphs"}
    if config.get("frontend_max_batch") is not None:
        compilation["encoder_cudagraph_max_vision_items_per_batch"] = config["frontend_max_batch"]
    trace_dir = args.stage / f"traces_{args.mode}"
    profiler = {
        "profiler": "torch",
        "torch_profiler_dir": str(trace_dir),
        "torch_profiler_use_gzip": True,
        "torch_profiler_with_stack": False,
        "torch_profiler_record_shapes": False,
        "detailed_trace_annotation": True,
    }
    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        args.model,
        "--served-model-name",
        "voxtral-realtime",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--tokenizer-mode",
        "mistral",
        "--config-format",
        "mistral",
        "--load-format",
        "mistral",
        "--max-model-len",
        str(config["max_model_len"]),
        "--max-num-seqs",
        str(config["max_num_seqs"]),
        "--max-num-batched-tokens",
        "4096",
        "--gpu-memory-utilization",
        ".75",
        "--mm-processor-cache-gb",
        "0",
        "--no-enable-prefix-caching",
        "--seed",
        "0",
        "--worker-extension-cls",
        "pr_probe.PRProbe",
        "--compilation-config",
        json.dumps(compilation),
        "--profiler-config",
        json.dumps(profiler),
    ]
    environment = dict(os.environ, VLLM_SERVER_DEV_MODE="1")
    result["command"] = command
    server_log = args.output.with_suffix(".server.log")
    result["server_log"] = str(server_log)
    process = None
    startup_started = time.monotonic()
    try:
        with server_log.open("w") as log:
            process = subprocess.Popen(
                command,
                cwd=os.environ.get("VOXTRAL_LOCAL_SOURCE", str(Path.cwd())),
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=900)
            ) as client:

                async def rpc(method, **kwargs):
                    async with client.post(
                        address + "/collective_rpc",
                        json={"method": method, "kwargs": kwargs},
                    ) as response:
                        body = await response.json()
                        assert response.status == 200, body
                        return body["results"]

                deadline = time.monotonic() + 1500
                while True:
                    if process.poll() is not None:
                        raise RuntimeError(
                            f"server exited: {server_log.read_text()[-12000:]}"
                        )
                    try:
                        async with client.get(address + "/health") as response:
                            if response.status == 200:
                                break
                    except (aiohttp.ClientError, TimeoutError):
                        pass
                    assert time.monotonic() < deadline, "server startup timeout"
                    await asyncio.sleep(0.5)
                result["info"] = await rpc("realtime_info")
                if config["variant"] == "unbatched":
                    assert config["frontend_max_batch"] == 1
                    assert result["info"][0]["frontend_capture_axes"][0] == [1], "cap-one graphs not active"
                result["server_startup_seconds"] = time.monotonic() - startup_started
                capture_started = time.monotonic()
                result["frontend_setup"] = await rpc("frontend_stats")
                result["frontend_capture_seconds"] = time.monotonic() - capture_started
                result["pr_integration_status"] = (
                    "production adapter: existing startup graph capture and memory accounting"
                )
                print(json.dumps({"ready": True, "info": result["info"]}), flush=True)
                save()

                async def transcribe(sample, position, paced=False, packets=(1280,)):
                    started = time.perf_counter()
                    first_delta = audio_end = None
                    deltas = []
                    async with websockets.connect(
                        f"ws://127.0.0.1:{port}/v1/realtime",
                        max_size=2**24,
                        open_timeout=30,
                        ping_timeout=120,
                    ) as ws:
                        created = json.loads(await ws.recv())
                        assert created["type"] == "session.created", created
                        await ws.send(
                            json.dumps(
                                {"type": "session.update", "model": "voxtral-realtime"}
                            )
                        )
                        await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))

                        async def send_audio():
                            nonlocal audio_end
                            upload_start = time.perf_counter()
                            offset = packet_index = 0
                            while offset < len(sample["pcm"]):
                                packet_bytes = 2 * packets[packet_index % len(packets)]
                                if paced:
                                    delay = (
                                        upload_start
                                        + offset / 32000
                                        - time.perf_counter()
                                    )
                                    if delay > 0:
                                        await asyncio.sleep(delay)
                                await ws.send(
                                    json.dumps(
                                        {
                                            "type": "input_audio_buffer.append",
                                            "audio": base64.b64encode(
                                                sample["pcm"][
                                                    offset : offset + packet_bytes
                                                ]
                                            ).decode(),
                                        }
                                    )
                                )
                                offset += packet_bytes
                                packet_index += 1
                            if paced:
                                delay = (
                                    upload_start
                                    + sample["duration"]
                                    - time.perf_counter()
                                )
                                if delay > 0:
                                    await asyncio.sleep(delay)
                            audio_end = time.perf_counter()
                            await ws.send(
                                json.dumps(
                                    {"type": "input_audio_buffer.commit", "final": True}
                                )
                            )

                        sender = asyncio.create_task(send_audio())
                        try:
                            while True:
                                event = json.loads(
                                    await asyncio.wait_for(ws.recv(), timeout=600)
                                )
                                if event["type"] == "error":
                                    raise RuntimeError(event)
                                if event["type"] == "transcription.delta":
                                    deltas.append(event["delta"])
                                if (
                                    event["type"] == "transcription.delta"
                                    and event.get("delta", "").strip()
                                    and first_delta is None
                                ):
                                    first_delta = time.perf_counter()
                                if event["type"] == "transcription.done":
                                    await sender
                                    ended = time.perf_counter()
                                    text = event["text"]
                                    assert "".join(deltas) == text
                                    assert len(deltas) == event["usage"]["completion_tokens"]
                                    return {
                                        "deltas": deltas if config.get("capture_deltas") else None,
                                        "position": position,
                                        "index": sample["index"],
                                        "duration": sample["duration"],
                                        "reference": sample["reference"],
                                        "text": text,
                                        "seconds": ended - started,
                                        "first_text_seconds": None
                                        if first_delta is None
                                        else first_delta - started,
                                        "after_audio_end_seconds": ended - audio_end,
                                        "usage": event.get("usage"),
                                    }
                        finally:
                            if not sender.done():
                                sender.cancel()
                            await asyncio.gather(sender, return_exceptions=True)

                async def run(
                    items,
                    concurrency,
                    label,
                    paced=False,
                    audit=False,
                    packets=(1280,),
                    pair=None,
                    warmup=False,
                ):
                    assert args.mode == expected_mode
                    previous_stats = await rpc("frontend_stats")
                    queue = asyncio.Queue()
                    for position, item in enumerate(items):
                        queue.put_nowait((position, item))
                    outputs = []
                    started = time.perf_counter()

                    async def worker():
                        while not queue.empty():
                            position, item = queue.get_nowait()
                            outputs.append(
                                await transcribe(item, position, paced, packets)
                            )

                    await asyncio.gather(*(worker() for _ in range(concurrency)))
                    seconds = time.perf_counter() - started
                    outputs.sort(key=lambda item: item["position"])
                    for item in outputs:
                        item["word_errors"] = (
                            None
                            if item["reference"] is None
                            else Levenshtein.distance(
                                item["reference"].split(),
                                normalize(item["text"]).split(),
                            )
                        )
                        usage = item.get("usage") or {}
                        assert usage.get("total_tokens", 0) >= math.floor(
                            item["duration"] * 12.5
                        ), "audio stream ended early"
                        assert usage.get("total_tokens", 0) < config["max_model_len"]
                    diagnostics = []
                    frontend = await rpc("frontend_stats")
                    if args.mode == "graphs":
                        assert sum(rank["graph_hits"] for rank in frontend) > sum(rank["graph_hits"] for rank in previous_stats)
                    scored = [item for item in outputs if item["reference"] is not None]
                    words = sum(len(item["reference"].split()) for item in scored)
                    errors = sum(item["word_errors"] for item in scored)
                    row = {
                        "label": label,
                        "pair": pair,
                        "warmup": warmup,
                        "transport_samples": packets,
                        "frontend_mode": args.mode,
                        "concurrency": concurrency,
                        "paced": paced,
                        "seconds": seconds,
                        "requests": len(outputs),
                        "requests_per_second": len(outputs) / seconds,
                        "audio_seconds": sum(item["duration"] for item in outputs),
                        "wer": 100 * errors / words
                        if words and len(scored) == len(outputs)
                        else None,
                        "scored_requests": len(scored),
                        "reference_words": words,
                        "word_errors": errors if len(scored) == len(outputs) else None,
                        "audio_seconds_per_second": sum(
                            item["duration"] for item in outputs
                        )
                        / seconds,
                        "p50_seconds": statistics.median(
                            item["seconds"] for item in outputs
                        ),
                        "p95_seconds": percentile(
                            [item["seconds"] for item in outputs], 0.95
                        ),
                        "p99_seconds": percentile(
                            [item["seconds"] for item in outputs], 0.99
                        ),
                        "p50_first_text_seconds": statistics.median(
                            [
                                item["first_text_seconds"]
                                for item in outputs
                                if item["first_text_seconds"] is not None
                            ]
                        )
                        if any(
                            item["first_text_seconds"] is not None for item in outputs
                        )
                        else None,
                        "p95_after_audio_end_seconds": percentile(
                            [item["after_audio_end_seconds"] for item in outputs], 0.95
                        ),
                        "p99_after_audio_end_seconds": percentile(
                            [item["after_audio_end_seconds"] for item in outputs], 0.99
                        ),
                        "p50_after_audio_end_seconds": statistics.median(
                            item["after_audio_end_seconds"] for item in outputs
                        ),
                        "diagnostics": diagnostics,
                        "frontend": frontend,
                        "outputs": outputs,
                    }
                    result["cases"].append(row)
                    print(
                        json.dumps(
                            {
                                key: row[key]
                                for key in (
                                    "label",
                                    "concurrency",
                                    "requests",
                                    "seconds",
                                    "wer",
                                )
                            }
                        ),
                        flush=True,
                    )
                    save()
                    return row

                expected_mode = args.mode
                warm = [samples[i] for i in manifest["pools"]["short_1_3"][:4]]
                await run(warm, 4, f"warmup:{args.mode}", warmup=True)
                for case in config["cases"]:
                    if case["mode"] != args.mode:
                        continue
                    await run(
                        [samples[i] for i in case["indices"]],
                        case["concurrency"],
                        case["label"],
                        paced=case["paced"],
                        packets=case["transport_samples"],
                        pair=case["pair"],
                    )
                result["final_memory"] = await rpc("realtime_info")
                result["status"] = "complete"
                save()
    except BaseException:
        result["status"] = "failed"
        result["error"] = traceback.format_exc()
        save()
        raise
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=30)


if __name__ == "__main__":
    asyncio.run(main())
