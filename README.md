# Voxtral Realtime audio embedding graph reproduction

This bundle uses the same public audio, request lists, WebSocket client, graph instrumentation and WER normalization as the A100 rerun. It contains no model weights or source audio. It checks model source hashes before each server start and changes no model methods.

Use a managed CUDA environment with Python 3.12.13, Torch 2.14.0+cu130, Transformers 5.22.0, FlashInfer 0.7.3 and Humming 0.1.19. The measurements used an A100-SXM4-80GB, TP1, 16 assigned CPU cores and `OMP_NUM_THREADS=1`.

Prepare two working vLLM source trees, with compatible compiled extensions available in each:

- Baseline/prior PR: `4233576700a4acc90da17bab2f8dc363f0b85c3f`.
- Simplified: `8349b6b3ed79c9c873d727f32ce71812c71c9a3f` (includes Andy Lo's `e3a0be385e2a` refactor plus the normalization regression and docs-link restoration).

The revisions have identical native code, build inputs and requirements. The validation reused precompiled native artifacts after checking that compatibility; it did not rebuild native kernels. The compatible official wheel is from upstream commit `827f2ccfafd92757955a8c8a9c3600b99e18734c` (see `NATIVE_BUILD.json`). Build an editable installation in each source tree with the runtime versions above:

```bash
VLLM_USE_PRECOMPILED=1 \
VLLM_PRECOMPILED_WHEEL_VARIANT=cu130 \
VLLM_PRECOMPILED_WHEEL_COMMIT=827f2ccfafd92757955a8c8a9c3600b99e18734c \
uv pip install -e . --torch-backend=auto
```

Install the client dependencies in that managed environment:

```bash
uv pip install soundfile pyarrow requests websockets rapidfuzz aiohttp
```

Reconstruct and hash-check the public 16 kHz mono PCM16 audio. Dataset repositories, revisions, row IDs, crop boundaries and WAV hashes are pinned in `audio_manifest.json`:

```bash
/path/to/.venv/bin/python prepare_audio.py
```

Use the official `mistralai/Voxtral-Mini-4B-Realtime-2602` snapshot at revision `2769294da9567371363522aac9bbcfdd19447add`. Its `consolidated.safetensors` SHA256 is `263f178fe752c90a2ae58f037a95ed092db8b14768b0978b8c48f66979c8345d`. Pass the local snapshot directory below.

Run the four paired throughput rounds on one A100. Alternatively, use separate A100 allocations for each round while preserving the same device within each three-version comparison. The baseline and prior PR may use the same source tree; their audio-graph flag differs:

```bash
/path/to/.venv/bin/python run_matrix.py \
  --model /path/to/model-snapshot \
  --baseline-source /path/to/baseline-vllm \
  --prior-source /path/to/baseline-vllm \
  --source /path/to/simplified-vllm \
  --config benchmark.json --rounds 4 --output capacity_results
```

Native/prior/simplified order is reversed in odd rounds. The matrix contains 1–3 s, 15–30 s and mixed audio at concurrency 1/8/32/64, with 64/64/128/256 requests per cell and variant: 18,432 measured requests. Eighty-millisecond PCM packets are sent faster than real time for capacity. The original transformer encoder/decoder graphs remain enabled in every mode. `max_num_seqs=64`, `max_model_len=16384`, `max_num_batched_tokens=4096`, `gpu_memory_utilization=0.75`, prefix caching and processor caching disabled, seed 0.

Run the quality and streaming controls separately, on the same A100 for all three versions:

```bash
/path/to/.venv/bin/python run_matrix.py \
  --model /path/to/model-snapshot \
  --baseline-source /path/to/baseline-vllm \
  --prior-source /path/to/baseline-vllm \
  --source /path/to/simplified-vllm \
  --config quality.json --rounds 1 --all-cases --output quality_results
```

The quality matrix repeats the same 511 Earnings22 clips three times at C32 and once at C1 per version. These are repeated measurements, not additional independent recordings. Long 60/180/300 s and irregular-packet controls have no aligned WER references; their WER is excluded. Paced C32/60 s streaming runs are latency controls, separate from capacity. Total quality/streaming requests: 6,510.

Each result includes source hashes, actual GPU UUID, graph hits, every transcript, streamed deltas for quality cases, normalized word errors, per-request latency, startup and KV-memory measurements. Compare matching request identities, token counts and devices; verify graph hits and delta/final-text agreement before aggregating. Report the geometric mean and per-round range. CUDA-graph batching can change floating-point rounding and greedy text; neither normalization independence nor a flat WER establishes universal bitwise equivalence.

The additional short-C32 robustness check is described separately in the results: two AB/BA pairs with 1,024 requests per cell on another A100 host. It retains the primary four-round results, including the original outlier.

```bash
/path/to/.venv/bin/python run_matrix.py \
  --model /path/to/model-snapshot \
  --baseline-source /path/to/baseline-vllm \
  --prior-source /path/to/baseline-vllm \
  --source /path/to/simplified-vllm \
  --config focus.json --rounds 2 --graphs-only-comparison --output focus_results
```

Regression checks in the simplified checkout:

```bash
.venv/bin/python -m pytest tests/v1/cudagraph/test_encoder_cudagraph.py -v
.venv/bin/python -m pytest tests/models/multimodal/generation/test_voxtral_realtime.py -m core_model -v
```

This reproduction wrapper adds source selection and portable paths around the measured serving client. It has CPU validation; the A100 rerun used the same serving client through the cluster launcher. No additional GPU measurement is claimed for the wrapper itself.
