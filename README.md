# Voxtral Realtime frontend graph validation

Evidence for PR source commit `2195f1c7e5871275b9a7add601944ab41ad653d1`, based on upstream `dce91fea6b13c99b8fa91d9f58673e7123c6dc65`.

[Download the reproduction bundle](voxtral_realtime_frontend_reproduction_20261109.zip?raw=true). It contains pinned public audio provenance, reconstruction and serving scripts, the patch, source hashes and audited metrics. No audio or model weights are bundled. Follow the README inside the archive.

Bundle SHA256: `e4088e36acb35ec1dd2815573d77653defb4f36b736899c2a0e5cc58f91de7f3`.

A100 80 GB, BF16, TP1, 64 maximum sequences; 16 assigned CPU cores and OMP_NUM_THREADS=1. Four balanced paired capacity rounds across C1/8/32/64 and three duration pools; 12,288 measured capacity requests. Separate 511-clip quality pass per mode and three paired paced C32/60-second rounds. Original encoder/decoder graphs enabled in both modes; only frontend batching/graph flag changes. See FINAL_RESULTS.json for all means, ranges, physical KV-cache costs and quality figures.

Local validation passed: 98 shared-manager tests, three official-model cases, complete manual-stage pre-commit, strict full docs/API build, 538 reconstructed WAV hashes, and measured C8 serving under VLLM_GPU_SYNC_CHECK=error. Hosted upstream CI is separate.

This validation branch is separate from the implementation PR to keep its diff limited to the eight implementation, test and documentation files. AI assistance was used.
