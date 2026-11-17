# External native HarnessVLA experiment launcher

This directory contains our external launch, transport, accounting, and empty-memory helpers. It does not vendor or replace the native HarnessVLA planner, prompts, robot tools, or environment.

Prepared configuration: native commit `0e406bb96d670d0c18a4cea9a6ea047119661d19`; `gpt-6.1-sol`, low, 65536 completion limit; seed0; local prompt with a fresh empty external memory directory per task; 1800 seconds of effective time. The original local prompt still includes task-specific experience, so this is **empty-external-memory evaluation**, not a claim of strict zero-shot.

The currently configured channel still uses a Flowith XML tool bridge. Private raw completion tracing is enabled for the next dedicated gateway; tracing is not a protocol fix. Credentials and runtime configuration stay outside Git. An existing native snapshot, prepared native command, model assets, configured gateway binary and credentials are required under `.runtime/harness_original_control_v168` and `.runtime/flowith_probe/private`; this source checkout alone is not a self-contained reproduction bundle. `HARNESS_PYTHON` and `HARNESS_GPU_UUID` override the local dependency interpreter and GPU selection.

Running `python tools/harness_eval/run_seed0.py` only displays the prepared mode and starts no services. A separately authorized launch uses `--execute`; the default is **Task0 only**, then stop for review. `HARNESS_TASKS` selects an explicitly authorized comma-separated task list, and `HARNESS_OUTPUT_DIR` / `HARNESS_STATE_FILE` select fresh locations. Existing output/state locations are refused. Do not use it to resume a prior scene. A genuinely live transport error retries only the same frozen request. Authentication, quota, format, and action errors are not network retries.

The native raw wall timer is disabled for this configuration. Effective budget is checked before a new model request and before delivering a complete model response, not by interrupting robot motion. Confirmed failed network attempts and actual retry backoff are excluded. Successful model inference, normal motion, startup and unknown waits count. The existing 100-turn limit and separate transport retry bounds remain. Small boundary overrun is possible; record it honestly.

Offline checks (loopback mock HTTP only; no provider/GPU calls):

```sh
python -m pytest tools/harness_eval/test_empty_memory.py tools/harness_eval/test_network_recovery.py -q
```

See [the audit](../../reports/harness-audit-20261117/README.md) for native configuration differences and limits of the conversion checks.
