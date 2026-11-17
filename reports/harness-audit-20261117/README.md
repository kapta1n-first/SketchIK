# HarnessVLA reproduction audit, 2026-11-17

Astra xhigh reviewed the native entry point, tool registration, visual payloads, tool-result encoding, transport adaptation, memory selection, and timing. No online probes or new simulations were launched for this audit.

The completed/ongoing Sol suite used native revision `0e406bb96d670d0c18a4cea9a6ea047119661d19`, HF memory, the Flowith XML bridge, and a native 1800-second wall timeout. It must not be relabeled as an empty-memory run. See [audit.json](audit.json).

The next-run configuration is **native local prompt + empty external memory + Sol low + Flowith XML bridge + 1800-second effective budget**. It is not strict zero-shot: the unchanged native prompt retains task-specific seed0 advice and thresholds. Empty local memory removes the external corpus, not these built-in instructions. The earlier DS runs used this local empty-directory approach, but different model, transport, output limits, and time budgets.

The XML bridge remains active. Offline checks found no simple tool duplication or loss of messages/images in the tested conversion; they do not establish behavioral equivalence to provider-native tool calling. Raw model completion tracing is enabled for the next dedicated gateway. Removing the bridge requires a model endpoint accepting structured tools directly; no verified native Sol endpoint was established during this audit. Do not silently deduplicate or rewrite model actions.

[Corrected historical timing](corrected-timing.json) excludes only confirmed failed network attempts and actual retry backoff. Unknown client-cancelled pending waits remain counted. Raw results and videos are unchanged. Historical native wall-time deadlines cannot be extended retroactively.

The versioned external launcher and offline checks are in [tools/harness_eval](../../tools/harness_eval). Neither the sibling native repository nor the active frozen run was modified.
