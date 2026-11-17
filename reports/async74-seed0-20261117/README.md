# Protocol74 asynchronous seed0/init0 results

Ten complete-initial-state trials, Tasks 0–9. Sol 6.1 low, 65536 maximum completion tokens, IK-only; 20 sketch batches and 30 execution A requests per task.

Official result: **8/10**. A reported success for all ten. Official results, model claims and human visual judgments are separate; no evaluation result is rewritten. Effective total 4345.750 seconds; raw total 4373.036 seconds, including 27.286 seconds of Task8 network interruption. This is a different suite from the earlier synchronous suite accepted by the user as 10/10.

| Task | Official success | A status | Effective seconds | Logs | Video |
|---|---|---|---:|---|---|
| 0 | True | success | 422.556 | [logs](task0) | [video](https://github.com/kapta1n-first/SketchIK/releases/download/async74-seed0-20261117/task0.mp4) |
| 1 | True | success | 412.968 | [logs](task1) | [video](https://github.com/kapta1n-first/SketchIK/releases/download/async74-seed0-20261117/task1.mp4) |
| 2 | True | success | 291.010 | [logs](task2) | [video](https://github.com/kapta1n-first/SketchIK/releases/download/async74-seed0-20261117/task2.mp4) |
| 3 | True | success | 507.975 | [logs](task3) | [video](https://github.com/kapta1n-first/SketchIK/releases/download/async74-seed0-20261117/task3.mp4) |
| 4 | True | success | 344.668 | [logs](task4) | [video](https://github.com/kapta1n-first/SketchIK/releases/download/async74-seed0-20261117/task4.mp4) |
| 5 | False | success | 267.277 | [logs](task5) | [video](https://github.com/kapta1n-first/SketchIK/releases/download/async74-seed0-20261117/task5.mp4) |
| 6 | False | success | 528.206 | [logs](task6) | [video](https://github.com/kapta1n-first/SketchIK/releases/download/async74-seed0-20261117/task6.mp4) |
| 7 | True | success | 345.701 | [logs](task7) | [video](https://github.com/kapta1n-first/SketchIK/releases/download/async74-seed0-20261117/task7.mp4) |
| 8 | True | success | 367.461 | [logs](task8) | [video](https://github.com/kapta1n-first/SketchIK/releases/download/async74-seed0-20261117/task8.mp4) |
| 9 | True | success | 857.928 | [logs](task9) | [video](https://github.com/kapta1n-first/SketchIK/releases/download/async74-seed0-20261117/task9.mp4) |

Each task contains conversation logs, events, original submitted actions and execution receipts, asynchronous candidate/continuation records, timings, source hashes, configuration and final native-resolution RGB images. See manifest.json for included-file hashes and sanitation. Dense point clouds, raw RGB-D archives, runtime credentials and raw HTTP transport payloads remain local; account balance fields are omitted. Absolute runtime paths inside the preserved logs refer to the original run and are historical references, not links into this export.
