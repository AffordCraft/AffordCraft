# Scripts

Entry points of the method. Commands and the order of the experiments are in `docs/running.md`; interpreters and paths
in `docs/setup.md`.

| Script | Interpreter | Purpose |
|---|---|---|
| `build_visual_index.py` | selector | encode every catalog preview (DINOv2, or CLIP for the encoder-replacement condition) into the visual index |
| `prepare_inputs.py` | any | runnable input manifests from `data/inputs` and local image copies (hash-checked) |
| `make_physics_fixtures.py` | runtime or Isaac (pxr) | the seven known positive and negative physics controls |
| `run_physics_job.py` | any | runs the physics worker on a job list (the controls) and keeps process and in-process outcomes apart |
| `prepare_calibration.py`, `calibrate_vision.py` | any, selector | optional grounding check on calibration images disjoint from the test inputs |
| `freeze_experiment.py` | any | frozen definition after the calibrations: inputs hash, policies, study variant, shard count, hashes of all code and model files |
| `run_pipeline.py` | selector | the method on a manifest or one shard of it: grounding, retrieval, selection, construction, construction check, repair, final validation; `--scene-mode` for detections of cluttered images; `--resume-from` after an infrastructure fault |
| `materialize_asset.py` | runtime | one candidate: source-preserving import, adaptation, collision decomposition cascade, USD export (started by `run_pipeline.py`) |
| `physics_worker.py` | Isaac Sim 5.1 | physics service for the construction check and, in a second process, the final validation (started by `run_pipeline.py`) |
| `coacd_health.py` | runtime | known-positive health check of the collision-decomposition library (run before a shard and after a native crash) |
| `merge_shards.py` | any | one run directory from the final attempt of every shard |
| `export_final_results.py` | any | pass rate with its Wilson interval, statuses and per-category counts of a complete run |
| `detect_scene_objects.py` | selector | object discovery in whole cluttered images (no reference boxes) |
| `check_retained_invariance.py` | selector | shows that the detection rules leave retained first-pass outputs unchanged |
| `evaluate_scenes.py` | any | scores constructed detections against the 237 reference objects (same category, one-to-one, IoU >= 0.5) |
