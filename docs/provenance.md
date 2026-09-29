# Provenance of the method code

The package and scripts are the frozen code of the paper's runs. Python files were reformatted with black (line length
120; black verifies that the syntax tree is unchanged), and only the changes listed below were made: absolute paths,
interpreters and host-specific defaults became environment variables (`affordcraft/paths.py`), comments that named
internal code versions were reworded, and infrastructure was left out (job queues and controllers, service units,
deployment, registration and verification of our campaigns, manuscript tools). No setting, threshold, seed, prompt or
decision rule was changed.

## Code versions

- **One-factor-study code**: the package in this repository. It ran the paired one-factor study (Table 2).
- **Main-campaign code**: ran the 2,000-input single-object experiment, the cluttered-image experiment and every
  AffordCraft number outside Table 2. The files below differ only by the machinery of the study conditions: the
  `variant` switch (default `full`) with one branch per removed component, the CLIP encoder of the encoder-replacement
  condition, and the variant field of the frozen definition. With `variant="full"` both versions build the same
  prompts and take the same decisions.

| File | sha256 (main-campaign code) | sha256 (one-factor-study code) |
|---|---|---|
| `method/affordcraft/backend.py` | `cda71bcefa83b5b127682d9f6cff0fb9e02eb357a6ae1221463a219e69a2b34a` | `14222b4668b3db45ee716118c881b38476933cb53883ca1d4bf71dbbafd89c06` |
| `method/affordcraft/catalog.py` | `4145fa21a351ed1e0eacc9757dc48d526173ff027587fe5da28351b04348c112` | `1f6f8cb6458e5241ca2ac699e0a170559cc8576c0197d9b7b878998dc754b86e` |
| `method/affordcraft/contracts.py` | `b74558651bf4f9aa63007487af4c8ce0accf687fc97e93108cae3d4be878c9f4` | `a2258b074879222fa427bbaadd468f3ccfc4151c88611af731819a662688caff` |
| `method/affordcraft/search.py` | `8633bff5927d1c0413c38ac88c1f758bca20b031288b3925c2162ad46af678ef` | `6a39c510ef0ee2e919a579032e6ee67fe852df7b7c870a585ec1440a974eefb3` |
| `method/affordcraft/vision.py` | `518014ab96d30f5b8e3e4a2a3c194e4a70175f6957c8f13a75116a3bc63f19f9` | `082877c99f3d2132f8c727e7d91f3446d841bfa59ef39222ab53b4e078ca8eb0` |
| `tools/build_visual_index.py` | `8f42b4cde448c9188dd2bfa4afd436deb7ae666a58432ec163646d3bc519c6a8` | `4c44b30248515fe9c9af91fa9a9c49caa29d95d64d0afe6f032fd804d75123ef` |
| `tools/materialize_asset.py` | `0fcd720ba2737b5c1cdb35a42ed5b2ff89859df92d355668227abf67461c02f7` | `7e59a680e84645da9e889d5b76b1f27b9cf117a5c850c753005456d56f36d2d1` |
| `tools/run_pipeline.py` | `9716ac40cc7d92d4115686ef8283d21f183da24707bf8a9dba8de47d69ecb9f9` | `a7125b4b4f235af8c1bf94b60e486875f1770bc1231b3ddb6be7a9e6912edbf2` |

Result files and the analysis scripts refer to the two versions by their internal labels `formal-code-009` (main
campaign) and `formal-code-010` (one-factor study).

- **Second detection pass**: `scripts/detect_scene_objects.py` re-detected the 18 cluttered images on which the first
  pass (the main-campaign detection routine) failed; see `docs/running.md`, step 6.

## Files

sha256 of the original (unformatted) file of the frozen tree.

| Published file | Original file | sha256 of the original | Changes besides formatting |
|---|---|---|---|
| `affordcraft/__init__.py` | method/affordcraft/__init__.py | `af1019da341282bdc54c124fb28608f8b6ec08f7b20693f62a10a851d1e9c4c0` | docstring |
| `affordcraft/backend.py` | method/affordcraft/backend.py | `14222b4668b3db45ee716118c881b38476933cb53883ca1d4bf71dbbafd89c06` | model, catalog and interpreter paths from `paths`; comments |
| `affordcraft/build.py` | method/affordcraft/build.py | `f1cbe6de58957aaf0f6d0702ce6817ad7f718d91cecac110e9fe2b1025dd01a4` | comments |
| `affordcraft/catalog.py` | method/affordcraft/catalog.py | `1f6f8cb6458e5241ca2ac699e0a170559cc8576c0197d9b7b878998dc754b86e` | - |
| `affordcraft/contracts.py` | method/affordcraft/contracts.py | `a2258b074879222fa427bbaadd468f3ccfc4151c88611af731819a662688caff` | comments |
| `affordcraft/execution.py` | method/affordcraft/execution.py | `667075d1e1158964287975fe367c996f04873c2274fe25d08bd46ef911ce7863` | health-check script path (`scripts/`) |
| `affordcraft/paths.py` | (new) | - | locations and interpreters from environment variables |
| `affordcraft/physics.py` | method/affordcraft/physics.py | `a6af2f0860a5a0b7f72e6c5235cad8b1ea5f10059bcfcd03b446e97b4dcd6c7d` | - |
| `affordcraft/search.py` | method/affordcraft/search.py | `6a39c510ef0ee2e919a579032e6ee67fe852df7b7c870a585ec1440a974eefb3` | - |
| `affordcraft/source_parser.py` | method/affordcraft/source_parser.py | `24cdda775fcee4f41d632611f100ede5a57c522f777d4d8705a44038940aa427` | batch command line of the earlier standalone importer removed (not used by the method); docstring |
| `affordcraft/vision.py` | method/affordcraft/vision.py | `082877c99f3d2132f8c727e7d91f3446d841bfa59ef39222ab53b4e078ca8eb0` | - |
| `scripts/build_visual_index.py` | tools/build_visual_index.py | `4c44b30248515fe9c9af91fa9a9c49caa29d95d64d0afe6f032fd804d75123ef` | package import path; model, catalog paths from `paths`; default GPU 0 |
| `scripts/calibrate_vision.py` | tools/calibrate_vision.py | `7d35801e2e98cdbb261a04c0f0981b5a671554dff6361839cd0133a3dd8b7d4c` | package import path; checkpoint from `paths`; default GPU 0 |
| `scripts/check_retained_invariance.py` | detection pass 2: tools/check_retained_invariance.py | `3053486bae4bd292a460ea4534fa9c06cef6951e26e46614b2b826f0fa685d63` | package import path; checkpoint from `paths`; docstring; default GPU 0 |
| `scripts/coacd_health.py` | tools/coacd_health.py | `382fc0af13ed78c8e259105863a76d165fae442cdc3463eff80d2d823c33cad0` | - |
| `scripts/detect_scene_objects.py` | detection pass 2: tools/detect_scene_objects.py | `3aa4e2ef3ee529175e54d58253549fbdad2c8d2e573dfbb03963f11d60e845c4` | package import path; checkpoint from `paths`; docstring; default GPU 0 |
| `scripts/evaluate_scenes.py` | tools/evaluate_scenes.py | `eaec8bb8c30cfccb568f6bc46d764735a189fecd06ef31779af37898e5a3f99d` | - |
| `scripts/export_final_results.py` | tools/export_final_results.py | `a52fd56330681e694935a939b4bbc49e601c48b2fa9715d16043ca759a73f36d` | - |
| `scripts/freeze_experiment.py` | tools/freeze_experiment.py | `d7bd606bb4545ad3c9dce8f5d2938417158ac5b96366b958d9b16bd984ea9be6` | calibration evidence as arguments; lock set = this repository and the model files; options `--variant`, `--shard-count`, `--cluttered` write the fields the study registrations wrote |
| `scripts/make_physics_fixtures.py` | tools/make_physics_fixtures.py | `429ac2b2e10cabb20c1899262dbfebbb7be6368119df440ff3787c43da8582f1` | - |
| `scripts/materialize_asset.py` | tools/materialize_asset.py | `7e59a680e84645da9e889d5b76b1f27b9cf117a5c850c753005456d56f36d2d1` | package import path; comment |
| `scripts/merge_shards.py` | (new) | - | merges the final shard attempts (reproduces the verified per-case file of the main campaign exactly) |
| `scripts/physics_worker.py` | tools/physics_worker.py | `b04a1deb83a192555b5db9e96817f63e56a2e565a3a0596e22a27f4f86ae4978` | package import path; default GPU 0 |
| `scripts/prepare_calibration.py` | tools/prepare_calibration.py | `30461763d973cb27928ea59a503e349743e12ae11b7c50b2530af25299af3952` | manifest, catalog and project root as arguments |
| `scripts/prepare_inputs.py` | (new) | - | builds the runnable manifests from `data/inputs` (replaces three campaign-specific preparation scripts; same output format) and the two input files of the external-method wrappers (byte-identical to the manifest of the paper when the images lie at the same relative paths) |
| `scripts/run_physics_job.py` | tools/run_physics_job.py | `0c5aa6a3a70db22fbf04e422c95f75f9af1e1691c3a815e50db27aab30ca99e5` | Isaac interpreter from `paths`; worker path; default GPU 0 |
| `scripts/run_pipeline.py` | tools/run_pipeline.py | `a7125b4b4f235af8c1bf94b60e486875f1770bc1231b3ddb6be7a9e6912edbf2` | package import path; project root and runtime interpreter from `paths`; default GPU 0 |
| `tests/test_code005.py` | tests/test_code005.py | `a8baa66033b69c3c2da652f2b3a96c4f07b2923dab0586571d92d67de0ad46f2` | package import path |
| `tests/test_contracts.py` | tests/test_contracts.py | `192c61d44b3e2ea3713c3d0cfea64fa3bd645ffa32aabdfae3574169867e4b58` | package import path |
| `tests/test_detection.py` | detection pass 2: tests/test_detection011.py | `6895c29ebafe8b3bb639e5db945cc75bc95d4ba13269c392b508c3702cd051be` | import path; docstring |
| `tests/test_geometry.py` | tests/test_geometry.py | `433473913819996bb1fa8aff88efe5bc1fb18142fac4d430ab326a05010e9b70` | package import path |
| `tests/test_recovery.py` | tests/test_recovery.py | `089265f9970b91141e9862d8ea340cc3d89d70b7285e0f850f8d5bb316e68094` | package import path; folder names |
| `tests/test_variants.py` | tests/test_variants.py | `af92a8aa46aaf85eef5ce9ee485f5cdbbf81d7de9cc239755f2900341fef895b` | package import path; folder names |

Not published from the frozen trees: the campaign queue, controllers and job runner (`campaign.py`, `controller_*.py`,
`job_runner.py`, `supervise_pipeline.py`, `launch_formal.py`, `interrupt_owned_run.py`, `cancel_owned_calibration.py`),
deployment and snapshot tools (`deploy.py`, `bootstrap_remote.py`, `sanitize_snapshot.py`, `record_index_process.py`),
campaign registration and verification (`register_recovery*.py`, `verify_campaign.py`, `verify_study_014.py`,
`validate_recovery.py`), development probes (`calibrate_geometry_005.py`), an earlier scene-construction entry point
superseded by `run_pipeline.py --scene-mode` (`run_scene_construction.py`), the input-preparation scripts replaced by
`prepare_inputs.py` (`make_subset_inputs.py`, `make_category_inputs.py`, `prepare_scene_inputs.py`), and manuscript
tools (`compile_paper.py`, `render_paper_review.py`, `draw_method.py`, `run_tests.py`).
