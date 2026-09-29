# PhysX-Omni

Official PhysX-Omni (arXiv:2605.21572) on the whole photograph, 2,000 inputs: rows "PhysX-Omni" of the main comparison
table (`tab:main`, Section 4.2) and of the resource table (`tab:resource-metrics`). Each stage registers a
`configuration.json` before it runs, with the hashes of the official scripts it calls (`1vlm_demo.py` and the prompt
file for the representation; `decoder_each.py`, `2infer_geo.py`, `3jsongen_update.py` for geometry and export) and of
every checkpoint file; the commit of the official checkout itself was not recorded.

## Stages

1. Native representation, `omni_representation_runner.py` (fine-tuned Qwen2.5-VL snapshot `765cd275` with the official
   `1vlm_demo.py` and prompt `dataset/example_64_finetune_rle.txt`): the whole photograph resized to 512 x 512 (Lanczos),
   bfloat16, SDPA attention, greedy decoding, seed 1, maximum sequence length 32,768; per case `cond_img.png`, `basic_info.txt`,
   `ind_<k>.npy`, `allind.npy`, `timing.json` and `native_representation_result.json` (`native_representation_emitted`,
   `part_count`, `input_sha256`). Shards are a fixed round-robin partition of the manifest; model calls are timed with
   CUDA synchronization and nothing else changes.
2. Geometry and export, `omni_geometry_runner.py` (sm_120 geometry environment): for every case whose representation was
   emitted, the official TRELLIS decoder (`microsoft/TRELLIS-image-large` @ `25e0d31f`) decodes each part
   (`pipeline.run_decoder(coords, cond_img, seed=1, eachcoords=...)`), `to_glb(simplify=0.5, texture_size=1024)`, OBJ
   rotated by +90 degrees about x as in the official code, then the official exporter `3jsongen_update.py --voxel_define
   64 --process 0 --fixed_base 0 --deformable 0` writes `basic.xml` / `basic.urdf` in a write-isolated copy of the case.
   No substitute mesh, no deletion of missing links; the exporter's physical parameters are kept as delivered.
3. Exporter, `omni_native_manifests.py`: native manifests from the official MJCF (mesh geoms carry the cm-to-m scale and
   the per-part density from `basic_info`; the URDF has visual-only links with placeholder inertia) plus one list of
   manifests for the gate, one entry per input (an emitted manifest is preferred over an out-of-memory attempt).

## Representation union and lanes

The geometry lanes read a union root with one symlink per case to its representation. Precedence, fixed before any
result: the registered-subset representation lanes (7 lanes, subset index % 7), then the shards over the first 200 rows
of the frozen manifest, then the lanes over the remaining 1,620 rows. Registered subset: `run_geometry_lane_subset200.sh`
(subset order, lane = index % lanes). Full 2,000: 16 geometry lanes over the other inputs in representation-availability
order (`run_geometry_lane.sh` with one query manifest per lane).

Out of memory: the geometry stage peaks at about 91 GiB. A case that failed with a CUDA out-of-memory on a 32 GB GPU is
an infrastructure failure: it is recomputed once with the identical runner and configuration on an 80 GB or 96 GB GPU,
in a new attempt directory (`run_geometry_lane_attempt.sh`: finished cases of earlier attempts are linked, OOM failures
and interrupted cases are recomputed; `attempt_lineage.json` records which).

## Usage

Representation (one shard; `--input-manifest` holds rows of `$AFFORDCRAFT_RUNS/inputs/input_manifest_2000.jsonl`, see
`../README.md`; the workspace holds the links `physx-omni` (official checkout), `physx-omni-765cd275` (checkpoint) and
`qwen2.5-vl-7b-processor`):

```bash
python baselines/physx_omni/omni_representation_runner.py --workspace /abs/omni-workspace \
    --input-manifest /abs/shard-0/input_manifest.jsonl --input-root $AFFORDCRAFT_PROJECT_ROOT \
    --output /abs/shard-0/native_representation --registration     # writes configuration.json once
python baselines/physx_omni/omni_representation_runner.py ...  (same arguments without --registration)
```

Geometry and export:

```bash
export AFFORDCRAFT_RUNS=/abs/runs PHYSX_OMNI_RUN=$AFFORDCRAFT_RUNS/external/physx-omni/geometry-subset200
export PHYSX_OMNI_GEO_PYTHON=/abs/envs/geom-sm120/bin/python PHYSX_OMNI_HOME=/abs/physx-omni
export DINOV2_HUB_REPO=/abs/dinov2-7764ea0f DINOV2_CKPT=/abs/dinov2_vitl14_reg4_pretrain.pth
export U2NET_ONNX=/abs/u2net.onnx TRELLIS_IMAGE_LARGE=/abs/TRELLIS-image-large-25e0d31f
GEO_REP=$PHYSX_OMNI_RUN/representation_union GEO_QUERY=/abs/query_manifest_lane0.jsonl GEO_OUT=$PHYSX_OMNI_RUN/lane-0 \
  GEO_GPU=0 GEO_TAG=lane0 bash baselines/physx_omni/run_geometry_lane.sh
python baselines/physx_omni/omni_native_manifests.py --lanes $PHYSX_OMNI_RUN/lane-0 --run-id geometry-subset200 --list manifests.txt
```

The runner workspace (`PHYSX_OMNI_GEO_WORKSPACE`, default `$PHYSX_OMNI_RUN/workspace`) holds the links `physx-omni`,
`dinov2-7764ea0f912e53c92e82eb78a2a1631e92725fc8`, `dinov2_vitl14_reg4_pretrain.pth.part`, `u2net.onnx`,
`trellis-image-large-25e0d31f`; the runner checks the DINOv2 and u2net hashes. Each lane first registers
(`--register` writes `configuration.json` once) and then runs; a re-run with a drifted configuration refuses to start.

Outputs per case (`<lane>/native_geometry*/cases/<source_id>/`): `native/<source_id>/` (official exporter outputs:
`objs/<k>/<k>.obj|.glb`, `basic.xml`, `basic.urdf`), `geometry_report.json`, `phase_timing_events.jsonl`, `timing.json`,
`native_asset_result.json`, `native_manifest.json`.

Dependencies: sm_120 geometry environment (torch 2.7.0+cu128, spconv-cu126 2.3.8, nvdiffrast and diff_gaussian_rasterization
rebuilt for sm_120, xformers 0.0.31, trimesh); the 80 GB lanes used an sm_80 environment with the rasterizer wheel of
the earlier run.

## Compatibility changes (official source untouched)

- `ATTN_BACKEND=SPARSE_ATTN_BACKEND=xformers` with FA3 dispatch disabled and `cutlass.FwOp` forced (sm_120);
  `SPCONV_ALGO=native`; offline DINOv2 (the `torch.hub.load` of `dinov2_vitl14_reg` is served from the local hub repo and
  checkpoint with a strict state-dict load); u2net from the workspace.
- The official exporter writes next to its inputs, so it runs on a hash-checked copy of the representation files.
- Out-of-memory cases recomputed on larger GPUs as described above.

## Changes from the executed version

- `omni_representation_runner.py` and `omni_geometry_runner.py` = the executed runners renamed (internal version
  suffixes dropped), imports updated.
- `execution_utils.py` = the helpers both runners import, copied verbatim from the executed helper module; the rest of
  that module (campaign registration, logged command execution, benchmark download) is not used by the runner.
- Lane scripts: run root, workspace sources and interpreter from environment variables; the runner is copied from this
  folder into `code/` of the run root (the executed scripts copied it from a tools directory).
- Comments: internal run names and dates removed. Formatting with black (AST unchanged).
- Not published: node setup and representation pull loops, union builders (rule above), OOM claim/dispatch loops,
  query-manifest reordering, gate refresh and sync scripts, a fixer for manifests written during a file transfer.

## Provenance

| original file | published path | sha256 of the original |
|---|---|---|
| `paper_completion_execution_v0_2.py` (excerpt: the helpers the runner imports) | `execution_utils.py` | `2a00b4f5783c32c8b0553fb576eedd255c75d4881720e77879e1171b8f4a422b` |
| `paper_completion_omni_native_timed_v0_3.py` (renamed; import updated) | `omni_representation_runner.py` | `ebc62f77fd76e9b529b67c9e6d0d5d08ad270abdf74de060526466967e853259` |
| `paper_completion_omni_geometry_timed_v0_5.py` (renamed; import updated) | `omni_geometry_runner.py` | `496f4749876990b42e1614d9919279a0d23f14e45896076ab85bcf964751ec74` |
| `omni_native_manifests.py` | `omni_native_manifests.py` | `8100d2e0369ec45a15dac08e1232fad95ad44f10a5add4d9460885c98e47ff3a` |
| `geometry_lane2.sh` (generic geometry lane (full-2000 lanes and later attempts)) | `run_geometry_lane.sh` | `baa121e6eff0902e26c46823cb6deafbc5a57a1ea6ac4a728617d6d420889b1f` |
| `geometry_lane_attempt.sh` (renamed) | `run_geometry_lane_attempt.sh` | `81bafacc74d8fc6e7780a073bb659f183d6894b998166ff6af2e204382a98661` |
| `geometry_lane.sh` (first registered-subset geometry lanes (lane split inside)) | `run_geometry_lane_subset200.sh` | `472257a7336373fff5279d012668a456c21c679361cc5321ced5663044453c4a` |
