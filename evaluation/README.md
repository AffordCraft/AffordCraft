# Evaluation of external image-to-asset methods

This folder holds the code that scores the seven external image-to-asset routes of the paper with AffordCraft's own
physical gate, aggregates the verdicts into the comparison table, measures per-input time on exclusive GPUs, and prices
the API-based routes. It supports Table 1 (main comparison), the resource table, the appendix on external methods and
resources, and the time-to-pass figure (the table and figure generators themselves are in `analysis/`).

| Folder | Content | Paper |
|---|---|---|
| `gate/` | the physical gate for one external output, native and adapted condition; the native-manifest contract | Table 1 (Export / Native / Adapted / Artic.), external appendix |
| `aggregate/` | verdict aggregation per method and track, finalization rules | Table 1, resource table |
| `timing/` | clean-GPU timing run (32-input sample, one worker per exclusive RTX 5090) and its summarizer | Table 1 per-input columns, resource table, time-to-pass figure |
| `api_cost/` | API cost per input at official list prices | resource table (API cost row) |

## Protocol

Every external method runs on the registered 200-input subset of the full-RGB protocol (whole photograph, no reference
box or mask; the methods that accept text also receive the task instruction) and, for the five local generators, on the
remaining 1,800 inputs of the frozen 2,000-input manifest. Each method's own exporter output is translated, without
changing it, into a common **native manifest** (`gate/manifest_contract.md`): links with their meshes, scale, density or
mass, joints with type, axis, origin and limits. A case whose method stops before exporting keeps `asset_emitted: false`
and stays in the denominator.

The gate scores every emitted asset in two conditions with the same frozen physics worker that admits AffordCraft's own
assets (`scripts/physics_worker.py` and `affordcraft/physics.py`, `EvaluationPolicy()`: 360 steps at 1/120 s, 60-step
tail, penetration 1 mm, settling 5 cm / 5 deg, floor tolerance 12 mm):

- **native**: the method's own export translated to the AffordCraft USD layout without changing geometry, joints,
  scale, density or mass. The MJCF is used first (links, joints, mass, center of mass and inertia from MuJoCo's own
  compile of it); otherwise the manifest's links and joints, with mass = declared density x convex-hull volume (or the
  declared mass). The collision of every delivered part is its convex hull, which is what MuJoCo does with mesh geoms.
  A method that delivers no physical parameters (no density, no mass) gets **not applicable** for the native condition.
- **adapted**: the same links and joints written as a URDF candidate and passed through AffordCraft's own build, i.e.
  exactly the common physical adaptation of the method: uniform scale from the main campaign's grounding size estimate
  for the same input (used when its confidence is at least 0.65), CoACD decomposition cascade with audits, uniform
  density 500 kg/m3, mass and inertia from the decomposed collision hulls, free-standing support contract; one attempt, no
  repair alternative. The build runs with a 600 s budget inside a 1,500 s process limit.

Support role: free-standing for every external asset (generated assets carry no installation evidence). Nothing in the
pipeline selects among outcomes; every case record is written once (`open(..., 'x')`).

### Verdict statuses (per condition, `cases/<source_id>/case_result.json`)

| Status | Meaning | Counted as |
|---|---|---|
| `evaluated` | the frozen gate ran; `physical_pass` true or false | pass / fail |
| `not_run` | the method exported no asset | fail |
| `not_applicable` | native condition only: the export declares no mass or density | no native verdict (never a native pass); when it holds for a whole method, Table 1 shows "--" and reports the adapted path |
| `blocked` | the native export cannot be loaded or authored (e.g. MuJoCo refuses the MJCF) | terminal method-side failure (`native_not_loadable`), fail in both conditions |
| `construction_failed` | adapted condition: AffordCraft's build rejected the candidate | fail |
| `infrastructure_blocked` | the build process was killed (signal -9/-6/-11 from the memory guard) or ran past 1,500 s, or a physics worker failed | open: re-gated, never counted as pass or fail while open |
| `capacity_exhausted` | an adapted build killed by the memory guard or the build time limit at its first gate AND again in a big-memory re-gate (and a third time alone, where tried), or still blocked at the finalization cutoff | physical failure |

Counts always use the full registered denominator; a track is *complete* when every input has a terminal verdict and
none is infrastructure-blocked. Rates carry Wilson 95 % intervals.

## Running the gate on one case

Environment (see the repository README for the interpreters): `AFFORDCRAFT_ISAAC_PYTHON` (Isaac Sim 5.1, physics
worker), `AFFORDCRAFT_RUNTIME_PYTHON` (numpy, scipy, trimesh, CoACD, mujoco, usd-core; also runs the pipeline itself),
`AFFORDCRAFT_PROJECT_ROOT`, `AFFORDCRAFT_RUNS`. The grounding table of the main campaign gives the size estimate per
input (`{"grounding": {"<source_id>": {"size_m": ..., "size_confidence": ...}}}`). It ships as
`gate/grounding_table.json`, the gates' default; the executed table (sha256
`ded167773ce8469378ee8600d57366407d1dbec11dedd93690116855be5eb8c6`) additionally held the path of each source record,
which is removed here.

```bash
echo /abs/path/to/<run>/lane-0/cases/<source_id>/native_manifest.json > manifests.txt
$AFFORDCRAFT_RUNTIME_PYTHON evaluation/gate/external_gate_pipeline.py \
    --manifests manifests.txt --method-id physx-anything-v0.1 \
    --out $AFFORDCRAFT_RUNS/gates/physx-anything --cache $AFFORDCRAFT_RUNS/gates/geometry_cache \
    --gpu 0 --lane lane0
```

Output: `<out>/cases/<source_id>/case_result.json` (both conditions: status, `physical_pass`, failure reasons, gate
measurements, seconds), `native/` (authored USD, `native_asset_manifest.json`), `adapted/` (URDF candidate, build
request, build log, `adapted_asset_manifest.json`), plus `<out>/configuration.json` (sha256 of the frozen code files, the
pipeline and the grounding table) and `<out>/configuration_amendment_001_materialize_slots.json`. `--phases native` or
`--phases adapted` runs one condition. A case directory that exists without `case_result.json` (an interrupted process)
is moved to `<out>/interrupted/` and recomputed.

In the paper's runs, many such processes worked through hash-partitioned lists of manifests on several machines. A case
that ended `infrastructure_blocked` was re-queued (up to four times, every attempt kept under `interrupted/`) and then
re-gated on machines with 360-450 GiB of memory (`<out>/retry-bigmem/`, and for some cases a clone stage
`<out>/clone-<name>/` with one builder alone); the aggregator reads all of these stages with a fixed precedence.

## Amendments of the gate pipeline

The pipeline was registered before the runs and changed three times; each change is recorded in the gate runs as
`configuration_amendment_00k_*.json`, and every earlier verdict was kept.

1. **001 materialize slots.** The containers' memory guard killed the largest process (signal -9, no cgroup OOM event)
   when a dozen CoACD decompositions (25-45 GB each) ran side by side, which killed about 40 % of the adapted builds.
   From then on a machine-level semaphore allows at most `GATE_MAT_SLOTS` concurrent builders (2 on 180 GiB machines,
   4 on 360 GiB); the wait is recorded separately and never counted as build time; blocked cases were re-queued up to
   `GATE_INFRA_RETRIES` = 4 times.
2. **002 fixed joints get axis None.** PAct's official `json_to_urdf` writes `<axis xyz="0 0 0"/>` on fixed joints; the
   frozen axis-quaternion routine cannot normalize it (ZeroDivisionError), so the native condition was wrongly `blocked`
   for the 64 of 200 PAct manifests with such joints. A fixed joint now carries no axis, as in AffordCraft's own URDF
   parser. No change for MJCF-based methods or joint-free meshes; the adapted condition never wrote an axis for fixed
   joints. The affected cases were recomputed.
3. **003 geometry-free structural links.** Articulate-Anything's URDFs contain the PartNet-Mobility `base` /
   `base_helper` links without geometry (`visual_obj: null`); the frozen code passed `None` to the mesh loader and
   every Articulate-Anything export was wrongly `blocked` as not loadable. Such a link is now read as an empty body, as
   the MJCF path and the tree normalization already represent geometry-less bodies. No change for manifests whose links
   all carry geometry (every other method). The affected verdicts were recomputed.

`gate/external_gate_pipeline.py` is the deployed version with all three amendments.

## Finalization rule

Two terminal rules turn verdicts that stay `infrastructure_blocked` in the adapted build into physical failures
(`capacity_exhausted`); both only mark verdicts, never recompute or delete them (`aggregate/finalize.py`):

- **retry exhausted** (aggregator v2.6): an adapted build killed by the memory guard or the build time limit at the
  origin gate, again in the big-memory re-gate, and (where tried) a third time alone on a big-memory machine is
  capacity exhausted; a `retry_exhausted.json` marker beside the re-gate verdict records it. Used for the PartCrafter
  2,000-input gates.
- **cutoff** (aggregator v2.12): the 2,000-input runs of the five local methods were finalized at one cutoff; a gate
  condition still blocked by the memory guard at that point, after its re-attempts and any big-memory re-gates so far,
  is counted as capacity exhausted. The finalization record lists every such input; a later unblocked verdict still
  takes precedence, and gate runs started after the cutoff are never finalized. The run refuses to finalize an input
  blocked in generation or for any other reason than the memory guard. The API routes were not part of the cutoff.

Only the adapted condition was affected; the native condition of these inputs was evaluated normally.

## Changes from the executed version

See the README of every subfolder. Common to all: absolute paths, interpreters and host-specific directories became
environment variables or command-line arguments; the orchestration that ran the executed version (dispatchers that
assigned batches to machines, copy and synchronization scripts, per-machine loops, progress monitors, watchdogs,
patchers of running files) is not included.

## Provenance

The provenance tables (original file name, published path, sha256 of the original file) are in the subfolder READMEs.
