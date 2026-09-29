# External-method native manifest contract

Every external method run writes, per input case, `cases/<source_id>/native_manifest.json` with exactly this schema.
The downstream physical-gate pipeline (`external_gate_pipeline.py`: frozen AffordCraft PhysX contract, native and
adapted conditions) consumes only this file plus the mesh files it points to. Paths are absolute, so the gate can read a
case wherever the method wrote it.

```json
{
  "schema": "affordcraft.external_native_manifest.v1",
  "source_id": "openimages_xxxxxxxxxxxxxxxx",
  "method_id": "physx-anything-v0.1",
  "run_id": "<run id>",
  "asset_emitted": true,
  "stage_reached": "export",
  "failure_reason": null,
  "units": "m",
  "metric_size_source": "method",
  "native_formats": ["MJCF", "URDF"],
  "mjcf": "/abs/path/basic.xml",
  "urdf": "/abs/path/basic.urdf",
  "root_link": "l_0",
  "links": [
    {"name": "l_0", "visual_obj": "/abs/path/objs/0/0.obj", "mesh_scale": [0.12, 0.12, 0.12],
     "density_kg_m3": 1050.0, "mass_kg": null, "origin_xyz": [0, 0, 0], "origin_rpy": [0, 0, 0]}
  ],
  "joints": [
    {"name": "pivot_7", "type": "revolute", "parent": "l_7", "child": "l_6",
     "axis": [1.0, 0.0, 0.0], "origin_xyz": [-0.06, 0.0295, -0.0013], "lower": 0.0, "upper": 0.9948}
  ],
  "physics_contract": true,
  "notes": "free text: how links/joints were derived from the native export"
}
```

Field rules
- `asset_emitted` is true only if the method's own exporter produced a structured asset (URDF/MJCF/GLB with parts). A
  case that stops earlier keeps `asset_emitted:false`, `stage_reached` (`vlm`, `decoder`, `split`, `export`, ...) and a
  `failure_reason`; it is never dropped from the denominator.
- `units`: `m` after applying `mesh_scale`. `mesh_scale` is the per-axis scale the native export applies to the OBJ
  (e.g. the MJCF `<mesh scale=...>`); the OBJ file itself stays untouched. If the native export is in normalized units
  with no metric scale, write `"metric_size_source": "none"` and `mesh_scale` = `[1,1,1]`.
- `origin_xyz`/`origin_rpy` of a link: pose of the link mesh frame in the parent-joint child frame (MJCF body `pos`/`euler`),
  meters / radians. Joint `origin_xyz` is the pivot in the parent link frame (MJCF joint `pos` + body `pos`), `axis` in the
  parent link frame, `lower`/`upper` in radians (revolute) or meters (prismatic). `type` in
  `revolute|continuous|prismatic|fixed|spherical|floating`.
- `density_kg_m3` and `mass_kg`: whatever the native export declares (MJCF geom density, URDF mass); `null` if absent.
  `physics_contract` is true iff the native export declares mass/density AND collision geometry (MJCF mesh geoms count as
  collision; a URDF with visual-only links does not).
- Methods without articulation (rigid single mesh or unarticulated parts) list their parts as links with no joints
  (fixed assembly is authored downstream); note it in `notes`.
- The manifest is written once (`open(..., 'x')`), never rewritten. Keep every intermediate artifact of the case.

How the gate reads the manifest (see `external_gate_pipeline.py`)
- If `mjcf` names an existing file, links, joints, mass, center of mass and inertia come from MuJoCo's own compile of
  that MJCF (`parse_mjcf`); the `links`/`joints` lists are not used.
- Otherwise the `links`/`joints` lists are read (`parse_manifest_links`). A link with `visual_obj: null` is a
  geometry-free structural link (amendment 003); a joint of type `fixed` never carries an axis (amendment 002).
  Mass/inertia are density x convex-hull volume when `density_kg_m3` or `mass_kg` is given; when neither is given the
  native condition is `not_applicable`.
