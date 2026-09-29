# Data

## Inputs (`data/inputs/`)

The repository lists the inputs by public image ids; `scripts/prepare_inputs.py` turns them into the runnable manifests
(absolute image paths plus image hashes, in the format of the frozen experiment definitions) and refuses images whose
bytes differ from the recorded hash.

| File | Content |
|---|---|
| `single_object_2000.json` | the 2,000 single-object photographs (Open Images V7; 1,704 from the validation split, 296 from the train split) from 31 categories, in the frozen manifest order: input id, Open Images image id and split, image sha256, requested category and task instruction |
| `subset_200.json` | the registered 200-input subset (seed 20260916; at least one input per source category, the rest in proportion to category size; drawn from the inputs alone, never from outcomes), in registered order. Used by the one-factor study and shared by all external methods |
| `cluttered_scenes_50.json` | the 50 cluttered COCO images (COCO image id, image sha256, instruction); the model sees only the whole image and the instruction |
| `cluttered_reference_objects.json` | the 237 annotated instances (category, normalized box) used only to score the cluttered-image experiment: same-category, one-to-one matching at IoU >= 0.5 (`scripts/evaluate_scenes.py`) |

Every single-object instruction has the form "Construct an interactive simulation asset for the `<category>` visible in
this image. Preserve its mechanism and natural support conditions." No method receives a reference box or mask.

```bash
python scripts/prepare_inputs.py single    --open-images /datasets/openimages_v7 --output runs/inputs/single_inputs.json
python scripts/prepare_inputs.py subset    --single runs/inputs/single_inputs.json --output runs/inputs/subset_200.json
python scripts/prepare_inputs.py category  --single runs/inputs/single_inputs.json --output runs/inputs/category_548.json
python scripts/prepare_inputs.py cluttered --coco /datasets/coco/val2017 /datasets/coco/train2017 --output runs/inputs/scenes
```

`category_548.json` holds every Window, Door and StorageFurniture input (263 + 145 + 140), the second track of the
one-factor study.

The external-method wrappers (`baselines/`) read the same inputs in two other files; write them with

```bash
python scripts/prepare_inputs.py baselines --single runs/inputs/single_inputs.json --subset runs/inputs/subset_200.json \
    --project-root $AFFORDCRAFT_PROJECT_ROOT --output $AFFORDCRAFT_RUNS/inputs
```

(see `baselines/README.md`, "Input files").

## Asset library (`data/library/library_index.csv`)

The library has 11,372 entries from four public collections, 283 catalog categories:

| Collection | Entries | Structure | Catalog preview used for selection |
|---|---|---|---|
| PartNet-Mobility (SAPIEN) | 2,335 | articulated URDF (revolute and/or prismatic joints) | front view rendered by the Articulate-Anything preprocessing (`robot_frontview.png`, 2,223 entries); a deterministic software render where it was missing (112) |
| Objaverse 1.0 | 8,986 | rigid mesh | deterministic mesh projection |
| Google Scanned Objects | 41 | rigid mesh | the collection's thumbnail |
| YCB | 10 | rigid mesh | deterministic mesh projection |

`library_index.csv` lists every entry: `candidate_id` (the id used throughout the code and the result files),
`library`, `source_object` (PartNet-Mobility object id, Objaverse uid, GSO or YCB object name), `category`,
`motion_types` (`revolute`, `prismatic`, `dynamic_rigid`), `joint_count` and `body_count`.

The catalog itself (`catalog.json`) is not distributed: it points into local copies of the four collections, whose
licenses we do not redistribute. Build it from the index with the schema below.

### Catalog schema

`catalog.json` is `{"entries": [...]}`. Paths are absolute or relative to `AFFORDCRAFT_PROJECT_ROOT`; every file
reference carries its sha256, and the code refuses a file whose hash changed. The fields the code reads:

| Field | Entries | Used by |
|---|---|---|
| `candidate_id`, `category` | all | retrieval (category preference), selection, results |
| `motion_types`, `joint_types`, `semantic_parts` (`[{"link", "part", "joint_semantic"}]`) | all / PartNet-Mobility | the structure the selector reads next to each preview |
| `preview` `{path, sha256}` | all | visual index (`build_visual_index.py`) and the selector's candidate images |
| `urdf` `{path, sha256}`, `source_dir`, `missing_mesh_paths`, `semantics` `{path, sha256}` | PartNet-Mobility | source-preserving import of links and joints (`affordcraft/source_parser.py`), installation metadata (root links declared static) |
| `source` `{visual, collision}` (each `{path, sha256}`), `structural.joint_count` | rigid entries | rigid import; entries whose collision mesh was derived as the convex hull of the visual mesh record the method in `source.collision_source` |
| `unit_metadata` | optional | declared units; PartNet-Mobility units are inferred from the dataset convention and recorded |

Example (PartNet-Mobility, paths shortened):

```json
{"candidate_id": "35059", "category": "StorageFurniture", "motion_types": ["revolute"], "joint_types": ["fixed", "revolute"],
 "semantic_parts": [{"joint_semantic": "hinge", "link": "link_0", "part": "rotation_door"},
                    {"joint_semantic": "heavy", "link": "link_1", "part": "furniture_body"}],
 "preview": {"path": "partnet-mobility/dataset/35059/robot_frontview.png", "sha256": "de8e739c..."},
 "urdf": {"path": "partnet-mobility/dataset/35059/mobility.urdf", "sha256": "d906fd73..."},
 "semantics": {"path": "partnet-mobility/dataset/35059/semantics.txt", "sha256": "dc496220..."},
 "source_dir": "partnet-mobility/dataset/35059", "missing_mesh_paths": [], "joint_count": 2, "body_count": 3}
```

## Library scaling

The five nested libraries of the library-scaling analysis answer the same 2,000 queries: 141 entries (base), 418 (adds
instances of covered categories), 472 (adds mechanisms), 614 (adds categories) and the full 11,372. Their recorded
coverage, top-1 agreement and query times are regenerated by `analysis/figures/fig6_library_compact.py` and
`analysis/tables/make_tables.py`.
