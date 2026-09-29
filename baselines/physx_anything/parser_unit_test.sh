#!/usr/bin/env bash
# Parser unit test against the OFFICIAL exporter: run 4_simready_gen.py (unmodified) on a scratch copy of the smoke case whose
# basic_info.txt is replaced by a synthetic articulated description (C hinge, B slide, D ball, CB hinge+slide), then cross-check
# code/test_mjcf_parser.py against MuJoCo and print the manifest joints. Scratch only; nothing under the run root is touched
# except the report copied to <REV>/parser_tests/.
# environment: PHYSX_ANYTHING_RUN, PHYSX_ANYTHING_HOME, PHYSX_ANYTHING_SMOKE_CASE (a finished case directory whose
# objs/ and ind_*.npy are reused), PHYSX_ANYTHING_VLM_PYTHON, PHYSX_ANYTHING_GEOM_PYTHON, PARSER_UNIT_DIR (scratch)
set -u
REV=${PHYSX_ANYTHING_RUN:-${AFFORDCRAFT_RUNS:-runs}/external/physx-anything/subset200}
SRC=${PHYSX_ANYTHING_HOME:?set PHYSX_ANYTHING_HOME to the official checkout}
U=${PARSER_UNIT_DIR:-${TMPDIR:-/tmp}/physx-anything-parser-unit}
C=${PHYSX_ANYTHING_SMOKE_CASE:?set PHYSX_ANYTHING_SMOKE_CASE to a finished case directory (the run used openimages_001bdfc9d80eead2)}
VPY=${PHYSX_ANYTHING_VLM_PYTHON:-python}
GPY=${PHYSX_ANYTHING_GEOM_PYTHON:-python}
export PYTHONDONTWRITEBYTECODE=1
rm -rf "$U"; mkdir -p "$U/test_demo/synthetic"
cp -a "$C/objs" "$C"/ind_*.npy "$C/allind.npy" "$U/test_demo/synthetic/"
cp "$REV/code/parser_unit_basic_info_synthetic.txt" "$U/test_demo/synthetic/basic_info.txt"
ln -s "$SRC/mjcf_source" "$U/mjcf_source"
cd "$U"
"$GPY" "$SRC/4_simready_gen.py" --voxel_define 32 --basepath ./test_demo --process 0 --fixed_base 0 --deformable 0 > export_stdout.log 2>&1
echo "export_exit=$?"; tail -2 exp_urdf.log
echo "---- MJCF worldbody"; sed -n '/<worldbody>/,/<\/worldbody>/p' test_demo/synthetic/basic.xml
echo "---- parser check"
"$VPY" "$REV/code/test_mjcf_parser.py" test_demo/synthetic --json "$U/parser_unit_synthetic_check.json" > check_stdout.log 2>&1; echo "check_exit=$?"
"$VPY" - "$U/parser_unit_synthetic_check.json" <<'PY'
import sys, json
r = json.load(open(sys.argv[1]))
print(json.dumps({k: r.get(k) for k in ('ok', 'mismatches', 'parser_joint_counts', 'mujoco_joint_counts', 'parser_free_bodies', 'mujoco_free_bodies', 'mujoco_load_ok', 'mujoco_error', 'mujoco_body_mass_kg')}, indent=1))
PY
echo "---- manifest joints"
"$VPY" - "$REV/code" "$U/test_demo/synthetic" "$U/native_manifest_synthetic.json" <<'PY'
import sys, json
sys.path.insert(0, sys.argv[1])
from mjcf_native_manifest import build_native_manifest
m = build_native_manifest(sys.argv[2], 'synthetic', 'physx-anything-v0.1', True)
for j in m['joints']:
    print(json.dumps({k: j[k] for k in ('name', 'type', 'parent', 'child', 'axis', 'origin_xyz', 'lower', 'upper', 'unbounded_revolute')}))
print('links', [(l['name'], l['parent'], l['geom_count'], l['mesh_scale'], l['density_kg_m3']) for l in m['links']])
print('root', m['root_link'], 'free', m['free_bodies'], 'warnings', m['parser_warnings'])
json.dump(m, open(sys.argv[3], 'w'), indent=1)
PY
mkdir -p "$REV/parser_tests"
cp -n "$U/parser_unit_synthetic_check.json" "$REV/parser_tests/parser_unit_synthetic_check.json"
cp -n "$U/test_demo/synthetic/basic.xml" "$REV/parser_tests/synthetic_basic.xml"
cp -n "$U/test_demo/synthetic/basic_info.txt" "$REV/parser_tests/synthetic_basic_info.txt"
cp -n "$U/native_manifest_synthetic.json" "$REV/parser_tests/native_manifest_synthetic.json"
cp -n "$C/mjcf_check.json" "$REV/parser_tests/smoke_case_openimages_001bdfc9d80eead2_mjcf_check.json"
echo "PARSER_UNIT_DONE"
