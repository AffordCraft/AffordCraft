#!/usr/bin/env python3
"""pxr helper, run by external_gate_pipeline.py with the runtime interpreter of the frozen construct_asset: apply the
construct_asset post-edits to a frozen author_usda stage (body transforms, articulation flags, joint axis tokens/continuous
flags, spherical joints, support floor). Input: JSON spec path."""
import json, sys
from pxr import Usd, UsdGeom, UsdPhysics, Gf, Sdf

spec = json.load(open(sys.argv[1]))
stage = Usd.Stage.Open(spec["usd"])
root = stage.GetPrimAtPath("/World/Asset")
if not spec["has_joints"]:
    root.SetMetadata("apiSchemas", Sdf.TokenListOp.CreateExplicit([]))
else:
    root.CreateAttribute("physxArticulation:enabledSelfCollisions", Sdf.ValueTypeNames.Bool).Set(True)
for b in spec["bodies"]:
    prim = stage.GetPrimAtPath(b["path"])
    tf = b["transform_rows"]
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    xf.AddTransformOp().Set(Gf.Matrix4d([[tf[c][r] for c in range(4)] for r in range(4)]))
    if b["empty"]:
        prim.CreateAttribute("affordcraft:emptyStaticFrame", Sdf.ValueTypeNames.Bool).Set(True)
        prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
for j in spec["joints"]:
    prim = stage.GetPrimAtPath(j["path"])
    if j["type"] == "spherical":
        attrs = {
            a.GetName(): (a.Get(), a.GetTypeName())
            for a in prim.GetAttributes()
            if a.GetName().startswith("physics:local")
        }
        rels = {r.GetName(): list(r.GetTargets()) for r in prim.GetRelationships()}
        stage.RemovePrim(j["path"])
        sp = UsdPhysics.SphericalJoint.Define(stage, j["path"])
        sprim = sp.GetPrim()
        for k, (v, tn) in attrs.items():
            (sprim.GetAttribute(k) if sprim.GetAttribute(k) else sprim.CreateAttribute(k, tn)).Set(v)
        for k, v in rels.items():
            (sprim.GetRelationship(k) if sprim.GetRelationship(k) else sprim.CreateRelationship(k)).SetTargets(v)
        continue
    if j["set_axis_x"]:
        prim.GetAttribute("physics:axis").Set("X")
    if j["continuous"]:
        prim.CreateAttribute("affordcraft:continuous", Sdf.ValueTypeNames.Bool).Set(True)
floor = UsdGeom.Cube.Define(stage, "/World/Floor")
floor.CreateSizeAttr(1)
floor.AddTranslateOp().Set(Gf.Vec3f(0, 0, -0.02))
floor.AddScaleOp().Set(Gf.Vec3f(spec["floor"]["sx"], spec["floor"]["sy"], 0.04))
UsdPhysics.CollisionAPI.Apply(floor.GetPrim())
floor.GetPrim().CreateAttribute("affordcraft:supportSurface", Sdf.ValueTypeNames.Bool).Set(True)
stage.GetRootLayer().Save()
print("ok")
