"""Known positive/negative controls, never experimental-result illustrations."""

from pathlib import Path
import argparse, hashlib, json, sys
from pxr import Usd, UsdGeom, UsdPhysics, Gf, Sdf

R = Path(__file__).resolve().parents[1]


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--output", required=True)
    args = a.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    spec = out / "fixture_specification.json"
    spec.write_text(
        json.dumps(
            {
                "purpose": "pre-evaluation software/physics calibration",
                "expected": [
                    "free_box:pass",
                    "pinned_free_box:fail",
                    "mounted_hinge:pass",
                    "unsupported_mount:fail",
                    "penetrating_box:fail",
                    "fixed_free_box:fail",
                    "locked_part:fail",
                ],
                "fixed_hinge_requires_mount": True,
            },
            indent=2,
        )
    )
    jobs = []
    for kind in [
        "free_box",
        "pinned_free_box",
        "mounted_hinge",
        "unsupported_mount",
        "penetrating_box",
        "fixed_free_box",
        "locked_part",
    ]:
        p = out / (kind + ".usda")
        s = Usd.Stage.CreateNew(str(p))
        UsdGeom.SetStageUpAxis(s, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(s, 1.0)
        root = UsdGeom.Xform.Define(s, "/World")
        s.SetDefaultPrim(root.GetPrim())
        scene = UsdPhysics.Scene.Define(s, "/World/Physics")
        scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
        scene.CreateGravityMagnitudeAttr(9.81)
        ground = UsdGeom.Cube.Define(s, "/World/Floor")
        ground.CreateSizeAttr(1)
        ground.AddTranslateOp().Set(Gf.Vec3f(0, 0, -0.02))
        ground.AddScaleOp().Set(Gf.Vec3f(5, 5, 0.04))
        UsdPhysics.CollisionAPI.Apply(ground.GetPrim())
        ground.GetPrim().CreateAttribute("affordcraft:supportSurface", Sdf.ValueTypeNames.Bool).Set(True)
        mounted = kind in ["mounted_hinge", "unsupported_mount", "locked_part"]
        body = UsdGeom.Cube.Define(s, "/World/Body")
        body.CreateSizeAttr(0.1)
        body.AddTranslateOp().Set(
            Gf.Vec3d(0.1 if mounted else 0, 0, 1.0 if mounted else (0.04 if kind == "penetrating_box" else 0.052))
        )
        api = UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
        api.CreateRigidBodyEnabledAttr(True)
        api.CreateKinematicEnabledAttr(kind == "pinned_free_box")
        UsdPhysics.CollisionAPI.Apply(body.GetPrim())
        m = UsdPhysics.MassAPI.Apply(body.GetPrim())
        m.CreateMassAttr(1.0)
        m.CreateCenterOfMassAttr(Gf.Vec3f(0))
        m.CreateDiagonalInertiaAttr(Gf.Vec3f(0.0016666667))
        m.CreatePrincipalAxesAttr(Gf.Quatf(1))
        meta = {}
        expected_internal = 0
        if mounted:
            frame = UsdGeom.Xform.Define(s, "/World/Mount")
            frame.AddTranslateOp().Set(Gf.Vec3d(0, 0, 1))
            fa = UsdPhysics.RigidBodyAPI.Apply(frame.GetPrim())
            fa.CreateKinematicEnabledAttr(True)
            fa.CreateRigidBodyEnabledAttr(True)
            frame.GetPrim().CreateAttribute("affordcraft:emptyStaticFrame", Sdf.ValueTypeNames.Bool).Set(True)
            if kind == "locked_part":
                j = UsdPhysics.FixedJoint.Define(s, "/World/Hinge")
            else:
                j = UsdPhysics.RevoluteJoint.Define(s, "/World/Hinge")
                j.CreateAxisAttr("Z")
                j.CreateLowerLimitAttr(-90)
                j.CreateUpperLimitAttr(90)
            j.CreateBody0Rel().SetTargets(["/World/Mount"])
            j.CreateBody1Rel().SetTargets(["/World/Body"])
            j.CreateLocalPos0Attr(Gf.Vec3f(0))
            j.CreateLocalPos1Attr(Gf.Vec3f(-0.1, 0, 0))
            expected_internal = 1
            if kind != "unsupported_mount":
                meta = {
                    "affordcraft_support": {
                        "role": "mounted",
                        "evidence": [
                            {
                                "kind": "explicit_task_installation",
                                "assertion": "fixed_installation_required",
                                "verified": True,
                                "artifact": {
                                    "path": str(spec),
                                    "sha256": hashlib.sha256(spec.read_bytes()).hexdigest(),
                                },
                            }
                        ],
                    }
                }
        if kind == "fixed_free_box":
            j = UsdPhysics.FixedJoint.Define(s, "/World/UnjustifiedAnchor")
            j.CreateBody1Rel().SetTargets(["/World/Body"])
            j.CreateLocalPos0Attr(Gf.Vec3f(0, 0, 0.052))
            j.CreateLocalPos1Attr(Gf.Vec3f(0))
        s.GetRootLayer().Save()
        jobs.append(
            {
                "job_id": kind,
                "usd": str(p),
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                "task_metadata": meta,
                "geometry_source_verified": True,
                "expected_internal_joints": expected_internal,
                "requires_articulation": mounted,
                "support_surface_z": 0.0,
                "expected_pass": kind in ["free_box", "mounted_hinge"],
            }
        )
    (out / "jobs.json").write_text(json.dumps(jobs, indent=2))
    print(json.dumps({"jobs": str(out / "jobs.json"), "n": len(jobs)}))


if __name__ == "__main__":
    main()
