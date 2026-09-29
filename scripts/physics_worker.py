"""Actual Isaac/PhysX evidence collector. No license acceptance, post-play pose edits or robot rollout."""

from pathlib import Path
import argparse, hashlib, json, math, os, sys, time, traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from affordcraft.contracts import EvaluationPolicy, SupportContract
from affordcraft.physics import evaluate_trace, ContactLifecycle, qangle, distance


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(p, v):
    p.parent.mkdir(parents=True, exist_ok=True)

    def json_safe(x):
        if isinstance(x, float) and not math.isfinite(x):
            return {"nonfinite_value": repr(x)}
        if isinstance(x, dict):
            return {k: json_safe(y) for k, y in x.items()}
        if isinstance(x, (tuple, list)):
            return [json_safe(y) for y in x]
        return x

    with p.open("x") as f:
        json.dump(json_safe(v), f, indent=2, allow_nan=False)


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--jobs")
    a.add_argument("--queue")
    a.add_argument("--owner-pid", type=int)
    a.add_argument("--output", required=True)
    a.add_argument("--gpu", type=int, default=0)
    args = a.parse_args()
    if bool(args.jobs) == bool(args.queue):
        raise ValueError("Specify a jobs file OR one private queue")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    save(
        out / "launch.json",
        {
            "argv": sys.argv,
            "pid": os.getpid(),
            "gpu": args.gpu,
            "jobs_sha256": sha(args.jobs) if args.jobs else None,
            "queue": args.queue,
            "policy": EvaluationPolicy().to_dict(),
            "code_sha256": sha(__file__),
            "license_acceptance_performed": False,
        },
    )
    from isaacsim import SimulationApp

    app = SimulationApp(
        {
            "headless": True,
            "hide_ui": True,
            "width": 640,
            "height": 480,
            "multi_gpu": False,
            "max_gpu_count": 1,
            "active_gpu": args.gpu,
            "physics_gpu": args.gpu,
            "fast_shutdown": True,
        }
    )
    import numpy as np
    import omni.usd
    from isaacsim.core.api import World
    from isaacsim.core.utils.stage import open_stage, create_new_stage, is_stage_loading
    from omni.physx import get_physx_simulation_interface
    from omni.physx.bindings._physx import ContactEventType
    from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, PhysicsSchemaTools, Gf

    policy = EvaluationPolicy()
    results = []

    def queued_jobs():
        q = Path(args.queue)
        q.mkdir(parents=True, exist_ok=True)
        (q / "requests").mkdir(exist_ok=True)
        (q / "responses").mkdir(exist_ok=True)
        (q / "ready.json").write_text(
            json.dumps({"pid": os.getpid(), "code_sha256": sha(__file__), "output": str(out)})
        )
        idle_start = time.time()
        while not (q / "STOP").exists():
            todo = [p for p in sorted((q / "requests").glob("*.json")) if not (q / "responses" / p.name).exists()]
            if not todo:
                if args.owner_pid:
                    try:
                        os.kill(args.owner_pid, 0)
                    except ProcessLookupError:
                        break
                elif time.time() - idle_start > 600:
                    break
                time.sleep(0.1)
                continue
            for path in todo:
                job = json.loads(path.read_text())
                job["_queue_response"] = str(q / "responses" / path.name)
                job["_request_sha256"] = sha(path)
                if not job["job_id"].replace("_", "").replace("-", "").isalnum():
                    raise ValueError("unsafe job identifier")
                yield job
                idle_start = time.time()

    jobs = json.loads(Path(args.jobs).read_text()) if args.jobs else queued_jobs()

    def has_shape(p):
        return any(q.IsA(UsdGeom.Gprim) for q in Usd.PrimRange(p))

    def matrix(p):
        return omni.usd.get_world_transform_matrix(p)

    def pose(p):
        m = matrix(p)
        q = m.ExtractRotationQuat()
        v = q.GetImaginary()
        return {
            "position": [float(x) for x in m.ExtractTranslation()],
            "orientation_wxyz": [float(q.GetReal()), *[float(x) for x in v]],
        }

    try:
        for job in jobs:
            dest = out / job["job_id"]
            dest.mkdir()
            start = time.perf_counter()
            world = sub = None
            try:
                asset = Path(job["usd"])
                before = sha(asset)
                if before != job["sha256"]:
                    raise RuntimeError("source_asset_hash_drift")
                if not open_stage(str(asset)):
                    raise RuntimeError("stage_open_failed")
                for _ in range(800):
                    app.update()
                    if not is_stage_loading():
                        break
                if is_stage_loading():
                    raise RuntimeError("stage_loading_timeout")
                stage = omni.usd.get_context().get_stage()
                stage.SetEditTarget(stage.GetSessionLayer())
                floor_paths = [
                    str(p.GetPath())
                    for p in stage.Traverse()
                    if p.GetAttribute("affordcraft:supportSurface").Get() is True
                ]
                bodies = [
                    p
                    for p in stage.Traverse()
                    if p.HasAPI(UsdPhysics.RigidBodyAPI)
                    and not any(str(p.GetPath()).startswith(f) for f in floor_paths)
                ]
                joints = [p for p in stage.Traverse() if p.IsA(UsdPhysics.Joint)]
                bodypaths = [str(p.GetPath()) for p in bodies]
                bodymap = {str(p.GetPath()): p for p in bodies}
                data = []
                gravity = True
                for p in bodies:
                    api = UsdPhysics.RigidBodyAPI(p)
                    mass = UsdPhysics.MassAPI(p)
                    i = mass.GetDiagonalInertiaAttr().Get()
                    com = mass.GetCenterOfMassAttr().Get()
                    empty = p.GetAttribute("affordcraft:emptyStaticFrame").Get() is True
                    role = "coordinate_frame" if empty else "physical_body"
                    data.append(
                        {
                            "id": str(p.GetPath()),
                            "role": role,
                            "has_geometry": has_shape(p),
                            "kinematic": api.GetKinematicEnabledAttr().Get() is True,
                            "source_declared_empty_static_frame": empty,
                            "mass": float(mass.GetMassAttr().Get()) if mass.GetMassAttr().Get() is not None else None,
                            "center_of_mass": [float(x) for x in com] if com is not None else None,
                            "inertia_diagonal": [float(x) for x in i] if i is not None else None,
                        }
                    )
                    if role == "physical_body":
                        gravity &= PhysxSchema.PhysxRigidBodyAPI(p).GetDisableGravityAttr().Get() is not True
                internal = []
                anchors = []
                edges = []
                frames_valid = True
                limits_valid = True
                movable = 0
                for p in joints:
                    j = UsdPhysics.Joint(p)
                    aa = j.GetBody0Rel().GetTargets()
                    bb = j.GetBody1Rel().GetTargets()
                    pa = str(aa[0]) if aa else None
                    pb = str(bb[0]) if bb else None
                    if not pa or not pb:
                        anchors.append(str(p.GetPath()))
                    else:
                        internal.append(str(p.GetPath()))
                        if j.GetJointEnabledAttr().Get() is not False:
                            edges.append((pa, pb))
                    for endpoint in [pa, pb]:
                        if endpoint and endpoint not in bodymap:
                            frames_valid = False
                    p0 = j.GetLocalPos0Attr().Get()
                    p1 = j.GetLocalPos1Attr().Get()
                    if p0 is not None and p1 is not None:
                        w0 = matrix(bodymap[pa]).Transform(Gf.Vec3d(p0)) if pa in bodymap else Gf.Vec3d(p0)
                        w1 = matrix(bodymap[pb]).Transform(Gf.Vec3d(p1)) if pb in bodymap else Gf.Vec3d(p1)
                        delta = np.asarray(w1) - np.asarray(w0)
                        if p.IsA(UsdPhysics.PrismaticJoint):
                            qq = j.GetLocalRot0Attr().Get()
                            token = str(p.GetAttribute("physics:axis").Get())
                            vv = Gf.Vec3d(1 if token == "X" else 0, 1 if token == "Y" else 0, 1 if token == "Z" else 0)
                            axis = Gf.Quatd(float(qq.GetReal()), Gf.Vec3d(qq.GetImaginary())).Transform(vv)
                            if pa in bodymap:
                                axis = matrix(bodymap[pa]).TransformDir(axis)
                            axis = np.asarray(axis)
                            axis /= np.linalg.norm(axis)
                            delta -= np.dot(delta, axis) * axis
                        frames_valid &= bool(np.isfinite(delta).all() and np.linalg.norm(delta) <= 1e-4)
                    if p.IsA(UsdPhysics.RevoluteJoint) or p.IsA(UsdPhysics.PrismaticJoint):
                        lo = p.GetAttribute("physics:lowerLimit").Get()
                        hi = p.GetAttribute("physics:upperLimit").Get()
                        continuous = p.GetAttribute("affordcraft:continuous").Get() is True
                        valid = continuous or (
                            lo is not None and hi is not None and math.isfinite(lo) and math.isfinite(hi) and lo < hi
                        )
                        limits_valid &= valid
                        if valid and j.GetJointEnabledAttr().Get() is not False:
                            movable += 1
                        expected = job.get("expected_joint_axes_world", {}).get(str(p.GetPath()))
                        if expected is not None:
                            token = str(p.GetAttribute("physics:axis").Get())
                            v = Gf.Vec3d(1 if token == "X" else 0, 1 if token == "Y" else 0, 1 if token == "Z" else 0)
                            q0 = j.GetLocalRot0Attr().Get()
                            rq = Gf.Quatd(float(q0.GetReal()), Gf.Vec3d(q0.GetImaginary()))
                            actual = rq.Transform(v)
                            if pa in bodymap:
                                actual = matrix(bodymap[pa]).TransformDir(actual)
                            av = np.asarray(actual)
                            ev = np.asarray(expected)
                            frames_valid &= bool(
                                np.isfinite(av).all()
                                and np.linalg.norm(av) > 0
                                and np.linalg.norm(av / np.linalg.norm(av) - ev) < 1e-5
                            )
                # A kinematic node is an anchor only if reachable by a valid constraint chain.
                seeds = {d["id"] for d in data if d["kinematic"]}
                reachable = set(seeds)
                for p in joints:
                    j = UsdPhysics.Joint(p)
                    aa = j.GetBody0Rel().GetTargets()
                    bb = j.GetBody1Rel().GetTargets()
                    if j.GetJointEnabledAttr().Get() is False:
                        continue
                    if not aa:
                        reachable.update(map(str, bb))
                    if not bb:
                        reachable.update(map(str, aa))
                for _ in range(len(bodies) + 1):
                    for x, y in edges:
                        if x in reachable:
                            reachable.add(y)
                        if y in reachable:
                            reachable.add(x)
                physical_ids = {d["id"] for d in data if d["role"] == "physical_body"}
                required_motion = job.get("requires_articulation") is True
                support = SupportContract.from_task_metadata(job.get("task_metadata", {}))
                for e in support.evidence:
                    ref = e.get("artifact", {})
                    if support.role == "mounted" and (
                        not Path(ref["path"]).is_file() or sha(ref["path"]) != ref["sha256"]
                    ):
                        raise RuntimeError("mount_evidence_hash_invalid")
                collisions = [
                    p
                    for p in stage.Traverse()
                    if p.HasAPI(UsdPhysics.CollisionAPI)
                    and not any(str(p.GetPath()).startswith(f) for f in floor_paths)
                ]
                static = {
                    "stage_load": True,
                    "visual_geometry": all(d["has_geometry"] for d in data if d["role"] == "physical_body"),
                    "collision_geometry": bool(collisions),
                    "collision_enabled": all(
                        UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get() is not False for p in collisions
                    ),
                    "units_frames_valid": abs(UsdGeom.GetStageMetersPerUnit(stage) - 1) < 1e-9
                    and str(UsdGeom.GetStageUpAxis(stage)) == "Z",
                    "source_geometry_preserved": job.get("geometry_source_verified") is True,
                    "internal_joint_topology_preserved": len(internal) == job.get("expected_internal_joints", 0),
                    "joint_frames_valid": frames_valid,
                    "joint_limits_valid": limits_valid,
                    "required_mobility_preserved": not required_motion or movable > 0,
                    "gravity_enabled": gravity,
                    "no_post_play_transform_writeback": True,
                    "bodies": data,
                    "external_anchor_present": bool(seeds or anchors),
                    "anchor_chain_valid": bool(physical_ids) and physical_ids.issubset(reachable),
                    "contact_instrumentation_valid": True,
                    "support_surface_z": job.get("support_surface_z", 0.0),
                }
                # Measure before World.reset()/play can resolve a penetrating starting state.
                bounds_cache = UsdGeom.BBoxCache(
                    Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy]
                )
                initial_z = [
                    float(bounds_cache.ComputeWorldBound(p).ComputeAlignedRange().GetMin()[2]) for p in collisions
                ]
                static["support_plane_present"] = bool(floor_paths)
                static["initial_floor_clearance_m"] = (
                    min(initial_z) - float(job.get("support_surface_z", 0.0)) if initial_z else None
                )
                for p in stage.Traverse():
                    if p.HasAPI(UsdPhysics.RigidBodyAPI) or p.HasAPI(UsdPhysics.CollisionAPI):
                        PhysxSchema.PhysxContactReportAPI.Apply(p).CreateThresholdAttr().Set(0.0)
                state = {"step": 0}
                events = []
                trace = []
                life = ContactLifecycle()

                def matches(path, roots):
                    return any(path == r or path.startswith(r + "/") for r in roots)

                types = {
                    int(ContactEventType.CONTACT_FOUND): "CONTACT_FOUND",
                    int(ContactEventType.CONTACT_PERSIST): "CONTACT_PERSIST",
                    int(ContactEventType.CONTACT_LOST): "CONTACT_LOST",
                }

                def callback(headers, contactdata):
                    for h in headers:
                        actors = [str(PhysicsSchemaTools.intToSdfPath(x)) for x in [h.actor0, h.actor1]]
                        cols = [str(PhysicsSchemaTools.intToSdfPath(x)) for x in [h.collider0, h.collider1]]
                        ob = [matches(actors[i], bodypaths) or matches(cols[i], bodypaths) for i in (0, 1)]
                        fl = [matches(actors[i], floor_paths) or matches(cols[i], floor_paths) for i in (0, 1)]
                        kind = "support" if (ob[0] and fl[1]) or (ob[1] and fl[0]) else "self" if all(ob) else "other"
                        if kind == "other":
                            continue
                        samples = [
                            {
                                "separation": float(contactdata[i].separation),
                                "impulse": [float(v) for v in contactdata[i].impulse],
                            }
                            for i in range(h.contact_data_offset, h.contact_data_offset + h.num_contact_data)
                        ]
                        e = {
                            "step": state["step"],
                            "kind": kind,
                            "colliders": cols,
                            "actors": actors,
                            "event_type": types.get(int(h.type), "unknown"),
                            "samples": samples,
                        }
                        events.append(e)
                        life.feed(e)

                sub = get_physx_simulation_interface().subscribe_contact_report_events(callback)
                world = World(physics_dt=policy.dt, rendering_dt=1 / 30, stage_units_in_meters=1.0)
                world.reset()
                world.play()
                time_origin = float(world.current_time)
                import omni.physics.tensors as tensors

                simulation_view = tensors.create_simulation_view("numpy")
                simulation_view.set_subspace_roots("/")
                registered_types = {p: str(simulation_view.get_object_type(p)) for p in physical_ids}
                static["runtime_body_types"] = registered_types
                static["runtime_bodies_registered"] = bool(physical_ids) and all(
                    v.rsplit(".", 1)[-1] in ("RigidBody", "ArticulationLink", "ArticulationRootLink")
                    for v in registered_types.values()
                )
                gravity_scenes = [UsdPhysics.Scene(p) for p in stage.Traverse() if p.IsA(UsdPhysics.Scene)]
                static["gravity_enabled"] &= bool(gravity_scenes) and all(
                    abs(float(g.GetGravityMagnitudeAttr().Get()) - 9.81) < 1e-3
                    and np.linalg.norm(np.asarray(g.GetGravityDirectionAttr().Get()) - np.array([0, 0, -1])) < 1e-6
                    for g in gravity_scenes
                )
                initial = {str(p.GetPath()): pose(p) for p in bodies}
                for step in range(1, policy.steps + 1):
                    state["step"] = step
                    world.step(render=False)
                    trace.append(
                        {
                            "step": step,
                            "simulation_seconds": float(world.current_time) - time_origin,
                            "bodies": {str(p.GetPath()): pose(p) for p in bodies},
                            "active_support_constraint": life.supported(),
                        }
                    )
                static["anchor_pose_stable"] = bool(seeds or anchors) and all(
                    distance(initial[p]["position"], trace[-1]["bodies"][p]["position"]) <= 1e-4
                    and qangle(initial[p]["orientation_wxyz"], trace[-1]["bodies"][p]["orientation_wxyz"]) <= 0.01
                    for p in seeds
                )
                consistency = True
                residuals = {}
                for p in joints:
                    j = UsdPhysics.Joint(p)
                    aa = j.GetBody0Rel().GetTargets()
                    bb = j.GetBody1Rel().GetTargets()
                    if j.GetJointEnabledAttr().Get() is False:
                        consistency = False
                        continue
                    a0 = str(aa[0]) if aa else None
                    b0 = str(bb[0]) if bb else None
                    pos0 = j.GetLocalPos0Attr().Get()
                    pos1 = j.GetLocalPos1Attr().Get()
                    w0 = matrix(bodymap[a0]).Transform(Gf.Vec3d(pos0)) if a0 in bodymap else Gf.Vec3d(pos0)
                    w1 = matrix(bodymap[b0]).Transform(Gf.Vec3d(pos1)) if b0 in bodymap else Gf.Vec3d(pos1)
                    delta = np.asarray(w1) - np.asarray(w0)
                    if p.IsA(UsdPhysics.PrismaticJoint):
                        token = str(p.GetAttribute("physics:axis").Get())
                        v = Gf.Vec3d(1 if token == "X" else 0, 1 if token == "Y" else 0, 1 if token == "Z" else 0)
                        q0 = j.GetLocalRot0Attr().Get()
                        axis = Gf.Quatd(float(q0.GetReal()), Gf.Vec3d(q0.GetImaginary())).Transform(v)
                        if a0 in bodymap:
                            axis = matrix(bodymap[a0]).TransformDir(axis)
                        axis = np.asarray(axis)
                        axis /= np.linalg.norm(axis)
                        delta -= np.dot(delta, axis) * axis
                    error = float(np.linalg.norm(delta))
                    residuals[str(p.GetPath())] = error
                    consistency &= math.isfinite(error) and error <= 0.001
                static["constraint_solver_consistency"] = bool(consistency)
                static["joint_pivot_residual_m"] = residuals
                for p in joints:
                    j = UsdPhysics.Joint(p)
                    aa = j.GetBody0Rel().GetTargets()
                    bb = j.GetBody1Rel().GetTargets()
                    if bool(aa) == bool(bb) or j.GetJointEnabledAttr().Get() is False:
                        continue
                    world_pos = j.GetLocalPos0Attr().Get() if not aa else j.GetLocalPos1Attr().Get()
                    local_pos = j.GetLocalPos1Attr().Get() if not aa else j.GetLocalPos0Attr().Get()
                    bp = str(bb[0] if not aa else aa[0])
                    actual = matrix(bodymap[bp]).Transform(Gf.Vec3d(local_pos))
                    delta = np.asarray(actual) - np.asarray(world_pos)
                    if p.IsA(UsdPhysics.PrismaticJoint):
                        q0 = j.GetLocalRot0Attr().Get()
                        token = str(p.GetAttribute("physics:axis").Get())
                        axis = Gf.Vec3d(1 if token == "X" else 0, 1 if token == "Y" else 0, 1 if token == "Z" else 0)
                        axis = np.asarray(Gf.Quatd(float(q0.GetReal()), Gf.Vec3d(q0.GetImaginary())).Transform(axis))
                        delta -= np.dot(delta, axis) * axis
                    static["anchor_pose_stable"] &= bool(np.isfinite(delta).all() and np.linalg.norm(delta) <= 0.001)
                cache = UsdGeom.BBoxCache(
                    Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy]
                )
                zz = [float(cache.ComputeWorldBound(p).ComputeAlignedRange().GetMin()[2]) for p in collisions]
                static["final_collision_min_z"] = min(zz) if zz else None
                result = evaluate_trace(trace, static, events, support, policy)
                result.update(
                    job_id=job["job_id"],
                    usd_sha256=before,
                    elapsed_seconds=time.perf_counter() - start,
                    role_reason=support.reason,
                )
                save(dest / "static.json", static)
                save(dest / "state_trace.json", trace)
                save(dest / "contact_evidence.json", events)
                save(dest / "result.json", result)
                if sha(asset) != before:
                    raise RuntimeError("input_asset_was_mutated")
                results.append(result)
                print(
                    json.dumps(
                        {"job": job["job_id"], "pass": result["physical_pass"], "reasons": result["failure_reasons"]}
                    ),
                    flush=True,
                )
                if job.get("_queue_response"):
                    save(
                        Path(job["_queue_response"]),
                        {"request_sha256": job["_request_sha256"], "result": result, "evidence_directory": str(dest)},
                    )
            except Exception as exc:
                result = {
                    "job_id": job["job_id"],
                    "status": "infrastructure_blocked",
                    "physical_pass": None,
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }
                save(dest / "runtime_error.json", result)
                results.append(result)
                print(json.dumps(result), flush=True)
                if job.get("_queue_response"):
                    save(
                        Path(job["_queue_response"]),
                        {"request_sha256": job["_request_sha256"], "result": result, "evidence_directory": str(dest)},
                    )
            finally:
                if sub:
                    try:
                        sub.unsubscribe()
                    except Exception:
                        pass
                if world:
                    try:
                        world.stop()
                        world.clear()
                    except Exception:
                        pass
                World.clear_instance()
        save(
            out / "in_process_completion.json",
            {
                "results": results,
                "jobs": len(results),
                "passed": sum(r.get("physical_pass") is True for r in results),
                "blocked": sum(r.get("physical_pass") is None for r in results),
                "scope": "physical validation only; no task-match or robot-success claim",
            },
        )
    finally:
        app.close()


if __name__ == "__main__":
    main()
