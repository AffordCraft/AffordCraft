"""Concrete backend with isolated CPU import and two independent PhysX processes."""

from pathlib import Path
import hashlib, json, os, subprocess, sys, time, uuid
from PIL import Image
from .contracts import SupportContract, fingerprint
from .catalog import VisualEncoder, CatalogIndex, file_sha
from .vision import LocalVisualBackend
from .search import InfrastructureBlocked
from .execution import known_coacd_abort, health_probe, save_new
from . import paths


class PhysicsServices:
    def __init__(self, code_root, out, gpu):
        self.code_root = Path(code_root)
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=False)
        self.children = {}
        self.logs = {}
        env = {**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}
        env.pop("CUDA_VISIBLE_DEVICES", None)
        for phase in ("construction", "final"):
            root = self.out / phase
            root.mkdir()
            queue = root / "queue"
            queue.mkdir()
            log = (root / "stdout.log").open("w")
            self.logs[phase] = log
            cmd = [
                paths.ISAAC_PYTHON,
                str(self.code_root / "scripts/physics_worker.py"),
                "--queue",
                str(queue),
                "--owner-pid",
                str(os.getpid()),
                "--output",
                str(root / "runtime"),
                "--gpu",
                str(gpu),
            ]
            child = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, env=env)
            self.children[phase] = child
            (root / "process.json").write_text(
                json.dumps(
                    {
                        "pid": child.pid,
                        "parent_pid": os.getpid(),
                        "argv": cmd,
                        "gpu": gpu,
                        "phase": phase,
                        "independent_process": True,
                        "license_acceptance_performed": False,
                    },
                    indent=2,
                )
            )
        start = time.time()
        while time.time() - start < 240:
            if any(c.poll() is not None for c in self.children.values()):
                raise InfrastructureBlocked("physics_service_start_failed")
            if all((self.out / p / "queue/ready.json").exists() for p in self.children):
                break
            time.sleep(1)
        else:
            raise InfrastructureBlocked("physics_service_start_timeout")

    def evaluate(self, asset, phase):
        if asset.get("exported") is not True or not asset.get("usd") or not asset.get("sha256"):
            raise InfrastructureBlocked("non_exported_asset_sent_to_physics")
        child = self.children[phase]
        if child.poll() is not None:
            raise InfrastructureBlocked("physics_service_terminated")
        q = self.out / phase / "queue"
        job_id = "job_" + uuid.uuid4().hex
        job = {
            k: asset[k]
            for k in (
                "usd",
                "sha256",
                "task_metadata",
                "geometry_source_verified",
                "expected_internal_joints",
                "requires_articulation",
                "support_surface_z",
                "expected_joint_axes_world",
            )
            if k in asset
        }
        job["job_id"] = job_id
        request = q / "requests" / (job_id + ".json")
        tmp = request.with_suffix(".tmp")
        tmp.write_text(json.dumps(job))
        tmp.replace(request)
        tick = time.monotonic()
        response = q / "responses" / request.name
        while not response.exists():
            if child.poll() is not None:
                raise InfrastructureBlocked("physics_process_failed_during_request")
            if time.monotonic() - tick > 300:
                raise InfrastructureBlocked("physics_request_timeout")
            time.sleep(0.1)
        d = json.loads(response.read_text())
        if d["request_sha256"] != file_sha(request):
            raise InfrastructureBlocked("physics_request_response_mismatch")
        r = d["result"]
        if type(r.get("physical_pass")) is not bool:
            raise InfrastructureBlocked("physics_measurement_unavailable:" + str(r.get("error")))
        return {**r, "phase": phase, "evidence_directory": d["evidence_directory"], "service_pid": child.pid}

    def close(self):
        outcomes = {}
        for phase, child in self.children.items():
            (self.out / phase / "queue/STOP").touch(exist_ok=False)
        for phase, child in self.children.items():
            try:
                rc = child.wait(timeout=120)
            except subprocess.TimeoutExpired:
                rc = None
            self.logs[phase].close()
            outcomes[phase] = {
                "pid": child.pid,
                "exit_code": rc,
                "in_process_completed": (self.out / phase / "runtime/in_process_completion.json").exists(),
            }
        (self.out / "process_outcomes.json").write_text(json.dumps(outcomes, indent=2))
        return outcomes


class RuntimeBackend:
    def __init__(self, project, index_root, output, services, code_root, variant="full"):
        self.project = Path(project)
        self.out = Path(output)
        self.out.mkdir(parents=True, exist_ok=True)
        self.services = services
        self.code_root = Path(code_root)
        self.variant = variant
        model = paths.DINOV2_ROOT
        if variant == "encoder_replacement":
            from .catalog import CLIPImageEncoder

            self.encoder = CLIPImageEncoder(paths.CLIP_VIT_B32)
        else:
            self.encoder = VisualEncoder(model / "image_encoder_dinov2", model / "feature_extractor_dinov2")
        self.index = CatalogIndex(
            project,
            paths.catalog_path(self.project),
            index_root,
            self.encoder,
        )
        self.vision = LocalVisualBackend(paths.QWEN3_VL, self.out / "model_calls")
        self.requests = {}
        self.serial = 0
        self.task_metadata = {}

    def set_context(self, case):
        # Passed separately from the allowlisted model inputs. Never sourced from a model reply.
        self.task_metadata = case.get("task_metadata", {})

    def ground(self, case):
        return self.vision.ground(case)

    def retrieve(self, case, grounding, limit):
        im = Image.open(case["image"]["path"]).convert("RGB")
        w, h = im.size
        bb = grounding["bbox_normalized"]
        crop = im.crop((int(bb[0] * w), int(bb[1] * h), int(bb[2] * w), int(bb[3] * h)))
        # Without task conditioning the requested category never touches retrieval or ranking.
        category = None if self.variant == "without_task_condition" else grounding["category_for_retrieval"]
        return self.index.retrieve(crop, category, limit)

    def rank(self, case, g, candidates):
        return self.vision.rank(
            case, g, candidates, self.project, task_conditioned=self.variant != "without_task_condition"
        )

    def _materialize(self, request):
        self.serial += 1
        key = f"build_{self.serial:07}"
        folder = self.out / "builds" / key
        folder.parent.mkdir(parents=True, exist_ok=True)
        req = self.out / "build_requests" / (key + ".json")
        req.parent.mkdir(parents=True, exist_ok=True)
        req.write_text(json.dumps(request), encoding="utf-8")
        cmd = [
            paths.RUNTIME_PYTHON,
            str(self.code_root / "scripts/materialize_asset.py"),
            "--request",
            str(req),
            "--output",
            str(folder),
            "--cache",
            str(getattr(self, "geometry_cache", self.out / "geometry_cache")),
        ]
        t = time.perf_counter()
        log = folder.parent / (key + ".log")
        with log.open("w") as f:
            # Thread count of the CPU builder is an execution setting recorded by the campaign
            # (AFFORDCRAFT_BUILD_THREADS); it never changes cascade parameters, audits or caps.
            threads = str(int(os.environ.get("AFFORDCRAFT_BUILD_THREADS", "2")))
            try:
                p = subprocess.run(
                    cmd,
                    stdin=subprocess.DEVNULL,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    env={
                        **os.environ,
                        "OMP_NUM_THREADS": threads,
                        "MKL_NUM_THREADS": threads,
                        "OPENBLAS_NUM_THREADS": threads,
                        "CUDA_VISIBLE_DEVICES": "",
                        "PYTHONFAULTHANDLER": "1",
                        "PYTHONDONTWRITEBYTECODE": "1",
                    },
                    timeout=600,
                )
            except subprocess.TimeoutExpired:
                r = {
                    "exported": False,
                    "status": "construction_failed",
                    "candidate_id": request["candidate"]["candidate_id"],
                    "reason": "materialization_time_budget_exceeded_600s",
                    "request_file": str(req),
                    "process_exit_code": None,
                    "total_materialization_wall_seconds": time.perf_counter() - t,
                    "physics_attempted": False,
                    "timeout_is_not_a_physical_gate_failure": True,
                }
                folder.mkdir(parents=True, exist_ok=True)
                (folder / "build_timeout.json").write_text(json.dumps(r, indent=2))
                self.requests[str(req)] = request
                return r
        result = folder / "build_result.json"
        evidence = {
            "command": cmd,
            "exit_code": p.returncode,
            "request_sha256": file_sha(req),
            "log_path": str(log),
            "log_sha256": file_sha(log),
            "result_present": result.exists(),
            "runtime_seconds": time.perf_counter() - t,
            "candidate_id": request["candidate"]["candidate_id"],
        }
        folder.mkdir(parents=True, exist_ok=True)
        if p.returncode != 0 or not result.exists():
            isolated = known_coacd_abort(p.returncode, log.read_text(errors="replace"))
            if isolated:
                probe = health_probe(self.code_root, cmd[0], folder / "coacd_health_after_abort")
                evidence["post_crash_health"] = probe
                isolated = probe["passed"]
            evidence["candidate_local_failure_confirmed"] = isolated
            save_new(folder / "materialization_process_receipt.json", evidence)
            if not isolated:
                raise InfrastructureBlocked(
                    "materialization_process_failed:" + str(folder / "materialization_process_receipt.json")
                )
            r = {
                "exported": False,
                "status": "construction_failed",
                "candidate_id": request["candidate"]["candidate_id"],
                "reason": "collision_decomposition_native_abort",
                "request_file": str(req),
                "process_exit_code": p.returncode,
                "total_materialization_wall_seconds": time.perf_counter() - t,
                "physics_attempted": False,
                "crash_not_counted_as_physics_failure": True,
                "process_receipt": str(folder / "materialization_process_receipt.json"),
            }
            save_new(folder / "build_failure_classification.json", r)
            self.requests[str(req)] = request
            return r
        save_new(folder / "materialization_process_receipt.json", evidence)
        r = json.loads(result.read_text())
        r.update(
            request_file=str(req),
            process_exit_code=p.returncode,
            total_materialization_wall_seconds=time.perf_counter() - t,
        )
        self.requests[str(req)] = request
        return r

    def build(self, case, grounding, candidate):
        structured = bool(candidate.get("urdf")) or candidate.get("structural", {}).get("joint_count", 0) > 0
        if self.variant == "without_articulation_adaptation" and structured:
            # The pre-adapter route: structured (jointed) sources are never imported, so they cannot export.
            return {
                "exported": False,
                "status": "construction_failed",
                "candidate_id": candidate["candidate_id"],
                "reason": "articulation_adapter_disabled",
                "variant": self.variant,
            }
        if self.variant == "without_scale_adaptation":
            grounding = {**grounding, "size_m": None, "size_confidence": 0.0, "scale_adaptation": "disabled_by_variant"}
        # Installation hints from the model are deliberately NOT promoted to proof. Only
        # dataset-declared installation metadata (hashed artifact) can authorize a mount.
        metadata = dict(self.task_metadata)
        declared = dict(metadata.get("affordcraft_support", {}))
        from .build import source_installation_evidence, parser

        # without_installation_evidence: every object is placed free-standing (the behaviour before
        # installation evidence was added); the dataset-declared installation metadata is never consulted.
        evidence = (
            None
            if self.variant == "without_installation_evidence"
            else source_installation_evidence(self.project, candidate, parser(self.project)[0])
        )
        if evidence is not None and declared.get("root_motion_required") is not True:
            declared.update(role="mounted", evidence=list(declared.get("evidence", [])) + [evidence])
            metadata["affordcraft_support"] = declared
        support = SupportContract.from_task_metadata(metadata)
        meta = support.task_metadata()
        request = {
            "project": str(self.project),
            "input_id": case["input_id"],
            "candidate": candidate,
            "grounding": grounding,
            "task_metadata": meta,
            "variant": self.variant,
        }
        return self._materialize(request)

    def screen(self, asset, grounding):
        return self.services.evaluate(asset, "construction")

    def repair(self, asset, grounding, check):
        request = self.requests.get(asset.get("request_file"))
        if request is None or request.get("repair_mode"):
            return None
        # Rebuilding collision from the visual surface is only an alternative when the
        # source supplies a distinct collision mesh; otherwise the rebuild is byte-identical.
        if (asset.get("diagnostics") or {}).get("visual_equals_collision") is True:
            return None
        reasons = " ".join(check.get("failure_reasons", []))
        if "repair_no_alternative_source_geometry" in reasons or "cascade_budget_exhausted" in reasons:
            return None
        eligible = any(x in reasons for x in ("penetration", "collision", "settle", "decomposition"))
        if not eligible:
            return None
        return self._materialize({**request, "repair_mode": "rebuild_collision_from_source_visual"})

    def validate_final(self, asset, grounding):
        return self.services.evaluate(asset, "final")


class SceneRuntimeBackend(RuntimeBackend):
    def set_context(self, case):
        super().set_context(case)
        self.automatic_case = case

    def ground(self, case):
        from .vision import parse_json, normalized_box

        c = self.automatic_case
        provenance = c["automatic_grounding_provenance"]
        g = c["automatic_grounding"]
        if (
            provenance["reference_annotations_used"] is not False
            or provenance["scene_input_sha256"] != case["image"]["sha256"]
        ):
            raise InfrastructureBlocked("Untrusted localization provenance")
        if file_sha(case["image"]["path"]) != case["image"]["sha256"]:
            raise InfrastructureBlocked("input_image_hash_drift")
        call = json.loads(
            (self.detection_root / "model_calls" / f"call_{provenance['model_call_index']:07}.json").read_text()
        )
        items = parse_json(call["response"])["objects"]
        index = int(c["input_id"].rsplit("__det_", 1)[1])
        if (
            items[index]["bbox_1000"] != g["bbox_1000"]
            or normalized_box(items[index]["bbox_1000"]) != g["bbox_normalized"]
        ):
            raise InfrastructureBlocked("Automatic region does not match recorded model output")
        return g
