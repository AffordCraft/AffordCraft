"""Model-neutral bounded construction. Final evaluator output never selects a candidate."""

import time
from .contracts import ConstructionPolicy, clean_model_input, semantic_eligible, fingerprint


class InfrastructureBlocked(RuntimeError):
    pass


class MethodFailure(RuntimeError):
    pass


def normalize_assessments(assessments, batch):
    """Deterministic repair of the selector reply: keep the first assessment per listed ID,
    drop unlisted IDs, and record listed IDs the model omitted as explicit abstentions.
    Nothing is invented: an omitted candidate is never eligible. An empty list (or one naming
    only unlisted IDs) is the selector abstaining on the whole batch: every listed candidate is
    recorded as omitted and the search continues with the next batch; only a non-list reply
    breaks the contract."""
    if not isinstance(assessments, list):
        raise MethodFailure("selector_contract_invalid")
    listed = {str(c["candidate_id"]): c["candidate_id"] for c in batch}
    ordered = []
    seen = set()
    dropped = []
    for row in assessments:
        if not isinstance(row, dict):
            dropped.append({"reason": "not_object"})
            continue
        cid = str(row.get("candidate_id"))
        if cid not in listed:
            dropped.append({"candidate_id": cid, "reason": "unlisted_id"})
            continue
        if cid in seen:
            dropped.append({"candidate_id": cid, "reason": "duplicate_id"})
            continue
        seen.add(cid)
        ordered.append({**row, "candidate_id": listed[cid]})
    omitted = [cid for cid in listed if cid not in seen]
    for cid in omitted:
        ordered.append(
            {
                "candidate_id": listed[cid],
                "category_compatible": None,
                "mechanism_compatible": None,
                "subtype_match": None,
                "confidence": None,
                "geometry_similarity": None,
                "evidence": "omitted_by_selector",
                "selector_omitted": True,
            }
        )
    return ordered, {
        "dropped": dropped,
        "omitted": omitted,
        "normalized": bool(dropped or omitted),
        "batch_abstention": len(omitted) == len(listed),
    }


class BudgetedConstruction:
    def __init__(self, backend, policy=ConstructionPolicy()):
        self.backend = backend
        self.policy = policy

    def run(self, case):
        start = time.perf_counter()
        safe = clean_model_input(case)
        attempts = []
        seen = set()
        timings = []

        def measured(phase, fn, *args, **kwargs):
            tick = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                timings.append(
                    {"phase": phase, "runtime_seconds": time.perf_counter() - tick, "timing_status": "measured_runtime"}
                )

        result = {
            "input_id": case["input_id"],
            "status": "construction_failed",
            "physical_pass": False,
            "attempts": attempts,
            "final_validation_used_for_selection": False,
            "policy": self.policy.to_dict(),
            "task_match": None,
            "robot_task_success": None,
            "robot_real_world": "not_evaluated",
            "robot_control_status": "not_evaluated",
            "fallback_used": False,
            "teleport_used": False,
            "post_play_transform_writeback": False,
            "phase_timings": timings,
        }
        try:
            if hasattr(self.backend, "set_context"):
                self.backend.set_context(case)
            grounding = measured("grounding", self.backend.ground, safe)
            if not grounding.get("localized"):
                result["failure_reason"] = "target_not_localized"
                return result
            result["grounding"] = grounding
            candidates = measured("retrieval", self.backend.retrieve, safe, grounding, limit=self.policy.candidates)
            unique = []
            for c in candidates:
                if c["candidate_id"] not in seen:
                    unique.append(c)
                    seen.add(c["candidate_id"])
                if len(unique) == self.policy.candidates:
                    break
            result["candidate_pool_size"] = len(unique)
            result["ordered_candidate_pool"] = [
                {
                    "candidate_id": c["candidate_id"],
                    "retrieval_score": c.get("retrieval_score"),
                    "category": c.get("category"),
                }
                for c in unique
            ]
            for offset in range(0, len(unique), self.policy.batch_size):
                batch = unique[offset : offset + self.policy.batch_size]
                if self.policy.variant == "without_multimodal_selection":
                    # Retrieval order stands in for selection: every candidate is eligible, no model call.
                    assessments = [
                        {
                            "candidate_id": c["candidate_id"],
                            "category_compatible": True,
                            "mechanism_compatible": True,
                            "subtype_match": "unknown",
                            "confidence": 1.0,
                            "geometry_similarity": 1.0,
                            "evidence": "selector_disabled_retrieval_order",
                        }
                        for c in batch
                    ]
                    timings.append(
                        {"phase": "multimodal_ranking", "runtime_seconds": 0.0, "timing_status": "skipped_by_variant"}
                    )
                else:
                    assessments = measured("multimodal_ranking", self.backend.rank, safe, grounding, batch)
                assessments, normalization = normalize_assessments(assessments, batch)
                if normalization["normalized"]:
                    result.setdefault("selector_normalizations", []).append({"batch_offset": offset, **normalization})
                by_id = {c["candidate_id"]: c for c in batch}
                for assessment in assessments:
                    t = time.perf_counter()
                    entry = {"candidate_id": assessment["candidate_id"], "selection": assessment, "repairs": 0}
                    attempts.append(entry)
                    if not semantic_eligible(assessment, self.policy):
                        entry["status"] = "semantic_rejection"
                        entry["seconds"] = time.perf_counter() - t
                        continue
                    asset = measured(
                        "asset_construction", self.backend.build, safe, grounding, by_id[assessment["candidate_id"]]
                    )
                    entry["asset"] = asset
                    if asset.get("status") == "infrastructure_blocked":
                        raise InfrastructureBlocked(asset.get("reason", "build_infrastructure_blocked"))
                    if not asset.get("exported") and self.policy.repairs_per_candidate:
                        repaired = measured(
                            "repair",
                            self.backend.repair,
                            asset,
                            grounding,
                            {"failure_reasons": [asset.get("reason", "not_exported")]},
                        )
                        if repaired is not None:
                            entry["repairs"] = 1
                            entry["repair"] = repaired
                            asset = repaired
                    if asset.get("status") == "infrastructure_blocked":
                        raise InfrastructureBlocked(asset.get("reason", "repair_infrastructure_blocked"))
                    if not asset.get("exported"):
                        entry["status"] = "not_exported"
                        entry["seconds"] = time.perf_counter() - t
                        continue
                    check = measured("construction_physics", self.backend.screen, asset, grounding)
                    entry["construction_check"] = check
                    if type(check.get("physical_pass")) is not bool:
                        raise InfrastructureBlocked("construction_physics_result_incomplete")
                    if not check["physical_pass"] and self.policy.repairs_per_candidate and entry["repairs"] == 0:
                        repaired = measured("repair", self.backend.repair, asset, grounding, check)
                        if repaired is not None:
                            entry["repairs"] = 1
                            entry["repair"] = repaired
                            asset = repaired
                            if asset.get("status") == "infrastructure_blocked":
                                raise InfrastructureBlocked(asset.get("reason", "repair_infrastructure_blocked"))
                            if not asset.get("exported"):
                                entry["status"] = "repair_not_exported"
                                entry["seconds"] = time.perf_counter() - t
                                continue
                            check = measured("construction_physics", self.backend.screen, asset, grounding)
                            entry["repaired_construction_check"] = check
                            if type(check.get("physical_pass")) is not bool:
                                raise InfrastructureBlocked("repaired_physics_result_incomplete")
                    entry["seconds"] = time.perf_counter() - t
                    if not check["physical_pass"]:
                        entry["status"] = "construction_physics_failed"
                        if self.policy.variant == "without_physics_reselection":
                            # The construction screen may reject but never redirects to another candidate.
                            result["failure_reason"] = "first_constructed_candidate_failed_physics"
                            return result
                        continue
                    entry["status"] = "selected"
                    result["selected_asset"] = asset
                    # No loop back after this call. Private final failures count as failures.
                    final = measured("final_physics", self.backend.validate_final, asset, grounding)
                    if type(final.get("physical_pass")) is not bool:
                        raise InfrastructureBlocked("final_physics_result_incomplete")
                    result.update(
                        status="physical_pass" if final["physical_pass"] else "final_validation_failed",
                        physical_pass=bool(final["physical_pass"]),
                        final_validation=final,
                    )
                    return result
            result["failure_reason"] = (
                "candidate_budget_exhausted"
                if len(unique) == self.policy.candidates
                else "no_acceptable_available_candidate"
            )
            return result
        except MethodFailure as exc:
            result.update(status="construction_failed", physical_pass=False, failure_reason=str(exc))
            return result
        except InfrastructureBlocked as exc:
            result.update(status="infrastructure_blocked", physical_pass=None, failure_reason=str(exc))
            return result
        finally:
            result["total_wall_seconds"] = time.perf_counter() - start
            result["considered_candidates"] = len(attempts)
