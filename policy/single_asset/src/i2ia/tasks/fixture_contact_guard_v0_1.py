"""Physical fixture contacts are not robot manipulation evidence."""
import math


def forbidden_fixture_contact(events, spec):
    fixtures=set(spec.get("forbidden_object_fixture_paths", []))
    objects=set(spec.get("object_body_paths", [spec["target_link"]]))
    reasons=[]
    for event in events:
        pair=set(event["paths"][:2])
        if not (pair & fixtures and pair & objects):
            continue
        if any(c["separation"]<=spec["maximum_contact_separation_m"] and
               math.sqrt(sum(float(x)**2 for x in c["impulse"]))>=spec["minimum_contact_impulse_ns"]
               for c in event["samples"]):
            reasons.append({"pair":sorted(pair),"reason":"object_contact_with_robot_mount"})
    return reasons
