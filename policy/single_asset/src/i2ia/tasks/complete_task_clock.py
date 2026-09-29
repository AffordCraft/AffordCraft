"""Exactly one PhysX step per state sample; render never advances physics."""
import math


def advance_one_sample(world, physics_step, render, expected_dt):
    before = float(world.current_time)
    physics_step(render=False, update_fabric=True)
    after = float(world.current_time)
    if not math.isfinite(after) or abs(after - before - expected_dt) > 1e-6:
        raise RuntimeError("physics_sample_dt_contract_failed")
    if render:
        world.render()
        if abs(float(world.current_time) - after) > 1e-7:
            raise RuntimeError("render_advanced_physics_time")
    return after
