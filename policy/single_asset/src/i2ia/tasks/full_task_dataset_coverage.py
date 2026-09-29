"""Dataset-level fail-closed coverage after independent episode admission.

These checks do not replace episode state-machine, physics, or raw media
verification. They prevent incomplete/duplicated/adaptive cohorts being silently
passed to training once those per-episode verifications have been performed.
"""
from collections import Counter


def coverage(assets, episodes, heldout_plan, required_successes=100):
    errors=[];ids=[a['asset_id'] for a in assets]
    if len(ids)!=10 or len(set(ids))!=10:errors.append('ten_unique_assets_required')
    if Counter(a['asset_role'] for a in assets)!=Counter({'rigid':5,'articulated':5}):errors.append('five_rigid_five_articulated_required')
    if required_successes<100:errors.append('training_target_below_user_requirement')
    planned={(p['asset_id'],p['seed']) for p in heldout_plan}
    heldout_states={(p['asset_id'],p['initial_state_camera_fingerprint']) for p in heldout_plan}
    heldout_counts=Counter(aid for aid,_ in planned)
    if any(heldout_counts[aid]<20 for aid in ids):errors.append('twenty_paired_heldout_episodes_per_asset_required')
    if len(planned)!=len(heldout_plan) or len(heldout_states)!=len(heldout_plan):errors.append('duplicate_heldout_seed_or_initial_state')
    counts=Counter();seeds=set();fingerprints=set()
    for e in episodes:
        aid=e.get('asset_id');key=(aid,e.get('seed'));fp=e.get('physical_trajectory_fingerprint')
        reasons=[]
        if aid not in ids:reasons.append('unregistered_asset')
        if e.get('split')!='train':reasons.append('non_train_episode')
        if e.get('independent_admission_passed') is not True or e.get('sim_task_success') is not True:reasons.append('unsuccessful_or_unverified_episode')
        if e.get('model_variant')!='demonstration_teacher':reasons.append('not_demonstration_teacher')
        if key in planned or (aid,e.get('initial_state_camera_fingerprint')) in heldout_states:reasons.append('heldout_leak')
        if key in seeds:reasons.append('duplicate_asset_seed')
        if not isinstance(fp,str) or len(fp)!=64:reasons.append('trajectory_fingerprint_missing')
        elif fp in fingerprints:reasons.append('duplicate_physical_trajectory')
        if not isinstance(e.get('initial_state_camera_fingerprint'),str):reasons.append('initial_state_camera_fingerprint_missing')
        seeds.add(key)
        if isinstance(fp,str):fingerprints.add(fp)
        if reasons:errors.extend(reasons)
        else:counts[aid]+=1
    deficits={aid:max(0,required_successes-counts[aid]) for aid in ids}
    if any(deficits.values()):errors.append('insufficient_success_trajectories')
    return {'training_allowed':not errors,'errors':sorted(set(errors)),
            'per_asset_accepted_count':{aid:counts[aid] for aid in ids},'per_asset_deficit':deficits,
            'required_training_trajectories':len(ids)*required_successes,
            'claim_scope':'coverage after independent episode admission; not training or VLA success'}
