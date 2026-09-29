"""Add independent PhysX lifecycle replay without changing full-task targets."""
from .full_task_training_admission import decide as legacy_decide
from .support_contact_lifecycle import independently_verify


def decide(spec,trace,physics,media,framing):
    result=legacy_decide(spec,trace,physics,media,framing)
    if spec.get('support_contact_observation_contract')=='physx_lifecycle_sleep_v0_1':
        replay=independently_verify(spec,trace)
        if not replay['passed']:
            result['failure_reasons']=sorted(set(result['failure_reasons']+['support_lifecycle_independent_replay_failed']))
            result['training_admitted']=False
    elif any('support_lifecycle' in state for state in trace.get('states',[])):
        result['failure_reasons']=sorted(set(result['failure_reasons']+['unregistered_support_lifecycle_observation']))
        result['training_admitted']=False
    return result
