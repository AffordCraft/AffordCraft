"""Auditable text/typed-task consistency, independent of episode outcomes.

Historical wording is preserved. An opposite initial verb is corrected from
the already frozen typed task, not by changing physical goals or limits.
Additional clauses (e.g. release) are never removed.
"""
import re

def instruction(spec):
    original=spec['instruction'];task=spec['task_type'];text=original;correction=None
    opposite={'close':'open','open':'close'}
    if task in opposite and re.match(r'^'+opposite[task]+r'\b',original,re.I):
        text=re.sub(r'^'+opposite[task],task.capitalize(),original,count=1,flags=re.I)
        text=re.sub(r'\b'+opposite[task]+r' range\b',('closed' if task=='close' else 'open')+' range',text,flags=re.I)
        correction='historical_instruction_opposes_frozen_typed_task; text corrected, task targets unchanged'
    return dict(historical_instruction=original,training_and_paired_evaluation_instruction=text,typed_task=task,correction_reason=correction,task_goals_changed=False,release_required=bool(re.search(r'\brelease\b',original,re.I)))

def release_evidence(spec,states):
    if not instruction(spec)['release_required']:return dict(required=False,passed=True)
    last_contact=None
    for state in states:
        for c in state['contacts']:
            if c['separation_m']<=spec['maximum_contact_separation_m'] and sum(x*x for x in c['impulse_world_ns'])**.5>=spec['minimum_contact_impulse_ns']:
                last_contact=state['timestamp_s']
    elapsed=states[-1]['timestamp_s']-last_contact if last_contact is not None else 0.
    return dict(required=True,passed=last_contact is not None and elapsed>=spec['hold_duration_s'],last_actual_robot_contact_s=last_contact,released_hold_seconds=elapsed,required_hold_seconds=spec['hold_duration_s'],scope='raw impulse/separation; no semantic clause discarded')
