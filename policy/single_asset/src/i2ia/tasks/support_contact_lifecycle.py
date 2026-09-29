"""Read-only PhysX support evidence across sleep, never synthesizes impulses.

Sleep is an explicit simulator readback. A previous measured support contact
may persist only while no LOST notification occurred and the body pose is
unchanged. Missing sleep/readback/event types fail closed.
"""
import math


class SupportContactLifecycle:
    def __init__(self, target, support, max_separation):
        self.target=target;self.support=support;self.limit=max_separation;self.active={}

    def observe(self, events, position, quaternion, sleeping, timestamp):
        pose=list(position)+list(quaternion)
        if len(pose)!=7 or not all(math.isfinite(float(x)) for x in pose):
            raise ValueError('support_pose_invalid')
        current=[]
        for event in events:
            paths=event['paths']
            if set(paths[:2])!={self.target,self.support}:continue
            key=tuple(sorted(zip(paths[:2],paths[2:4])))
            kind=event.get('event_type')
            if kind not in ('CONTACT_FOUND','CONTACT_PERSIST','CONTACT_PERSISTS','CONTACT_LOST'):
                self.active.pop(key,None)
                raise ValueError('support_contact_event_type_missing')
            if kind=='CONTACT_LOST':
                self.active.pop(key,None);continue
            samples=event.get('samples',[])
            valid=[x for x in samples if isinstance(x.get('separation'),(int,float))
                   and math.isfinite(x['separation']) and x['separation']<=self.limit
                   and len(x.get('impulse',[]))==3 and all(math.isfinite(v) for v in x['impulse'])]
            if valid:
                self.active[key]={'pose':pose,'timestamp_s':timestamp,'event':event}
                current.append(key)
            else:self.active.pop(key,None)
        # A changed pose invalidates retained contact, even if a faulty sleep
        # readback says True. Do not extrapolate contact during awake motion.
        for key,entry in list(self.active.items()):
            old=entry['pose'];position_error=max(abs(x-y) for x,y in zip(pose[:3],old[:3]))
            quaternion_error=min(max(abs(x-y) for x,y in zip(pose[3:],old[3:])),max(abs(x+y) for x,y in zip(pose[3:],old[3:])))
            if position_error>1e-6 or quaternion_error>1e-6:self.active.pop(key)
        active=[{'actor_collider_pair':[list(x) for x in key],**entry} for key,entry in sorted(self.active.items())]
        measured=any(key in self.active for key in current)
        retained=not measured and sleeping is True and bool(active)
        return {'support_contact':measured or retained,
                'basis':'current_physx_contact' if measured else 'sleeping_unchanged_pose_active_pair' if retained else 'no_verified_support',
                'sleeping_readback':sleeping,'active_pairs':active,'timestamp_s':timestamp,
                'object_commands_sent':False,'impulses_synthesized':False}


def independently_verify(spec, trace):
    tracker=SupportContactLifecycle(spec['target_link'],'/World/Exp1Table',spec['maximum_contact_separation_m'])
    errors=[]
    for i,state in enumerate(trace.get('states',[])):
        try:
            expected=tracker.observe(state['raw_contacts'],state['object_position_world_m'],state['object_orientation_wxyz'],
                                     state['support_lifecycle']['sleeping_readback'],state['timestamp_s'])
            if expected!=state['support_lifecycle'] or expected['support_contact'] is not state['support_contact']:
                errors.append({'index':i,'reason':'support_lifecycle_mismatch'})
        except (KeyError,ValueError,TypeError) as exc:errors.append({'index':i,'reason':str(exc)})
    return {'passed':bool(trace.get('states')) and not errors,'errors':errors,'scope':'contact evidence only, not task success'}
