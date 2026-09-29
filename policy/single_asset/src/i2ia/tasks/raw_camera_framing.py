"""Read-only live camera/mesh framing capture; never synthesized video."""
import itertools
import numpy as np


def project_check(points, intrinsics, world_to_camera, resolution, clipping):
    points=np.asarray(points,dtype=float);k=np.asarray(intrinsics,dtype=float);view=np.asarray(world_to_camera,dtype=float)
    if points.ndim!=2 or points.shape[1]!=3 or not len(points) or k.shape!=(3,3) or view.shape!=(4,4):
        return {'passed':False,'reason':'invalid_projection_input'}
    if not all(np.isfinite(x).all() for x in (points,k,view)):
        return {'passed':False,'reason':'nonfinite_projection'}
    camera=np.c_[points,np.ones(len(points))]@view.T;depth=camera[:,2]
    if (depth<clipping[0]).any() or (depth>clipping[1]).any():return {'passed':False,'reason':'near_far_clipping'}
    xy=camera[:,:3]@k.T;xy=xy[:,:2]/xy[:,2,None]
    lo=xy.min(0);hi=xy.max(0);width,height=resolution
    passed=bool((lo>=0).all() and hi[0]<width and hi[1]<height)
    return {'passed':passed,'reason':None if passed else 'frame_edge_clipping','pixel_bounds':[lo.tolist(),hi.tolist()],
            'depth_range_m':[float(depth.min()),float(depth.max())]}


class LiveFramingCapture:
    def __init__(self, stage, robot_path, object_roots, cameras, Usd, UsdGeom, Gf, omni):
        self.stage=stage;self.Gf=Gf;self.omni=omni;self.cameras=cameras;self.meshes=[];self.frames=[]
        roots=[('robot',stage.GetPrimAtPath(robot_path))]+[('object',p) for p in object_roots]
        for role,root in roots:
            for prim in Usd.PrimRange(root,Usd.TraverseInstanceProxies()):
                if not prim.IsA(UsdGeom.Mesh):continue
                if role=='object' and not prim.GetName().startswith('visual_'):continue
                points=np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get() or [],dtype=float)
                if points.ndim!=2 or len(points)<3:continue
                corners=np.asarray(list(itertools.product(*zip(points.min(0),points.max(0)))))
                self.meshes.append((role,prim,corners))
        if {r for r,_,_ in self.meshes}!={'robot','object'}:raise ValueError('framing_geometry_missing')

    def capture(self, index, simulation_time):
        groups={'robot':[],'object':[]}
        for role,prim,corners in self.meshes:
            matrix=self.omni.usd.get_world_transform_matrix(prim)
            groups[role].extend(list(matrix.Transform(self.Gf.Vec3d(*p.tolist()))) for p in corners)
        entries=[]
        for name,camera in self.cameras:
            k=np.asarray(camera.get_intrinsics_matrix(),dtype=float).tolist()
            view=np.asarray(camera.get_view_matrix_ros(),dtype=float).tolist()
            resolution=list(camera.get_resolution());clipping=list(camera.get_clipping_range())
            entries.append({'name':name,'intrinsics':k,'world_to_camera':view,'resolution':resolution,'clipping_m':clipping,
                            'groups':{role:project_check(points,k,view,resolution,clipping) for role,points in groups.items()}})
        self.frames.append({'frame':index,'simulation_time_s':simulation_time,'world_mesh_box_corners':groups,'cameras':entries})

    def report(self):
        return {'schema':'live_isaac_camera_geometry_framing_v1','camera_capture_type':'raw_isaac_rgb','derived_animation_used':False,
                'scope':'frustum containment only; no task success or occlusion claim','mesh_paths':[{'role':r,'path':str(p.GetPath())} for r,p,_ in self.meshes],
                'frames':self.frames}


def verify(report, synchronized):
    errors=[];frames=report.get('frames',[]);sync=synchronized.get('frames',[])
    if len(frames)<2 or len(frames)!=len(sync):errors.append('frame_count_mismatch')
    for a,b in zip(frames,sync):
        if a['frame']!=b['frame'] or abs(a['simulation_time_s']-b['simulation_time_s'])>1e-6:errors.append('framing_camera_clock_mismatch')
        if {c['name'] for c in a['cameras']}!={'policy','evidence'}:errors.append('two_camera_evidence_required')
        for camera in a['cameras']:
            for role in ('robot','object'):
                r=project_check(a['world_mesh_box_corners'][role],camera['intrinsics'],camera['world_to_camera'],camera['resolution'],camera['clipping_m'])
                if not r['passed']:errors.append(camera['name']+':'+role+':'+r['reason'])
    if report.get('camera_capture_type')!='raw_isaac_rgb' or report.get('derived_animation_used') is not False:errors.append('raw_camera_scope_missing')
    return {'passed':not errors,'failure_reasons':sorted(set(errors)),'frames_checked':len(frames),'scope':'frustum containment, not task or occlusion success'}
