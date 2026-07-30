#!/usr/bin/env python3
"""
carla_calib_check.py -- objective IPM calibration check. Runs on the CARLA
box only.

tools/ipm_overlay.py asks a human "does that line look like 5 m?".  This
asks CARLA instead: it projects known ground points with the simulator's own
camera transform and compares against what homography_from_extrinsics
predicts for the same points, in pixels.  It also sweeps --cam-height to
report the best-fit value, which is how the 1.6-vs-1.939 mount-offset error
documented in DEPLOY.md section 3 was found.

Usage:
    PYTHONPATH=. python3 tools/carla_calib_check.py
"""

import os
import sys

import carla
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from perception_costmap.occupancy import GridSpec
from perception_costmap.bev import homography_from_extrinsics
c=carla.Client('127.0.0.1',2000); c.set_timeout(30.0)
w=c.get_world(); bl=w.get_blueprint_library()
sp=w.get_map().get_spawn_points()[0]
ego=w.spawn_actor(bl.filter('vehicle.*')[0], sp); actors=[ego]
try:
    bp=bl.find('sensor.camera.rgb')
    for k,v in (('image_size_x','640'),('image_size_y','480'),('fov','90')): bp.set_attribute(k,v)
    cam=w.spawn_actor(bp, carla.Transform(carla.Location(x=1.5,z=1.6)), attach_to=ego); actors.append(cam)
    for _ in range(60): w.wait_for_tick()
    K=np.array([[320.,0,320],[0,320.,240],[0,0,1]])
    g=GridSpec(x_min=-4,x_max=16,y_min=-10,y_max=10,resolution=0.1)
    A=np.array([[1/g.resolution,0,-g.x_min/g.resolution],[0,1/g.resolution,-g.y_min/g.resolution],[0,0,1.]])
    tf=ego.get_transform(); ct=cam.get_transform()
    road_z=w.get_map().get_waypoint(tf.location,project_to_road=True).transform.location.z
    w2c=np.array(ct.get_inverse_matrix()); fwd=tf.get_forward_vector(); right=tf.get_right_vector()
    probes=[(5,0),(8,0),(12,0),(8,-2),(8,2),(4,1.5),(16,-3),(6,-1),(10,2.5)]
    truth={}
    for (dx,dy) in probes:
        loc=carla.Location(x=tf.location.x+fwd.x*dx-right.x*dy, y=tf.location.y+fwd.y*dx-right.y*dy, z=road_z)
        pc=w2c@np.array([loc.x,loc.y,loc.z,1.0]); pcv=np.array([pc[1],-pc[2],pc[0]])
        px=K@pcv; px/=px[2]; truth[(dx,dy)]=(px[0],px[1])
    def maxerr(h):
        Hi=np.linalg.inv(homography_from_extrinsics(K,(1.5,0.0,h),0.0,0.0,g)); e=0
        for k,(tu,tv) in truth.items():
            gp=A@np.array([k[0],k[1],1.0]); q=Hi@np.array([gp[0],gp[1],1.0]); q/=q[2]
            e=max(e,float(np.hypot(tu-q[0],tv-q[1])))
        return e
    hs=np.arange(1.60,2.20,0.001); errs=[maxerr(h) for h in hs]; hbest=hs[int(np.argmin(errs))]
    print('ego origin above road : %.3f m' % (tf.location.z-road_z))
    print('camera world z - road : %.3f m   (= 1.6 mount + origin offset)' % (ct.location.z-road_z))
    print('vehicle pitch         : %.3f deg' % tf.rotation.pitch)
    print()
    print('max reprojection error vs CARLA ground truth:')
    for h,label in ((1.600,'naive mount offset (what DEPLOY.md/carla_feed imply)'),
                    (ct.location.z-road_z,'geometric cam-z above road'),
                    (hbest,'BEST FIT')):
        print('  cam-height %.3f  -> %6.2f px   %s' % (h, maxerr(h), label))
finally:
    for a in reversed(actors):
        try: a.destroy()
        except Exception: pass
