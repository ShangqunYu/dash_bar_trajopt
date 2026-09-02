"""Export the 6-spoke wheel (and its post) as a STEP assembly.

Dimensions in mm, matching get_bar_spec(num_spokes=6) in
src/dash_mjlab/tasks/bar_angle/env_cfgs.py exactly:

  wheel plane height   690
  spoke: capsule, r=20, centreline length 200 (hemispherical cap -> 220 to
         the cap's apex), 6 spokes every 60 deg
  tip marker balls     r=26 at each spoke end
  post:  r=20 cylinder from floor to 655 (stops 35 below the wheel plane)
  axle:  r=8 from 655 to 700
  foot:  r=70 disc, 12 tall
"""

import cadquery as cq
from math import cos, sin, pi

BAR_HEIGHT = 690.0
BAR_LENGTH = 200.0
BAR_RADIUS = 20.0
POST_RADIUS = 20.0
AXLE_RADIUS = 8.0
TIP_BALL_RADIUS = 26.0
NUM_SPOKES = 6

# Wheel: built in its own frame with the hinge axis = Z and the wheel plane at
# z=0; placed at BAR_HEIGHT by the assembly. Spoke 0 along -X (the red one).
wheel = None
for i in range(NUM_SPOKES):
    theta = 2.0 * pi * i / NUM_SPOKES
    tip = (-BAR_LENGTH * cos(theta), -BAR_LENGTH * sin(theta), 0.0)
    spoke = (
        cq.Workplane("XY")
        .transformed(rotate=(0, 0, 180.0 + theta * 180.0 / pi))
        .transformed(rotate=(0, 90, 0))
        .circle(BAR_RADIUS)
        .extrude(BAR_LENGTH)
    )
    cap = cq.Workplane("XY").transformed(offset=tip).sphere(BAR_RADIUS)
    ball = cq.Workplane("XY").transformed(offset=tip).sphere(TIP_BALL_RADIUS)
    hub_cap = cq.Workplane("XY").sphere(BAR_RADIUS)
    piece = spoke.union(cap).union(ball).union(hub_cap)
    wheel = piece if wheel is None else wheel.union(piece)

# Post: foot disc + shaft + axle stub, floor at z=0.
post = (
    cq.Workplane("XY")
    .circle(70.0)
    .extrude(12.0)
    .union(cq.Workplane("XY").circle(POST_RADIUS).extrude(BAR_HEIGHT - 35.0))
    .union(
        cq.Workplane("XY")
        .transformed(offset=(0, 0, BAR_HEIGHT - 35.0))
        .circle(AXLE_RADIUS)
        .extrude(45.0)
    )
)

asm = cq.Assembly(name="dash_bar_wheel_6_spoke")
asm.add(post, name="post", color=cq.Color(0.3, 0.3, 0.32))
asm.add(
    wheel,
    name="wheel_6_spoke",
    loc=cq.Location((0, 0, BAR_HEIGHT)),
    color=cq.Color(0.82, 0.16, 0.12),
)

import sys

out = sys.argv[1]
asm.save(out, exportType="STEP")
print("wrote", out)

# Sanity: re-import and report solids + bounding box.
imported = cq.importers.importStep(out)
solids = imported.solids().vals()
bb = imported.val().BoundingBox()
print(f"solids: {len(solids)}")
print(f"bbox x [{bb.xmin:.1f}, {bb.xmax:.1f}] y [{bb.ymin:.1f}, {bb.ymax:.1f}] z [{bb.zmin:.1f}, {bb.zmax:.1f}] mm")
