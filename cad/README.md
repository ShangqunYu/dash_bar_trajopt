# CAD reference

`wheel_6_spoke.step` is the 6-spoke ship's wheel and its post exactly as
simulated (`get_bar_spec(num_spokes=6)` in
`src/dash_mjlab/tasks/bar_angle/env_cfgs.py`), exported as a STEP assembly of
two solids for use as a design reference in Onshape or any other CAD tool.
Units are millimetres; the floor is z = 0.

Key dimensions:

| feature | value |
| --- | --- |
| wheel plane height above floor | 690 |
| spokes | 6, every 60 deg, in the horizontal plane |
| spoke centreline length (hinge axis to tip centre) | 200 |
| spoke radius (capsule, hemispherical tip cap) | 20 |
| tip marker ball radius | 26 |
| post shaft radius / top | 20 / 655 |
| axle stub radius, from 655 to 700 | 8 |
| foot disc radius x height | 70 x 12 |
| hinge axis | vertical (+z) through the post |
| pivot distance in front of the robot torso | 340 |

Simulation-side properties that the physical build should aim for:

- Each spoke weighs 0.3 kg in sim (a 6-spoke wheel is 1.8 kg total, moment of
  inertia dominated by the spokes as rods about the hinge).
- The hinge is vertical, so gravity exerts no torque about it; the sim adds
  0.05 Nm s/rad of viscous damping at the hinge, which is what parks the
  wheel after a push.
- In sim the wheel is rigid and the post is a passive obstacle; there is no
  rim connecting the spoke tips.

Regenerate with:

```bash
uv run --no-project --with cadquery python cad/make_wheel_step.py cad/wheel_6_spoke.step
```

If the sim geometry changes, update the constants at the top of
`make_wheel_step.py` to match `env_cfgs.py` and re-run.
