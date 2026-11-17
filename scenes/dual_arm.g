# Two sequential robots beside a conveyor. Metres; world +z up, belt +x.
# Python adds the two bundled Panda models and animates the belt separately.
world: {}

floor(world): {
  Q: [0 0 -.05], shape: ssBox, size: [6 4 .1 .01],
  color: [.66 .68 .70], contact: 1
}

conveyor(world): {
  Q: [0 0 .71], shape: ssBox, size: [2.4 .4 .08 .01],
  color: [.10 .14 .17], contact: 1, friction: .5
}
conveyor_leg_1(world): {
  Q: [-1.0 -.13 .335], shape: ssBox, size: [.08 .08 .67 .005],
  color: [.52 .57 .61], contact: 1
}
conveyor_leg_2(world): {
  Q: [1.0 -.13 .335], shape: ssBox, size: [.08 .08 .67 .005],
  color: [.52 .57 .61], contact: 1
}
conveyor_leg_3(world): {
  Q: [-1.0 .13 .335], shape: ssBox, size: [.08 .08 .67 .005],
  color: [.52 .57 .61], contact: 1
}
conveyor_leg_4(world): {
  Q: [1.0 .13 .335], shape: ssBox, size: [.08 .08 .67 .005],
  color: [.52 .57 .61], contact: 1
}

# Fixed end rollers are visual geometry; the deck provides collision geometry.
conveyor_roller_in(world): {
  Q: "t(-1.2 0 .71) d(90 1 0 0)", shape: cylinder, size: [.42 .04],
  color: [.62 .67 .70], contact: 0
}
conveyor_roller_out(world): {
  Q: "t(1.2 0 .71) d(90 1 0 0)", shape: cylinder, size: [.42 .04],
  color: [.62 .67 .70], contact: 0
}

robot_a_mount(world): { Q: "t(-.55 -.60 .75) d(90 0 0 1)" }
robot_a_pedestal(robot_a_mount): {
  Q: [0 0 -.375], shape: ssBox, size: [.28 .28 .75 .01],
  color: [.20 .27 .32], contact: 1
}
robot_b_mount(world): { Q: "t(.55 -.60 .75) d(90 0 0 1)" }
robot_b_pedestal(robot_b_mount): {
  Q: [0 0 -.375], shape: ssBox, size: [.28 .28 .75 .01],
  color: [.20 .27 .32], contact: 1
}

# Decorative hardware only. Collision geometry stays deliberately simple.
# Side members sit below the working surface and leave both exits open.
visual_side_frame_near(world): {
  Q: [0 -.225 .635], shape: ssBox, size: [2.30 .035 .07 .005],
  color: [.64 .69 .72], contact: 0
}
visual_side_frame_far(world): {
  Q: [0 .225 .635], shape: ssBox, size: [2.30 .035 .07 .005],
  color: [.64 .69 .72], contact: 0
}
visual_crossmember_in(world): {
  Q: [-1 0 .205], shape: ssBox, size: [.055 .32 .055 .004],
  color: [.49 .55 .59], contact: 0
}
visual_crossmember_out(world): {
  Q: [1 0 .205], shape: ssBox, size: [.055 .32 .055 .004],
  color: [.49 .55 .59], contact: 0
}
visual_lower_tie_near(world): {
  Q: [0 -.13 .205], shape: ssBox, size: [2 .045 .045 .004],
  color: [.49 .55 .59], contact: 0
}
visual_lower_tie_far(world): {
  Q: [0 .13 .205], shape: ssBox, size: [2 .045 .045 .004],
  color: [.49 .55 .59], contact: 0
}

visual_foot_in_near(world): {
  Q: [-1 -.13 .01], shape: ssBox, size: [.15 .15 .02 .004],
  color: [.26 .31 .35], contact: 0
}
visual_foot_in_far(world): {
  Q: [-1 .13 .01], shape: ssBox, size: [.15 .15 .02 .004],
  color: [.26 .31 .35], contact: 0
}
visual_foot_out_near(world): {
  Q: [1 -.13 .01], shape: ssBox, size: [.15 .15 .02 .004],
  color: [.26 .31 .35], contact: 0
}
visual_foot_out_far(world): {
  Q: [1 .13 .01], shape: ssBox, size: [.15 .15 .02 .004],
  color: [.26 .31 .35], contact: 0
}

visual_bearing_in_near(world): {
  Q: "t(-1.2 -.245 .71) d(90 1 0 0)", shape: cylinder, size: [.035 .024],
  color: [.28 .34 .39], contact: 0
}
visual_bearing_in_far(world): {
  Q: "t(-1.2 .245 .71) d(90 1 0 0)", shape: cylinder, size: [.035 .024],
  color: [.28 .34 .39], contact: 0
}
visual_bearing_out_near(world): {
  Q: "t(1.2 -.245 .71) d(90 1 0 0)", shape: cylinder, size: [.035 .024],
  color: [.28 .34 .39], contact: 0
}
visual_bearing_out_far(world): {
  Q: "t(1.2 .245 .71) d(90 1 0 0)", shape: cylinder, size: [.035 .024],
  color: [.28 .34 .39], contact: 0
}
visual_drive_housing(world): {
  Q: [-1.12 .315 .615], shape: ssBox, size: [.20 .13 .11 .015],
  color: [.20 .29 .35], contact: 0
}
visual_drive_endcap(world): {
  Q: "t(-1.12 .392 .615) d(90 1 0 0)", shape: cylinder, size: [.025 .039],
  color: [.49 .55 .59], contact: 0
}

visual_robot_a_base_plate(robot_a_mount): {
  Q: [0 0 -.735], shape: ssBox, size: [.38 .38 .03 .006],
  color: [.49 .55 .59], contact: 0
}
visual_robot_b_base_plate(robot_b_mount): {
  Q: [0 0 -.735], shape: ssBox, size: [.38 .38 .03 .006],
  color: [.49 .55 .59], contact: 0
}
visual_robot_a_mount_plate(robot_a_mount): {
  Q: [0 0 -.006], shape: ssBox, size: [.31 .31 .012 .003],
  color: [.64 .69 .72], contact: 0
}
visual_robot_b_mount_plate(robot_b_mount): {
  Q: [0 0 -.006], shape: ssBox, size: [.31 .31 .012 .003],
  color: [.64 .69 .72], contact: 0
}

# Preliminary destination bins: reference origins are inner floor centres.
# Each open bin has a .30 x .24 m interior, .14 m walls and .012 m wall thickness.
# Placement/approach frames are planning references, not verified gripper goals.
target_bin_a(world): {
  Q: [-1.0 -.85 .75], inner_size_m: [.30 .24 .14],
  preferred_robot_id: "robot_a"
}
target_bin_a_bottom(target_bin_a): {
  Q: [0 0 -.0075], shape: ssBox, size: [.324 .264 .015 .003],
  color: [.28 .49 .62], contact: 1, friction: .45
}
target_bin_a_wall_left(target_bin_a): {
  Q: [-.156 0 .07], shape: ssBox, size: [.012 .264 .14 .003],
  color: [.15 .42 .59], contact: 1, friction: .45
}
target_bin_a_wall_right(target_bin_a): {
  Q: [.156 0 .07], shape: ssBox, size: [.012 .264 .14 .003],
  color: [.15 .42 .59], contact: 1, friction: .45
}
target_bin_a_wall_near(target_bin_a): {
  Q: [0 -.126 .07], shape: ssBox, size: [.30 .012 .14 .003],
  color: [.15 .42 .59], contact: 1, friction: .45
}
target_bin_a_wall_far(target_bin_a): {
  Q: [0 .126 .07], shape: ssBox, size: [.30 .012 .14 .003],
  color: [.15 .42 .59], contact: 1, friction: .45
}
target_bin_a_leg_left_near(target_bin_a): {
  Q: [-.13 -.10 -.3825], shape: ssBox, size: [.035 .035 .735 .004],
  color: [.49 .55 .59], contact: 1
}
target_bin_a_leg_left_far(target_bin_a): {
  Q: [-.13 .10 -.3825], shape: ssBox, size: [.035 .035 .735 .004],
  color: [.49 .55 .59], contact: 1
}
target_bin_a_leg_right_near(target_bin_a): {
  Q: [.13 -.10 -.3825], shape: ssBox, size: [.035 .035 .735 .004],
  color: [.49 .55 .59], contact: 1
}
target_bin_a_leg_right_far(target_bin_a): {
  Q: [.13 .10 -.3825], shape: ssBox, size: [.035 .035 .735 .004],
  color: [.49 .55 .59], contact: 1
}
target_bin_a_place(target_bin_a): { Q: [0 0 .10] }
target_bin_a_approach(target_bin_a): { Q: [0 0 .30] }

target_bin_b(world): {
  Q: [1.0 -.85 .75], inner_size_m: [.30 .24 .14],
  preferred_robot_id: "robot_b"
}
target_bin_b_bottom(target_bin_b): {
  Q: [0 0 -.0075], shape: ssBox, size: [.324 .264 .015 .003],
  color: [.78 .57 .25], contact: 1, friction: .45
}
target_bin_b_wall_left(target_bin_b): {
  Q: [-.156 0 .07], shape: ssBox, size: [.012 .264 .14 .003],
  color: [.72 .43 .14], contact: 1, friction: .45
}
target_bin_b_wall_right(target_bin_b): {
  Q: [.156 0 .07], shape: ssBox, size: [.012 .264 .14 .003],
  color: [.72 .43 .14], contact: 1, friction: .45
}
target_bin_b_wall_near(target_bin_b): {
  Q: [0 -.126 .07], shape: ssBox, size: [.30 .012 .14 .003],
  color: [.72 .43 .14], contact: 1, friction: .45
}
target_bin_b_wall_far(target_bin_b): {
  Q: [0 .126 .07], shape: ssBox, size: [.30 .012 .14 .003],
  color: [.72 .43 .14], contact: 1, friction: .45
}
target_bin_b_leg_left_near(target_bin_b): {
  Q: [-.13 -.10 -.3825], shape: ssBox, size: [.035 .035 .735 .004],
  color: [.49 .55 .59], contact: 1
}
target_bin_b_leg_left_far(target_bin_b): {
  Q: [-.13 .10 -.3825], shape: ssBox, size: [.035 .035 .735 .004],
  color: [.49 .55 .59], contact: 1
}
target_bin_b_leg_right_near(target_bin_b): {
  Q: [.13 -.10 -.3825], shape: ssBox, size: [.035 .035 .735 .004],
  color: [.49 .55 .59], contact: 1
}
target_bin_b_leg_right_far(target_bin_b): {
  Q: [.13 .10 -.3825], shape: ssBox, size: [.035 .035 .735 .004],
  color: [.49 .55 .59], contact: 1
}
target_bin_b_place(target_bin_b): { Q: [0 0 .10] }
target_bin_b_approach(target_bin_b): { Q: [0 0 .30] }

# Independent root allows belt motion to stop controlling a later grasped box.
object_001: {
  X: [-1.05 0 .79], shape: ssBox, size: [.06 .045 .08 .005],
  color: [.90 .32 .12], contact: 1, mass: .12, friction: .5
}
