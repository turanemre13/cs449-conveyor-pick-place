# Step 1: stationary geometry only. Units: metres, world +z up, future belt +x.
world: {}

floor(world): {
  Q: [0 0 -.05], shape: ssBox, size: [6 4 .1 .01],
  color: [.65 .67 .70], contact: 1
}

conveyor(world): {
  Q: [0 0 .71], shape: ssBox, size: [1.6 .4 .08 .01],
  color: [.15 .22 .28], contact: 1, friction: .5
}
conveyor_leg_1(world): {
  Q: [-.65 -.13 .335], shape: ssBox, size: [.08 .08 .67 .005],
  color: [.38 .40 .43], contact: 1
}
conveyor_leg_2(world): {
  Q: [.65 -.13 .335], shape: ssBox, size: [.08 .08 .67 .005],
  color: [.38 .40 .43], contact: 1
}
conveyor_leg_3(world): {
  Q: [-.65 .13 .335], shape: ssBox, size: [.08 .08 .67 .005],
  color: [.38 .40 .43], contact: 1
}
conveyor_leg_4(world): {
  Q: [.65 .13 .335], shape: ssBox, size: [.08 .08 .67 .005],
  color: [.38 .40 .43], contact: 1
}

# The model's +x direction faces the belt after a +90 degree world-z rotation.
robot_a_mount(world): { Q: "t(0 -.60 .75) d(90 0 0 1)" }
robot_a_pedestal(robot_a_mount): {
  Q: [0 0 -.375], shape: ssBox, size: [.28 .28 .75 .01],
  color: [.42 .44 .47], contact: 1
}

# Independent root so a later motion/dynamics stage can move it off the belt.
# The bottom is at .75 m, exactly on the conveyor's top surface.
object_001: {
  X: [-.25 0 .79], shape: ssBox, size: [.06 .045 .08 .005],
  color: [.90 .32 .12], contact: 1, mass: .12, friction: .5
}
