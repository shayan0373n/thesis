import pymunk
import pymunk.pygame_util
import pygame
import random

ORIGIN = (300, 300)
SHANK_LENGTH = 100
THIGH_LENGTH = 100
TORSO_LENGTH = 100

# Create a space
space = pymunk.Space()
space.gravity = (0, 980)

ground_body = pymunk.Body(body_type=pymunk.Body.STATIC)

# Create the shank body
shank_body = pymunk.Body()
shank_body.position = ORIGIN
shank_shape = pymunk.Segment(shank_body, (0, 0), (0, -SHANK_LENGTH), 5)
shank_shape.mass = 1
shank_shape.color = (0, 0, 0, 255)
space.add(shank_body, shank_shape)

# Create the thigh body
thigh_body = pymunk.Body()
thigh_body.position = (ORIGIN[0], ORIGIN[1] - SHANK_LENGTH)
thigh_shape = pymunk.Segment(thigh_body, (0, 0), (-THIGH_LENGTH, 0), 5)
thigh_shape.mass = 1
thigh_shape.color = (0, 0, 0, 255)
space.add(thigh_body, thigh_shape)

# Create the torso body
torso_body = pymunk.Body()
torso_body.position = (ORIGIN[0] - THIGH_LENGTH, ORIGIN[1] - SHANK_LENGTH)
torso_shape = pymunk.Segment(torso_body, (0, 0), (0, -TORSO_LENGTH), 5)
torso_shape.mass = 1
torso_shape.color = (0, 0, 0, 255)
space.add(torso_body, torso_shape)

# Group the shapes together
shape_filter = pymunk.ShapeFilter(group=1)
shank_shape.filter = shape_filter
thigh_shape.filter = shape_filter
torso_shape.filter = shape_filter

# Create pivot joints
def random_torque_func(spring, angle):
    t = 10000 * random.random()
    return t
ground_shank_joint = pymunk.PivotJoint(ground_body, shank_body, ORIGIN)
ground_shank_joint_motor = pymunk.DampedRotarySpring(ground_body, shank_body, 0, 1000000, 0)
ground_shank_joint_motor.torque_func = random_torque_func
shank_thigh_joint = pymunk.PivotJoint(shank_body, thigh_body, (0, -SHANK_LENGTH), (0, 0))
# shank_thigh_joint_motor = pymunk.DampedRotarySpring(shank_body, thigh_body, 0, 10, 0)
# shank_thigh_joint_motor.torque_func = random_torque_func
thigh_torso_joint = pymunk.PivotJoint(thigh_body, torso_body, (-THIGH_LENGTH, 0), (0, 0))
space.add(ground_shank_joint, ground_shank_joint_motor, shank_thigh_joint, thigh_torso_joint)


print(f"Shank Moment: {shank_body.moment}")
print(f"Thigh Moment: {thigh_body.moment}")
# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((600, 600))
clock = pygame.time.Clock()
draw_options = pymunk.pygame_util.DrawOptions(screen)

# Simulation loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill((255, 255, 255))  # Fill the screen with white
    space.debug_draw(draw_options)  # Draw the pendulum
    space.step(0.01)  # Step the simulation
    pygame.display.flip()  # Update the display
    clock.tick(60)  # Limit the frame rate to 60 FPS

pygame.quit()