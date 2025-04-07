import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.patches import Polygon

# Link lengths (in mm) – tweak these for your desired knee geometry:
L1 = 30.0  # Fixed distance between femoral attachment points (A-D)
L2 = 45.0  # ACL length (A-B)
L3 = 30.0  # Tibial plateau (coupler, B-C)
L4 = 45.0  # PCL length (C-D)

def circle_circle_intersection(p1, r1, p2, r2):
    """
    Compute the intersection points of two circles.
    p1, p2: 2D points (x, y) for the circle centers
    r1, r2: radii of the circles
    """
    p1, p2 = np.array(p1), np.array(p2)
    d = np.linalg.norm(p2 - p1)
    if d > r1 + r2 or d < abs(r1 - r2):
        return None  # No intersection
    if d == 0 and r1 == r2:
        return None  # Circles are coincident
    l = (r1**2 - r2**2 + d**2) / (2 * d)
    h = np.sqrt(max(r1**2 - l**2, 0))
    midpoint = p1 + l * (p2 - p1) / d
    offset = h * np.array([-(p2 - p1)[1], (p2 - p1)[0]]) / d
    pout1 = midpoint + offset
    pout2 = midpoint - offset
    return pout1, pout2

def four_bar_configuration(phi):
    """
    Given input angle phi (radians) for the ACL, compute the four-bar knee configuration.
    The loop is A->B->C->D->A, where:
      A = femoral attachment of ACL (fixed at [0, 0])
      D = femoral attachment of PCL (fixed at [0, L1])
      B = tibial attachment of ACL (computed from phi)
      C = tibial attachment of PCL (found via circle intersection)
    
    phi: angle of flexion (ACL angle) such that phi = 0 is full extension.
    """
    A = np.array([0, 0])
    D = np.array([0, L1])
    phi = -phi  # Negative so the tibia rotates clockwise
    B = A + L2 * np.array([np.cos(phi), np.sin(phi)])
    # Intersection of circle centered at B (radius L3) and at D (radius L4)
    C1, C2 = circle_circle_intersection(B, L3, D, L4)
    # Choose the solution with the lower y-value so the tibia lies "below" the femur.
    C = C1 if C1[1] < C2[1] else C2
    return A, B, C, D

def create_rectangle(P, Q, thickness, offset_direction):
    """
    Given an edge defined by points P and Q, return the four corner points of a rectangle.
    'thickness' is how far (in mm) the rectangle extends from the edge,
    and 'offset_direction' (a 2D unit vector) indicates the desired outward direction.
    The rectangle is defined such that the edge P->Q is one of its shorter sides.
    """
    edge = Q - P
    edge_length = np.linalg.norm(edge)
    if edge_length == 0:
        return None
    edge_unit = edge / edge_length
    # Perpendicular vector (choose the one aligned with offset_direction)
    perp = np.array([-edge_unit[1], edge_unit[0]])
    if np.dot(perp, offset_direction) < 0:
        perp = -perp
    R1 = P
    R2 = Q
    R3 = Q + thickness * perp
    R4 = P + thickness * perp
    return np.array([R1, R2, R3, R4])

# Bone thickness (in mm)
femur_thickness = 100
tibia_thickness = 100

# Set up plot
XLIM, YLIM = 150, 150
fig, ax = plt.subplots()
plt.subplots_adjust(left=0.1, bottom=0.25)
ax.set_aspect('equal')
ax.set_xlim(-XLIM, XLIM)
ax.set_ylim(-YLIM, YLIM)
ax.set_title("Four-Bar Knee Simulation with Bone Rectangles")

# Initial flexion angle (in degrees)
init_deg = 30
phi0 = np.deg2rad(init_deg)
A, B, C, D = four_bar_configuration(phi0)

# Draw the four-bar mechanism for reference.
line, = ax.plot([A[0], B[0], C[0], D[0], A[0]],
                [A[1], B[1], C[1], D[1], A[1]], 'o-', lw=2, color='blue')

# Create femur rectangle (fixed) along the A-D edge.
# For the femur, we'll extend to the left (negative x-direction).
femur_rect_pts = create_rectangle(A, D, femur_thickness, np.array([-1, 0]))
femur_patch = Polygon(femur_rect_pts, closed=True, color='red', alpha=0.5)
ax.add_patch(femur_patch)

# Create tibia rectangle (moving) along the B-C edge.
# For the tibia, we'll extend downward (negative y-direction).
tibia_rect_pts = create_rectangle(B, C, tibia_thickness, np.array([0, -1]))
tibia_patch = Polygon(tibia_rect_pts, closed=True, color='green', alpha=0.5)
ax.add_patch(tibia_patch)

# Add slider: Flexion from -30° to 60°.
ax_slider = plt.axes([0.1, 0.1, 0.8, 0.05])
slider = Slider(ax_slider, 'Flexion (°)', -35, 60, valinit=init_deg)

def update(val):
    phi = np.deg2rad(slider.val)
    A, B, C, D = four_bar_configuration(phi)
    # Update the four-bar mechanism line.
    x_points = [A[0], B[0], C[0], D[0], A[0]]
    y_points = [A[1], B[1], C[1], D[1], A[1]]
    line.set_data(x_points, y_points)
    # Femur remains fixed (A and D don't change).
    # Update tibia rectangle along the new B-C edge.
    new_tibia_rect_pts = create_rectangle(B, C, tibia_thickness, np.array([0, -1]))
    tibia_patch.set_xy(new_tibia_rect_pts)
    fig.canvas.draw_idle()

slider.on_changed(update)
plt.show()
