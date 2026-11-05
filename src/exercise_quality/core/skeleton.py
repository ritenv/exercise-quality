
import numpy as np
import matplotlib.pyplot as plt

# Body segment lengths in meters
SHIN_LENGTH = 0.43
THIGH_LENGTH = 0.45
TORSO_LENGTH = 0.55
NECK_LENGTH = 0.12
FOOT_LENGTH = 0.25

# Joint positions (x, y), measured in meters

def calculate_joints(
    shin_angle=0,
    thigh_angle=0,
    torso_angle=0,
):
    """Angles in degrees, measured from vertical."""
    shin = np.radians(shin_angle)
    thigh = np.radians(thigh_angle)
    torso = np.radians(torso_angle)

    ankle = np.array([0.0, 0.0])
    toe = ankle + np.array([FOOT_LENGTH, 0.0])

    knee = ankle + SHIN_LENGTH * np.array([
        np.sin(shin), np.cos(shin)
    ])

    hip = knee + THIGH_LENGTH * np.array([
        np.sin(thigh), np.cos(thigh)
    ])

    shoulder = hip + TORSO_LENGTH * np.array([
        np.sin(torso), np.cos(torso)
    ])

    head = shoulder + NECK_LENGTH * np.array([
        np.sin(torso), np.cos(torso)
    ])

    return {
        "ankle": ankle,
        "knee": knee,
        "hip": hip,
        "shoulder": shoulder,
        "head": head,
        "toe": toe,
    }


joints = calculate_joints(
    shin_angle=25,
    thigh_angle=-65,
    torso_angle=35,
)

connections = [
    ("ankle", "knee"),
    ("knee", "hip"),
    ("hip", "shoulder"),
    ("shoulder", "head"),
    ("ankle", "toe"),
]


def draw_skeleton():
    fig, ax = plt.subplots(figsize=(5, 7))

    for start, end in connections:
        p1 = joints[start]
        p2 = joints[end]

        ax.plot(
            [p1[0], p2[0]],
            [p1[1], p2[1]],
            linewidth=4,
            color="steelblue",
        )

    for name, position in joints.items():
        ax.scatter(*position, s=80, color="darkorange")
        ax.annotate(name, position, xytext=(10, 5),
                    textcoords="offset points")

    ax.set_xlim(-0.8, 0.8)
    ax.set_ylim(-0.1, 1.8)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)
    ax.set_title("Exercise Quality — 2D Skeleton")

    plt.show()


if __name__ == "__main__":
    draw_skeleton()
