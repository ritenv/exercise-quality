"""Animate one 3D squat rep: 3D view you can rotate with the mouse.

    PYTHONPATH=src python scripts/animate_squat_3d.py [seed] [--cave DEG] [--cave-right DEG]
        [--shift M] [--protocol ishida|zawadka] [--save out.gif]

Options and colours: see squat_animation.py. The same seed gives the same
person and rep in animate_squat_3d.py, animate_squat_side.py and
animate_squat_front.py.
"""

from squat_animation import run

if __name__ == "__main__":
    run("3d")
