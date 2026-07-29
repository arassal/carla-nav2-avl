"""Class registry: the fixed contract from PROMPT.md.

Every class has a BGR color, a compositing priority (higher wins the
pixel), and the model stage that produces it.
"""

from collections import OrderedDict

# name -> dict(color=BGR, priority=int, model=str)
CLASSES = OrderedDict([
    ("road",          dict(color=(80, 200, 80),    priority=1, model="road")),
    ("lane_line",     dict(color=(230, 230, 80),   priority=2, model="road")),
    ("white_line",    dict(color=(255, 255, 255),  priority=3, model="course")),
    ("traffic_light", dict(color=(60, 170, 255),   priority=4, model="scene")),
    ("traffic_sign",  dict(color=(60, 220, 230),   priority=5, model="scene")),
    ("vehicle",       dict(color=(230, 120, 60),   priority=6, model="scene")),
    ("cone",          dict(color=(40, 120, 255),   priority=7, model="course")),
    ("person",        dict(color=(60, 60, 230),    priority=8, model="scene")),
])

MODELS = ("scene", "road", "course")

OVERLAY_ALPHA = 0.45      # translucency of the class fill
CONTOUR_GAIN = 1.35       # region outline brightness multiplier


def classes_for_model(model_name):
    return [n for n, c in CLASSES.items() if c["model"] == model_name]


def paint_order():
    """Class names sorted low->high priority (paint low first)."""
    return sorted(CLASSES, key=lambda n: CLASSES[n]["priority"])
