"""Bounded multi-camera inference scheduling."""


class DetectionScheduler:
    """Run the primary camera every tick and rotate secondary cameras.

    ``secondary_stride`` limits extra inference load. A value of one selects
    one secondary every tick; two selects one every other tick.
    """

    def __init__(self, primary="front", secondary_stride=1):
        self.primary = primary
        self.secondary_stride = max(1, int(secondary_stride))
        self.tick = 0
        self.secondary_index = 0

    def select(self, available):
        available = list(available)
        selected = []
        if self.primary in available:
            selected.append(self.primary)

        secondary = [name for name in available if name != self.primary]
        if secondary and self.tick % self.secondary_stride == 0:
            selected.append(secondary[self.secondary_index % len(secondary)])
            self.secondary_index += 1
        self.tick += 1
        return selected
