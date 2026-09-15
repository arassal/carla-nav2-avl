"""Small timestamped buffers for synchronizing perception samples."""

from collections import deque
import math


class TimestampedBuffer:
    """Bounded timestamp/value buffer with nearest-sample lookup."""

    def __init__(self, maxlen=12):
        self.samples = deque(maxlen=int(maxlen))

    def add(self, stamp, value):
        self.samples.append((float(stamp), value))

    def nearest(self, stamp, max_delta):
        if not self.samples:
            return None
        target = float(stamp)
        best = min(self.samples, key=lambda sample: abs(sample[0] - target))
        if abs(best[0] - target) > float(max_delta):
            return None
        return best


class PoseBuffer(TimestampedBuffer):
    """Timestamped ``(x, y, yaw)`` poses with shortest-arc interpolation."""

    def interpolate(self, stamp, max_delta):
        if not self.samples:
            return None
        target = float(stamp)
        before = None
        after = None
        for sample in self.samples:
            if sample[0] <= target and (before is None or sample[0] > before[0]):
                before = sample
            if sample[0] >= target and (after is None or sample[0] < after[0]):
                after = sample

        if before is None or after is None:
            nearest = self.nearest(target, max_delta)
            return None if nearest is None else nearest[1]
        if max(target - before[0], after[0] - target) > float(max_delta):
            return None
        if after[0] == before[0]:
            return before[1]

        fraction = (target - before[0]) / (after[0] - before[0])
        bx, by, byaw = before[1]
        ax, ay, ayaw = after[1]
        yaw_delta = math.atan2(math.sin(ayaw - byaw), math.cos(ayaw - byaw))
        return (
            bx + fraction * (ax - bx),
            by + fraction * (ay - by),
            math.atan2(
                math.sin(byaw + fraction * yaw_delta),
                math.cos(byaw + fraction * yaw_delta)),
        )
