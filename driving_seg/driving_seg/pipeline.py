"""Run the three models concurrently, merge their mask dicts, time stages."""

import time
from concurrent.futures import ThreadPoolExecutor

from . import fusion


class Pipeline:
    def __init__(self, models=None):
        if models is None:
            from .models.scene import SceneModel
            from .models.road import RoadModel
            from .models.course import CourseModel
            models = [SceneModel(), RoadModel(), CourseModel()]
        self.models = models
        self._pool = ThreadPoolExecutor(max_workers=len(models))
        self.timings = {}                # stage -> last ms

    def masks(self, bgr):
        """-> merged {class: bool mask}; updates self.timings."""
        def run(m):
            t0 = time.perf_counter()
            r = m.predict(bgr)
            return m.name, r, (time.perf_counter() - t0) * 1000.0

        merged = {}
        for name, r, ms in self._pool.map(run, self.models):
            self.timings[name] = ms
            for cls, mask in r.items():
                merged[cls] = merged[cls] | mask if cls in merged else mask
        return merged

    def overlay(self, bgr, draw_legend=True):
        t0 = time.perf_counter()
        masks = self.masks(bgr)
        out = fusion.compose(bgr, masks)
        if draw_legend:
            fusion.legend(out, set(masks))
        self.timings["total"] = (time.perf_counter() - t0) * 1000.0
        return out, masks
