"""SegModel contract: predict(bgr) -> {class_name: HxW bool mask}.

Wrappers lazy-import their backends and degrade to an empty contribution
(with a single warning) if weights/deps are missing — the pipeline must
never crash because one stage is unavailable.
"""

import logging

log = logging.getLogger("driving_seg")


class SegModel:
    name = "base"

    def __init__(self):
        self._backend = None
        self._dead = False

    def _load(self):                     # -> backend or raise
        raise NotImplementedError

    def predict(self, bgr):              # -> dict[str, bool mask]
        if self._dead:
            return {}
        if self._backend is None:
            try:
                self._backend = self._load()
            except Exception as e:       # missing weights/deps: warn once
                log.warning("%s unavailable (%s); contributing nothing",
                            self.name, e)
                self._dead = True
                return {}
        return self._predict(bgr)

    def _predict(self, bgr):
        raise NotImplementedError
