"""Single-worker latest-task execution for bounded detector latency."""

import queue
import threading


class LatestTaskWorker:
    """Process tasks serially and replace stale queued work with the newest."""

    def __init__(self, processor, name="perception-inference"):
        self.processor = processor
        self.tasks = queue.Queue(maxsize=1)
        self.lock = threading.Lock()
        self.result = None
        self.error = None
        self.closed = False
        self.submitted = 0
        self.replaced = 0
        self.completed = 0
        self.thread = threading.Thread(
            target=self._run, name=name, daemon=True)
        self.thread.start()

    def submit(self, task):
        if self.closed:
            return False
        self.submitted += 1
        try:
            self.tasks.put_nowait(task)
            return True
        except queue.Full:
            try:
                self.tasks.get_nowait()
            except queue.Empty:
                pass
            self.replaced += 1
            self.tasks.put_nowait(task)
            return True

    def take_latest(self):
        with self.lock:
            result, self.result = self.result, None
            return result

    def take_error(self):
        with self.lock:
            error, self.error = self.error, None
            return error

    def close(self, timeout=2.0):
        self.closed = True
        try:
            self.tasks.put_nowait(None)
        except queue.Full:
            try:
                self.tasks.get_nowait()
            except queue.Empty:
                pass
            self.tasks.put_nowait(None)
        self.thread.join(timeout=timeout)

    def _run(self):
        while True:
            task = self.tasks.get()
            if task is None:
                return
            try:
                result = self.processor(task)
                with self.lock:
                    self.result = result
                    self.completed += 1
            except Exception as error:  # surfaced on the ROS timer thread
                with self.lock:
                    self.error = error
