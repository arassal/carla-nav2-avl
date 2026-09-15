import time
import threading

from perception_costmap.inference_worker import LatestTaskWorker


def _wait_for_result(worker, timeout=1.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        result = worker.take_latest()
        if result is not None:
            return result
        time.sleep(0.005)
    raise AssertionError("worker result timed out")


def test_worker_returns_processed_result():
    worker = LatestTaskWorker(lambda value: value * 2)
    try:
        worker.submit(3)
        assert _wait_for_result(worker) == 6
    finally:
        worker.close()


def test_worker_surfaces_processor_error():
    def fail(_value):
        raise ValueError("bad inference")

    worker = LatestTaskWorker(fail)
    try:
        worker.submit(1)
        end = time.monotonic() + 1.0
        error = None
        while time.monotonic() < end and error is None:
            error = worker.take_error()
            time.sleep(0.005)
        assert isinstance(error, ValueError)
    finally:
        worker.close()


def test_worker_replaces_stale_pending_task():
    started = threading.Event()
    release = threading.Event()

    def process(value):
        if value == 1:
            started.set()
            assert release.wait(timeout=1.0)
        return value

    worker = LatestTaskWorker(process)
    try:
        worker.submit(1)
        assert started.wait(timeout=1.0)
        worker.submit(2)
        worker.submit(3)
        release.set()
        end = time.monotonic() + 1.0
        while time.monotonic() < end and worker.completed < 2:
            time.sleep(0.005)
        assert worker.take_latest() == 3
        assert worker.replaced == 1
    finally:
        worker.close()
