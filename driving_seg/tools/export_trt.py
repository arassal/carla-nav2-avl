"""Export FP16 TensorRT engines for deployment. Run ON the target machine
(engines are hardware-specific — build Jetson engines on the Jetson).

    python3 tools/export_trt.py                # scene + course (ultralytics)
    python3 tools/export_trt.py --road         # also YOLOPv2 via ONNX

Wrappers accept .engine paths directly (ultralytics loads engines
transparently), e.g. demo --scene-weights models/yolo11n-seg.engine.
"""

import argparse
import os


def export_ultra(weights, imgsz):
    from ultralytics import YOLO
    if not os.path.exists(weights):
        print("skip (missing):", weights)
        return
    YOLO(weights).export(format="engine", half=True, imgsz=imgsz)
    print("exported:", weights.replace(".pt", ".engine"))


def export_road(weights, hw):
    """YOLOPv2 torchscript -> ONNX -> trtexec engine."""
    import torch
    model = torch.jit.load(weights, map_location="cuda").eval().half()
    dummy = torch.zeros(1, 3, hw[0], hw[1]).cuda().half()
    onnx_path = weights.replace(".pt", ".onnx")
    torch.onnx.export(model, dummy, onnx_path, opset_version=13,
                      input_names=["img"],
                      output_names=["det", "drivable", "lane"])
    print("onnx:", onnx_path)
    print("now run: trtexec --onnx=%s --fp16 --saveEngine=%s" %
          (onnx_path, weights.replace(".pt", ".engine")))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--road", action="store_true")
    args = ap.parse_args()
    export_ultra("models/yolo11n-seg.pt", args.imgsz)
    export_ultra("models/course.pt", args.imgsz)
    if args.road:
        export_road("models/yolopv2.pt", (384, 640))
