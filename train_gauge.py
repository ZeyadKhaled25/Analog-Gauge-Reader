"""
Train a YOLOv8 detector on the Roboflow gauge dataset (classes: center, gauge, max, min, tip).

    python -m pip install ultralytics
    python train_gauge.py

Result: runs/gauge/weights/best.pt (next to this script)
"""
import os

from ultralytics import YOLO

DATASET_DIR = r"D:\gauge-analog.v1i.yolov8"   # folder that contains train/ valid/ test/
EPOCHS = 20                            # lower to 30-50 for a quick test on CPU
HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    # Roboflow's data.yaml uses "../train/images", which breaks depending on where you run from.
    # Write a clean yaml with absolute paths instead.
    ds = DATASET_DIR.replace("\\", "/")
    data_yaml = os.path.join(HERE, "gauge_data.yaml")
    with open(data_yaml, "w") as f:
        f.write(
            f"path: {ds}\n"
            "train: train/images\n"
            "val: valid/images\n"
            "test: test/images\n"
            "nc: 5\n"
            "names: ['center', 'gauge', 'max', 'min', 'tip']\n"
        )

    model = YOLO("yolov8n.pt")
    model.train(
        data=data_yaml,
        epochs=EPOCHS,
        imgsz=640,
        project=os.path.join(HERE, "runs"),
        name="gauge",
        exist_ok=True,
        fliplr=0.0,   # mirroring a gauge swaps min/max sides and reverses the scale; don't do it
    )


if __name__ == "__main__":   # required on Windows (dataloader worker processes)
    main()
