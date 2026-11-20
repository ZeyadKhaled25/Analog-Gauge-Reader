# Analog Gauge Reader

Detects analog gauge keypoints with YOLOv8 and converts the needle position into a timestamped reading. This is the gauge reading module of **Project HELIOS**, an autonomous industrial inspection rover for combined-cycle power plants.

![Demo](docs/demo.gif)

## Overview

Power plants and factories still rely on people to read analog gauges by hand. This project reads them automatically from a camera feed. A YOLOv8n model locates each gauge and its keypoints, the gauge's scale range is detected automatically, and the needle position is converted into a numeric reading. Every reading is logged with a timestamp and a status to a CSV file for condition monitoring.

## Results

| Metric | Value | Setup |
| --- | --- | --- |
| mAP@0.5 | **0.897** | YOLOv8n, validation data |

## Features

- Detects each gauge and four keypoints: needle tip, needle center, scale minimum, and scale maximum
- Detects each gauge's scale range automatically
- Converts needle position into a reading
- Logs timestamped values with a status to `gauge_readings.csv`
- Runs live on a camera or video input

## How it works

```
Camera frame
    -> YOLOv8n detector (gauge + keypoints)
    -> scale range detection
    -> needle position between scale minimum and maximum
    -> reading
    -> timestamped CSV log
```

1. **Detect.** YOLOv8n finds the gauge and its keypoints in each frame.
2. **Calibrate.** The scale minimum and maximum keypoints, together with the detected scale range, define the gauge's range, so no manual setup is needed per gauge.
3. **Measure.** The needle position relative to the scale endpoints is converted into a reading.
4. **Log.** Each reading is saved with a timestamp and a status.

## Repository structure

```
analog-gauge-reader/
├── train_gauge.py        # trains the YOLOv8n gauge and keypoint detector
├── gauge_live_local.py   # live pipeline: detect, read, log
├── data.yaml             # dataset configuration (Roboflow export)
├── gauge_data.yaml       # dataset configuration used for training
├── models/               # trained weights (best.pt)
├── docs/                 # demo GIF
├── README.roboflow.txt   # dataset source and license from Roboflow
├── requirements.txt
└── README.md
```

## Dataset

The training data was exported from Roboflow in YOLOv8 format, with `train`, `valid`, and `test` splits. The images are not included in this repository because of their size. See `README.roboflow.txt` for the dataset source and license.

## Getting started

### Requirements

- Python 3.9 or newer
- A webcam or a video file for the live pipeline

### Install

```bash
git clone https://github.com/ZeyadKhaled25/analog-gauge-reader.git
cd analog-gauge-reader
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux and macOS
pip install -r requirements.txt
```

### Train

Export the dataset from Roboflow in YOLOv8 format into this folder (so `train/`, `valid/`, and `test/` sit next to `data.yaml`), then run:

```bash
python train_gauge.py
```

Training output is written to `runs/`. Copy the best weights to `models/best.pt`.

### Run the live reader

```bash
python gauge_live_local.py
```

Readings are appended to `gauge_readings.csv` with a timestamp and status.

## Part of Project HELIOS

HELIOS is a modular autonomous inspection rover for combined-cycle power plants. It combines ROS 2 navigation with an onboard computer vision stack for gauge reading and PPE compliance monitoring. This repository is its gauge reading perception module.

## Author

Zeyad Khaled Ehab: [LinkedIn](https://www.linkedin.com/in/zeyad-khaled2) | zeyadkhaledehab@gmail.com
