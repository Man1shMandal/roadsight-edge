# RoadSight Edge

Vehicle tail-light detector for a Raspberry Pi 5 with a Pi camera. When it sees a vehicle ahead with its tail lights on, it sounds a buzzer and turns on an LED.

Each frame goes through two stages:

1. YOLOv8n finds cars, motorcycles, buses and trucks (COCO classes 2, 3, 5, 7).
2. Inside each vehicle box, an HSV red mask finds bright red blobs in the lower part of the vehicle, and the script looks for a matching pair: similar size, roughly level, and spaced a sensible fraction of the vehicle width apart.

The alert only turns on after 3 consecutive frames with a tail-light pair and turns off after 5 frames without one, so a single noisy frame won't trigger it.

A preview window shows vehicle boxes, candidate blobs, the matched pair, the alert status and FPS. A second window shows the red mask for tuning.

## Hardware

- Raspberry Pi 5
- Pi camera module (Picamera2)
- Buzzer and LED

| Part | BCM GPIO |
| --- | --- |
| Buzzer | 17 |
| LED | 27 |

The script sets GPIO Zero to use `gpiochip0` through lgpio, which is what the Pi 5 needs.

## Stack

- Python 3
- Ultralytics YOLOv8 (`yolov8n.pt` included)
- OpenCV, NumPy
- Picamera2
- gpiozero + lgpio

## Run

On Raspberry Pi OS (Bookworm), Picamera2 and lgpio come from apt. Create the venv with system packages so it can see them:

```bash
sudo apt install -y python3-picamera2 python3-lgpio
python3 -m venv --system-site-packages venv
source venv/bin/activate
pip install ultralytics opencv-python numpy gpiozero
python detect.py
```

It needs a desktop session (or X forwarding) for the preview windows.

Keys:

- `q` - quit
- `m` - show/hide the mask window

## Tuning

All settings are constants at the top of `detect.py`: camera resolution (640x480), YOLO input size (320) and confidence (0.20), red saturation/value thresholds, blob size limits, pair spacing ratios, and the number of frames needed to turn the alert on or off.
