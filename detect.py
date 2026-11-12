from picamera2 import Picamera2
import cv2
import numpy as np
import time

from gpiozero import LED, Buzzer, Device
from gpiozero.pins.lgpio import LGPIOFactory, LGPIOPin
from gpiozero.pins.local import LocalPiFactory
import lgpio

from ultralytics import YOLO
from ultralytics.nn.autobackend import AutoBackend


# ============================================================
# CONFIGURATION
# ============================================================

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

YOLO_IMAGE_SIZE = 320
YOLO_CONFIDENCE = 0.20

MODEL_PATH = "yolov8n.pt"

BUZZER_PIN = 17
LED_PIN = 27

# COCO vehicle classes
VEHICLE_CLASSES = [2, 3, 5, 7]

CLASS_NAMES = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}

# ------------------------------------------------------------
# Tail-light detection settings
# ------------------------------------------------------------

RED_SAT_MIN = 70
RED_VALUE_MIN = 80

MIN_CONTOUR_AREA = 20
MAX_CONTOUR_AREA = 5000

# Tail lights should be roughly similar in size
MAX_SIZE_RATIO = 3.0

# How close the Y positions must be relative to vehicle size
MAX_Y_DIFFERENCE_RATIO = 0.12

# Tail-light separation relative to vehicle width
MIN_PAIR_DISTANCE_RATIO = 0.10
MAX_PAIR_DISTANCE_RATIO = 0.90

# Tail lights should generally be in the lower/rear part
ROI_TOP_RATIO = 0.20

# Temporal filtering
REQUIRED_DETECTIONS = 3
REQUIRED_MISSES = 5


# ============================================================
# RASPBERRY PI 5 GPIO FIX
# ============================================================

class RP1GPIOFactory(LGPIOFactory):
    """
    Force GPIO Zero to use gpiochip0.

    Useful on Raspberry Pi systems where the default GPIO chip
    selected by the installed GPIO Zero version is not correct.
    """

    def __init__(self):
        LocalPiFactory.__init__(self)

        self._handle = lgpio.gpiochip_open(0)
        self._chip = 0
        self.pin_class = LGPIOPin


Device.pin_factory = RP1GPIOFactory()


# ============================================================
# ULTRALYTICS / TORCHVISION WORKAROUND
# ============================================================

_original_warmup = AutoBackend.warmup


def warmup_without_torchvision(self, *args, **kwargs):
    """
    Skip CPU warm-up when torchvision NMS is unavailable.

    Keeps the workaround from the original program.
    """

    if self.device.type == "cpu" and self.format != "triton":
        return

    return _original_warmup(self, *args, **kwargs)


AutoBackend.warmup = warmup_without_torchvision


# ============================================================
# CAMERA SETUP
# ============================================================

picam2 = Picamera2()

camera_config = picam2.create_video_configuration(
    main={
        "size": (CAMERA_WIDTH, CAMERA_HEIGHT),
        "format": "RGB888"
    }
)

picam2.configure(camera_config)
picam2.start()

picam2.set_controls({
    "AeEnable": True,
    "AwbEnable": True
})

time.sleep(1)


# ============================================================
# GPIO SETUP
# ============================================================

buzzer = Buzzer(BUZZER_PIN)
led = LED(LED_PIN)


# ============================================================
# YOLO SETUP
# ============================================================

print("Loading YOLO model...")

model = YOLO(MODEL_PATH)

print("YOLO model loaded.")
print("Press 'q' to quit.")


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def normalize_frame(frame):
    """
    Picamera2 is configured as RGB888.

    Convert RGB -> BGR because OpenCV expects BGR.
    """

    if frame is None:
        return None

    if len(frame.shape) == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    elif frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

    elif frame.shape[2] == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    return np.ascontiguousarray(frame)


def find_tail_light_candidates(roi):
    """
    Find red/bright regions that could be tail lights.
    """

    if roi is None or roi.size == 0:
        return [], None

    height, width = roi.shape[:2]

    # Focus more on the lower part of the vehicle.
    start_y = int(height * ROI_TOP_RATIO)

    search_roi = roi[start_y:height, :]

    if search_roi.size == 0:
        return [], None

    hsv = cv2.cvtColor(search_roi, cv2.COLOR_BGR2HSV)

    # Red hue wraps around HSV.
    lower_red_1 = np.array(
        [0, RED_SAT_MIN, RED_VALUE_MIN],
        dtype=np.uint8
    )

    upper_red_1 = np.array(
        [10, 255, 255],
        dtype=np.uint8
    )

    lower_red_2 = np.array(
        [170, RED_SAT_MIN, RED_VALUE_MIN],
        dtype=np.uint8
    )

    upper_red_2 = np.array(
        [180, 255, 255],
        dtype=np.uint8
    )

    mask1 = cv2.inRange(
        hsv,
        lower_red_1,
        upper_red_1
    )

    mask2 = cv2.inRange(
        hsv,
        lower_red_2,
        upper_red_2
    )

    mask = cv2.bitwise_or(mask1, mask2)

    # Remove very small noise.
    kernel = np.ones((3, 3), np.uint8)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []

    for contour in contours:

        area = cv2.contourArea(contour)

        if area < MIN_CONTOUR_AREA:
            continue

        if area > MAX_CONTOUR_AREA:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        if w <= 0 or h <= 0:
            continue

        # Tail lights shouldn't be extremely thin.
        aspect_ratio = w / float(h)

        if aspect_ratio < 0.20 or aspect_ratio > 5.0:
            continue

        cx = x + w // 2
        cy = y + h // 2

        # Convert search ROI coordinates back to vehicle ROI.
        cy += start_y

        candidates.append({
            "x": x,
            "y": y + start_y,
            "w": w,
            "h": h,
            "cx": cx,
            "cy": cy,
            "area": area
        })

    return candidates, mask


def find_tail_light_pair(candidates, vehicle_width, vehicle_height):
    """
    Find the best pair of tail-light candidates.

    Uses relative vehicle dimensions instead of fixed pixel
    distances, making it work better with vehicles at different
    distances from the camera.
    """

    if len(candidates) < 2:
        return None

    best_pair = None
    best_score = -1

    min_distance = vehicle_width * MIN_PAIR_DISTANCE_RATIO
    max_distance = vehicle_width * MAX_PAIR_DISTANCE_RATIO

    max_y_difference = vehicle_height * MAX_Y_DIFFERENCE_RATIO

    for i in range(len(candidates)):

        for j in range(i + 1, len(candidates)):

            a = candidates[i]
            b = candidates[j]

            horizontal_distance = abs(a["cx"] - b["cx"])
            vertical_difference = abs(a["cy"] - b["cy"])

            # Must be sufficiently separated.
            if horizontal_distance < min_distance:
                continue

            if horizontal_distance > max_distance:
                continue

            # They should be approximately on the same horizontal line.
            if vertical_difference > max_y_difference:
                continue

            # Similar size.
            smaller_area = min(a["area"], b["area"])
            larger_area = max(a["area"], b["area"])

            size_ratio = larger_area / max(smaller_area, 1)

            if size_ratio > MAX_SIZE_RATIO:
                continue

            # Score the pair.
            horizontal_score = (
                horizontal_distance / max(vehicle_width, 1)
            )

            vertical_score = 1.0 - (
                vertical_difference /
                max(max_y_difference, 1)
            )

            size_score = 1.0 / size_ratio

            score = (
                horizontal_score * 0.45 +
                vertical_score * 0.35 +
                size_score * 0.20
            )

            if score > best_score:
                best_score = score
                best_pair = (a, b)

    return best_pair


def draw_tail_light_pair(roi, pair):
    """
    Draw the detected tail-light pair.
    """

    if pair is None:
        return

    a, b = pair

    cv2.rectangle(
        roi,
        (a["x"], a["y"]),
        (a["x"] + a["w"], a["y"] + a["h"]),
        (0, 255, 0),
        2
    )

    cv2.rectangle(
        roi,
        (b["x"], b["y"]),
        (b["x"] + b["w"], b["y"] + b["h"]),
        (0, 255, 0),
        2
    )

    cv2.line(
        roi,
        (a["cx"], a["cy"]),
        (b["cx"], b["cy"]),
        (255, 255, 0),
        2
    )


# ============================================================
# STATE
# ============================================================

confirmed_count = 0
miss_count = 0

fps = 0.0
last_time = time.time()

show_mask = True


# ============================================================
# MAIN LOOP
# ============================================================

try:

    while True:

        # ----------------------------------------------------
        # Capture
        # ----------------------------------------------------

        frame = picam2.capture_array()

        frame = normalize_frame(frame)

        if frame is None:
            continue

        # Make sure frame has expected size.
        if (
            frame.shape[1] != CAMERA_WIDTH
            or frame.shape[0] != CAMERA_HEIGHT
        ):
            frame = cv2.resize(
                frame,
                (CAMERA_WIDTH, CAMERA_HEIGHT)
            )

        # ----------------------------------------------------
        # YOLO
        # ----------------------------------------------------

        results = model.predict(
            source=frame,
            imgsz=YOLO_IMAGE_SIZE,
            conf=YOLO_CONFIDENCE,
            classes=VEHICLE_CLASSES,
            verbose=False
        )[0]

        detected_tail_light_vehicles = 0

        debug_mask = None

        # ----------------------------------------------------
        # Process vehicles
        # ----------------------------------------------------

        for box in results.boxes:

            cls = int(box.cls[0])
            conf = float(box.conf[0])

            # This is normally already filtered by classes=...
            if cls not in VEHICLE_CLASSES:
                continue

            # Bounding box.
            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            # Clamp coordinates.
            x1 = max(0, min(x1, frame.shape[1] - 1))
            y1 = max(0, min(y1, frame.shape[0] - 1))
            x2 = max(0, min(x2, frame.shape[1] - 1))
            y2 = max(0, min(y2, frame.shape[0] - 1))

            vehicle_width = x2 - x1
            vehicle_height = y2 - y1

            if vehicle_width < 30 or vehicle_height < 30:
                continue

            label = CLASS_NAMES.get(
                cls,
                "Vehicle"
            )

            # ------------------------------------------------
            # Vehicle box
            # ------------------------------------------------

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (255, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"{label} {conf:.2f}",
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2
            )

            # ------------------------------------------------
            # Vehicle ROI
            # ------------------------------------------------

            roi = frame[
                y1:y2,
                x1:x2
            ]

            if roi.size == 0:
                continue

            # ------------------------------------------------
            # Tail light detection
            # ------------------------------------------------

            candidates, mask = find_tail_light_candidates(
                roi
            )

            if mask is not None:
                debug_mask = mask

            pair = find_tail_light_pair(
                candidates,
                vehicle_width,
                vehicle_height
            )

            # Draw candidate regions.
            for candidate in candidates:

                cv2.rectangle(
                    roi,
                    (
                        candidate["x"],
                        candidate["y"]
                    ),
                    (
                        candidate["x"] + candidate["w"],
                        candidate["y"] + candidate["h"]
                    ),
                    (0, 165, 255),
                    1
                )

            # ------------------------------------------------
            # Tail light confirmed
            # ------------------------------------------------

            if pair is not None:

                detected_tail_light_vehicles += 1

                draw_tail_light_pair(
                    roi,
                    pair
                )

                cv2.putText(
                    frame,
                    "TAIL LIGHT",
                    (x1, max(45, y1 - 30)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 0, 255),
                    2
                )

        # ====================================================
        # TEMPORAL FILTER
        # ====================================================

        if detected_tail_light_vehicles > 0:

            confirmed_count += 1
            miss_count = 0

        else:

            miss_count += 1
            confirmed_count = 0

        # Turn ON only after several consecutive detections.
        alarm_active = (
            confirmed_count >= REQUIRED_DETECTIONS
        )

        # Turn OFF after several consecutive misses.
        if miss_count >= REQUIRED_MISSES:
            alarm_active = False

        # ====================================================
        # GPIO OUTPUT
        # ====================================================

        if alarm_active:
            buzzer.on()
            led.on()
        else:
            buzzer.off()
            led.off()

        # ====================================================
        # FPS
        # ====================================================

        current_time = time.time()

        elapsed = current_time - last_time

        if elapsed > 0:
            instant_fps = 1.0 / elapsed

            # Smooth FPS.
            fps = (
                fps * 0.90 +
                instant_fps * 0.10
            )

        last_time = current_time

        # ====================================================
        # DISPLAY
        # ====================================================

        status = "ALERT" if alarm_active else "NORMAL"

        status_color = (
            (0, 0, 255)
            if alarm_active
            else (0, 255, 0)
        )

        cv2.putText(
            frame,
            f"Tail-light vehicles: {detected_tail_light_vehicles}",
            (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Status: {status}",
            (15, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            status_color,
            2
        )

        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (15, 95),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            "Vehicle Tail Light Detection",
            frame
        )

        # ----------------------------------------------------
        # Debug mask
        # ----------------------------------------------------

        if show_mask and debug_mask is not None:
            cv2.imshow(
                "Tail Light Mask",
                debug_mask
            )

        # ----------------------------------------------------
        # Keyboard
        # ----------------------------------------------------

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        # Toggle mask with M.
        elif key == ord("m"):
            show_mask = not show_mask

            if not show_mask:
                cv2.destroyWindow(
                    "Tail Light Mask"
                )


# ============================================================
# CLEANUP
# ============================================================

finally:

    print("Stopping...")

    buzzer.off()
    led.off()

    try:
        picam2.stop()
    except Exception:
        pass

    cv2.destroyAllWindows()

