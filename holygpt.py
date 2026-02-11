import time
import cv2
import numpy as np
import os
import subprocess
import psutil
from gpiozero import LED, Button, Servo, DistanceSensor
import tflite_runtime.interpreter as tflite

# ================= CONFIG =================
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
BACKGROUND_BLUR = (11, 11)
DIFF_THRESHOLD = 30
MIN_CONTOUR_AREA = 600
CONFIDENCE_THRESHOLD = 0.3

# ================= GPIO =================
WHITE_LED = LED(23)
RED_LED = LED(22)
RESET_BUTTON = Button(7)
SERVO = Servo(18)
ULTRASONIC = DistanceSensor(trigger=5, echo=6)

# ================= STATE =================
isFull = False
full_hits = 0
FULL_CONFIRMATIONS = 3

# ================= SERVO ANGLES =================
SERVO_CENTER = 0
SERVO_PLASTIC = -60
SERVO_ALUMINUM = 60
SERVO_LOCKED = 180

# ================= SERVO =================
def rotate_servo(deg):
    SERVO.value = max(-1, min(1, deg / 90))
    time.sleep(0.4)

# ================= DISTANCE =================
def get_distance_cm():
    return ULTRASONIC.distance * 100

def record_fill_amount():
    global isFull, full_hits
    readings = []

    for _ in range(5):
        d = get_distance_cm()
        if d < 99:
            readings.append(d)
        time.sleep(0.05)

    if not readings:
        return

    avg = sum(readings) / len(readings)
    print(f"Avg distance: {avg:.2f} cm")

    if avg < 4:
        full_hits += 1
    else:
        full_hits = 0

    if full_hits >= FULL_CONFIRMATIONS:
        isFull = True

# ================= AI MODEL =================
def load_labels(path="labels.txt"):
    try:
        with open(path) as f:
            return [l.strip() for l in f]
    except:
        return None

labels = load_labels()

interpreter = tflite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]
input_dtype = input_details[0]['dtype']

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

# ================= CAMERA PIPELINE =================
fifo = "/tmp/camera_stream.h264"
if os.path.exists(fifo):
    os.remove(fifo)
os.mkfifo(fifo)

os.system("pkill rpicam-vid")
time.sleep(1)

os.system(
    f"rpicam-vid -t 0 --width {FRAME_WIDTH} --height {FRAME_HEIGHT} "
    f"--codec h264 -o {fifo} --inline --listen &"
)
time.sleep(2)

ffmpeg = subprocess.Popen(
    [
        "ffmpeg", "-i", fifo,
        "-f", "image2pipe",
        "-pix_fmt", "rgb24",
        "-vcodec", "rawvideo", "-"
    ],
    stdout=subprocess.PIPE,
    stderr=subprocess.DEVNULL,
    bufsize=10**8
)

FRAME_SIZE = FRAME_WIDTH * FRAME_HEIGHT * 3

# ================= BACKGROUND =================
raw_bg = ffmpeg.stdout.read(FRAME_SIZE)
bg = np.frombuffer(raw_bg, np.uint8).reshape((FRAME_HEIGHT, FRAME_WIDTH, 3))
bg_gray = cv2.cvtColor(bg, cv2.COLOR_RGB2GRAY)
bg_gray = cv2.GaussianBlur(bg_gray, BACKGROUND_BLUR, 0)

# ================= VISION =================
def detect_object(frame_rgb):
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, BACKGROUND_BLUR, 0)
    diff = cv2.absdiff(gray, bg_gray)
    _, mask = cv2.threshold(diff, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        if cv2.contourArea(c) >= MIN_CONTOUR_AREA:
            return True, contours
    return False, contours

def classify_object(frame_rgb, contours):
    if contours:
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        obj = frame_rgb[y:y+h, x:x+w]
    else:
        obj = frame_rgb

    img = cv2.resize(obj, (width, height))
    if input_dtype != np.uint8:
        img = img.astype(np.float32) / 255.0

    img = np.expand_dims(img, axis=0)
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])[0]
    probs = softmax(output)

    class_id = np.argmax(probs)
    confidence = probs[class_id]

    if confidence < CONFIDENCE_THRESHOLD:
        return None

    if labels:
        return labels[class_id]
    return str(class_id)

# ================= MAIN LOOP =================
try:
    rotate_servo(SERVO_CENTER)

    while True:
        if RESET_BUTTON.is_active:
            isFull = False
            full_hits = 0
            RED_LED.off()
            print("Reset pressed")

        raw = ffmpeg.stdout.read(FRAME_SIZE)
        if len(raw) != FRAME_SIZE:
            continue

        frame_rgb = np.frombuffer(raw, np.uint8).reshape((FRAME_HEIGHT, FRAME_WIDTH, 3))
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        detected, contours = detect_object(frame_rgb)

        if detected:
            WHITE_LED.on()

            if isFull:
                WHITE_LED.off()
                RED_LED.on()
                rotate_servo(SERVO_LOCKED)
                print("BIN FULL")
                continue

            trash_type = classify_object(frame_rgb, contours)
            WHITE_LED.off()

            if trash_type == "plastic":
                rotate_servo(SERVO_PLASTIC)
            elif trash_type == "aluminum":
                rotate_servo(SERVO_ALUMINUM)
            else:
                rotate_servo(SERVO_CENTER)

            print(f"Sorted: {trash_type}")
            time.sleep(2)
            record_fill_amount()
            rotate_servo(SERVO_CENTER)

        cv2.imshow("Camera", frame_bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Stopping")

finally:
    ffmpeg.terminate()
    os.system("pkill rpicam-vid")
    if os.path.exists(fifo):
        os.remove(fifo)
    cv2.destroyAllWindows()
