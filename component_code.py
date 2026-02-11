import time
import os
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import psutil
from gpiozero import LED, Servo, Button, DistanceSensor

# ------------------ CONFIG ------------------
SHOW_WINDOW = False          
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
BACKGROUND_BLUR = (11, 11)
DIFF_THRESHOLD = 30
MIN_CONTOUR_AREA = 500
INFERENCE_INTERVAL = 0.2    

# GPIO pins (BCM numbering)
TRIG_PIN = 5
ECHO_PIN = 6
RESET_BUTTON_PIN = 7
RED_LED_PIN = 22
WHITE_LED_PIN = 23
SERVO1_PIN = 18
SERVO2_PIN = 19  

# ------------------ COMPONENT SETUP ------------------
# We set pulse widths to 0.5ms - 2.5ms which is standard for most servos
REDLED = LED(RED_LED_PIN)
WHITELED = LED(WHITE_LED_PIN)
SERVO1 = Servo(SERVO1_PIN, min_pulse_width=0.5/1000, max_pulse_width=2.5/1000)
SERVO2 = Servo(SERVO2_PIN, min_pulse_width=0.5/1000, max_pulse_width=2.5/1000)
RESETBUTTON = Button(RESET_BUTTON_PIN, pull_up=True)

# Note: If distance still says "No Echo", check that the sensor has 5V power!
ULTRASONIC = DistanceSensor(echo=ECHO_PIN, trigger=TRIG_PIN, max_distance=2)

containerFilled = False
peak_ram_mb = 0
process = psutil.Process()

# ------------------ HELPER FUNCTIONS ------------------
def get_distance():
    return round(ULTRASONIC.distance * 100, 2)

def rotate_servo(servo, degree):
    """Moves servo and then detaches signal to stop twitching/jitter."""
    # Maps 0-180 to -1 to 1
    target_value = (degree / 90.0) - 1.0
    servo.value = target_value
    time.sleep(0.8)  # Time to reach position
    servo.value = None  # STOP signal to kill jitter

def load_labels(labels_path="labels.txt"):
    try:
        with open(labels_path, 'r') as f:
            return [line.strip() for line in f.readlines()]
    except FileNotFoundError:
        print("labels.txt not found")
        return None

def softmax(x):
    x = np.clip(x, -500, 500)
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()

def detect_object(frame, bg_gray):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, BACKGROUND_BLUR, 0)
    diff = cv2.absdiff(gray, bg_gray)
    _, mask = cv2.threshold(diff, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) >= MIN_CONTOUR_AREA:
            x, y, w, h = cv2.boundingRect(largest)
            return True, frame[y:y+h, x:x+w]
    return False, frame

def classify_object(interpreter, input_details, output_details, img, input_dtype, labels):
    img_resized = cv2.resize(img, (input_details[0]['shape'][2], input_details[0]['shape'][1]))
    if input_dtype != np.uint8:
        img_resized = img_resized.astype(np.float32) / 255.0
    img_resized = np.expand_dims(img_resized, axis=0)
    interpreter.set_tensor(input_details[0]['index'], img_resized)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])[0]
    probabilities = softmax(output)
    class_id = np.argmax(probabilities)
    confidence = probabilities[class_id]
    label = labels[class_id] if labels else f"Class {class_id}"
    return label, confidence

# ------------------ LOAD MODEL ------------------
labels = load_labels()
interpreter = tflite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_dtype = input_details[0]['dtype']

# ------------------ CAMERA SETUP (BOOKWORM FIX) ------------------
# We use V4L2 and MJPG to avoid the select() timeout errors
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

time.sleep(2)
ret, frame_bg = cap.read()
if ret:
    frame_bg = frame_bg[100:500, 100:500] 
    bg_gray = cv2.cvtColor(frame_bg, cv2.COLOR_BGR2GRAY)
    bg_gray = cv2.GaussianBlur(bg_gray, BACKGROUND_BLUR, 0)
else:
    print("Failed to capture background. Check camera connection.")

print("Setup complete. Starting loop...")

# ------------------ MAIN LOOP ------------------
last_inference_time = 0
last_valid_label = "Waiting..."
frame_count = 0

try:
    while True:
        frame_count += 1
        ret, frame = cap.read()
        if not ret: continue
        
        frame = frame[100:500, 100:500]

        if RESETBUTTON.is_pressed:
            containerFilled = False
            REDLED.off()
            print("Reset Pressed")

        object_detected, obj_frame = detect_object(frame, bg_gray)

        if containerFilled:
            REDLED.on()
        elif object_detected:
            WHITELED.on()
            current_time = time.time()
            if current_time - last_inference_time >= INFERENCE_INTERVAL:
                label, confidence = classify_object(interpreter, input_details, output_details, obj_frame, input_dtype, labels)
                if confidence > 0.3:
                    last_valid_label = label
                    print(f">>> {label} ({confidence*100:.1f}%)")

                    if label.lower() == "plastic":
                        rotate_servo(SERVO1, 45)  # Adjust degrees as needed
                        time.sleep(1)
                        rotate_servo(SERVO1, 90)
                    elif label.lower() == "aluminum":
                        rotate_servo(SERVO2, 45)
                        time.sleep(1)
                        rotate_servo(SERVO2, 90)

                last_inference_time = current_time
        else:
            WHITELED.off()

        if SHOW_WINDOW:
            cv2.imshow("Feed", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

        peak_ram_mb = max(peak_ram_mb, process.memory_info().rss / 1024 / 1024)
        time.sleep(0.01)

except KeyboardInterrupt:
    print("Stopping...")
finally:
    cap.release()
    cv2.destroyAllWindows()
    # Servos are released automatically by gpiozero
