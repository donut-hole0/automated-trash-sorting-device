import RPi.GPIO as GPIO
import time
#from gpiozero import LED, Button, Servo, DistanceSensor
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
from picamera2 import Picamera2

import pigpio
pi = pigpio.pi()

#camera code
picam = Picamera2()
config = picam.create_preview_configuration(
    main={"format": "RGB888", "size": (640, 480)}
)
picam.configure(config)

#AI model setup
# Load labels
with open("labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

# Load model
interpreter = tflite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

#Electrical component setup
GPIO.setmode(GPIO.BCM)

#SENSOR = DistanceSensor(echo=6, trigger=5)
ECHO_PIN = 6
TRIG_PIN = 5
RESETBUTTON_PIN = 7
LIM_SWITCH_PIN = 8
"""You Can Change These GPIOs If Needed"""

REDLED_PIN = 22
WHITELED_PIN = 23
SERVO_R_PIN = 18
SERVO_C_PIN = 17
	
pi.set_mode(TRIG_PIN, pigpio.OUTPUT)
pi.set_mode(ECHO_PIN, pigpio.INPUT)
pi.set_mode(RESETBUTTON_PIN, pigpio.INPUT)
pi.set_mode(LIM_SWITCH_PIN, pigpio.INPUT)
pi.set_mode(REDLED_PIN, pigpio.OUTPUT)
pi.set_mode(WHITELED_PIN, pigpio.OUTPUT)
mult = 1.913

frame1 = None

containerFilled = False

def get_frame():
    frame = picam.capture_array()
    return frame[100:500, 100:500]

def get_distance():
    pi.write(TRIG_PIN, 0)
    time.sleep(0.05)

    pi.write(TRIG_PIN, 1)
    time.sleep(0.00001)
    pi.write(TRIG_PIN, 0)

    timeout = time.time() + 0.2

    while pi.read(ECHO_PIN)== 0:
        pulse_start = time.time()
        if time.time() > timeout:
            return -1

    while pi.read(ECHO_PIN) == 1:
        pulse_end = time.time()
        if time.time() > timeout:
            return -1

    pulse_duration = pulse_end - pulse_start
    distance = pulse_duration * 17150
    distance = round(distance, 2)

    return distance

def detect_object():
    frame2 = get_frame()
    if frame1 is not None:
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        blurredGray1 = cv2.GaussianBlur(gray1, (11, 11), 0)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        blurredGray2 = cv2.GaussianBlur(gray2, (11, 11), 0)
        diff = cv2.absdiff(blurredGray1, blurredGray2)
        _, threshedDiff = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(threshedDiff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) >= 600: #change to 500 or 550? Or maybe 650?
                return True
    return False

def classify_object():
    frame3 = get_frame()
    # Resize to model size (usually 224x224)
    frame3 = cv2.resize(frame3, (224, 224))

    # Convert to numpy and normalize
    input_data = np.array(frame3, dtype=np.float32)

    # IMPORTANT: try this first (most Teachable Machine models use this)
    input_data = (input_data / 127.5) - 1

    # Add batch dimension
    input_data = np.expand_dims(input_data, axis=0)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Get results
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Get prediction
    predicted_index = np.argmax(output_data)

    return labels[predicted_index]

def rotate_servo(duration, direction):
  time.sleep(0.01)
  pi.set_servo_pulsewidth(SERVO_R_PIN, 1500 + int(444 * direction))
  time.sleep(duration*mult/1000)
  pi.set_servo_pulsewidth(SERVO_R_PIN, 1500)
  time.sleep(0.03)
  pi.set_servo_pulsewidth(SERVO_R_PIN, 1500)
  time.sleep(0.1)

def return_home():
    time.sleep(0.01)
    print("returnSpin")
    pi.set_servo_pulsewidth(SERVO_R_PIN, 1944) #change to -0.44 if wrong direction
    start = time.monotonic() * 1000
    while start+700 > time.monotonic()*1000: #700ms is rotation duration
        if pi.read(LIM_SWITCH_PIN) == 1:
            pi.set_servo_pulsewidth(SERVO_R_PIN, 1500)
            pi.set_servo_pulsewidth(SERVO_R_PIN, 0)
            print("home")
            return

    pi.set_servo_pulsewidth(SERVO_R_PIN, 1500)
    pi.set_servo_pulsewidth(SERVO_R_PIN, 0)
    print("home")
   

def open_chute():
    pi.set_servo_pulsewidth(SERVO_C_PIN, 1500)
    time.sleep(0.16)
    pi.set_servo_pulsewidth(SERVO_C_PIN, 0)
    print("Chute opened")

def close_chute():
    pi.set_servo_pulsewidth(SERVO_C_PIN, 2500)
    time.sleep(0.16)
    pi.set_servo_pulsewidth(SERVO_C_PIN, 0)
    print("Chute closed")

def record_fill_amount():
    distance = get_distance()
    print("distance" + str(distance))
    if distance < 8:
        global containerFilled
        containerFilled = True
        print("recording succeeded")

try: 
    picam.start()
    frame1 = get_frame()
    while True:
        if pi.read(RESETBUTTON_PIN):
            containerFilled = False
            pi.write(REDLED_PIN, 0)
            frame1 = get_frame()
            print("reset fill")
        if containerFilled:
            pi.write(REDLED_PIN, 1)
            print("full")
        else:
            time.sleep(1000)
            if detect_object():
                pi.write(WHITELED_PIN, 1)

                trash_type = classify_object()
                pi.write(WHITELED_PIN, 0)

                if trash_type == "metal":
                    rotate_servo(120, -1)
                    open_chute()
                    time.sleep(0.7)
                    close_chute()
                    rotate_servo(180, -1)
                    record_fill_amount()
                    return_home()
                
                elif trash_type == "trash":
                    open_chute()
                    time.sleep(0.7)
                    close_chute()
                    rotate_servo(180, 1)
                    record_fill_amount()
                    return_home() 
                
                elif trash_type == "plastic":
                    rotate_servo(120, 1)
                    open_chute()
                    time.sleep(0.7)
                    close_chute()
                    rotate_servo(180, 1)
                    record_fill_amount()
                    return_home()

            frame1 = get_frame()

except KeyboardInterrupt:
    print("Exiting program...")
finally:
    pi.stop()
