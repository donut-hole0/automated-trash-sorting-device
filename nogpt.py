import RPi.GPIO as GPIO
import time
from gpiozero import LED, Button, Servo
import cv2
import numpy as np

GPIO.setmode(GPIO.BCM)

TRIG = 5
ECHO = 6

RESETBUTTON = Button(7) 
"""You Can Change This GPIO from 7 if you want"""

GPIO.setup(TRIG, GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)

REDLED = LED(22)
WHITELED = LED(23)
SERVO_R = Servo(18)
SERVO_C = Servo(17)

"""You Can Use This For AI Code As Well, This Is The First Frame On Startup For Comparison with Absdiff"""
cap = cv2.VideoCapture(0) #change to 1 or 2 if 0 doesnt work
ret1, frame1 = cap.read()
frame1 = frame1[100:500, 100:500]

global containerFilled
containerFilled = False

def get_distance():
    GPIO.output(TRIG, False)
    time.sleep(0.05)

    GPIO.output(TRIG, True)
    time.sleep(0.00001)
    GPIO.output(TRIG, False)

    timeout = time.time() + 0.2

    while GPIO.input(ECHO) == 0:
        pulse_start = time.time()
        if time.time() > timeout:
            return -1

    while GPIO.input(ECHO) == 1:
        pulse_end = time.time()
        if time.time() > timeout:
            return -1

    pulse_duration = pulse_end - pulse_start
    distance = pulse_duration * 17150
    distance = round(distance, 2)

    return distance

def detect_object():
    ret2, frame2 = cap.read()
    frame2 = frame2[100:500, 100:500]
    if ret1 and ret2:
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        blurredGray1 = cv2.GaussianBlur(gray1, (11, 11), 0)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        blurredGray2 = cv2.GaussianBlur(gray2, (11, 11), 0)
        diff = cv2.absdiff(blurredGray1, blurredGray2)
        _, threshedDiff = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(threshedDiff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        frame2Copy = frame2.copy()
        for contour in contours:
            if cv2.contourArea(contour) >= 600: #change to 500 or 550? Or maybe 650?
                return True
        return False
    return None

def classify_object():
    # PUT AI CODE HERE
    return

def rotate_servo(degree):
    SERVO_R.value = degree / 180
    time.sleep(0.5)

def open_chute():
    SERVO_C.value = 90 / 180
    print("Chute opened")

def close_chute():
    SERVO_C.value = -90 / 180
    print("Chute closed")

def record_fill_amount(compartment):
    distance = get_distance()
    if distance < 0:
        print("Recording Failed")
    elif distance < 8:
        global containerFilled
        containerFilled = True
        print("Recording fill amount for " + compartment)


try: 
    while True:
        if RESETBUTTON.is_active:
            containerFilled = False
            REDLED.off()
        if containerFilled:
            REDLED.on()
            print("bin is full")
        else:
            if detect_object():
                WHITELED.on()

                trash_type = classify_object()
                WHITELED.off()

                if trash_type == "plastic":
                    rotate_servo(-120)
                    open_chute()
                    time.sleep(1)
                    close_chute()
                    rotate_servo(-180)
                    record_fill_amount("plastic")
                    rotate_servo(-60)
                
                elif trash_type == "trash":
                    open_chute()
                    time.sleep(1)
                    close_chute()
                    rotate_servo(180)
                    record_fill_amount("trash")
                    rotate_servo(180)    
                
                elif trash_type == "aluminum":
                    rotate_servo(120)
                    open_chute()
                    time.sleep(1)
                    close_chute()
                    rotate_servo(180)
                    record_fill_amount("aluminum")
                    rotate_servo(60)
        ret1, frame1 = cap.read()
        frame1 = frame1[100:500, 100:500]
        time.sleep(1)

except KeyboardInterrupt:
    print("Exiting program...")
finally:
    cap.release()
    GPIO.cleanup()
