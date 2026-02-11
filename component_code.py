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
SERVO = Servo(18)

"""You Can Use This For AI Code As Well, This Is The First Frame On Startup For Comparison with Absdiff"""
ret1, frame1 = cap.read()
frame1 = frame1[100:500, 100:500]

containerFilled = False

def get_distance():
    GPIO.output(TRIG, False)
    time.sleep(0.05)

    GPIO.output(TRIG, True)
    time.sleep(0.00001)
    GPIO.output(TRIG, False)

    while GPIO.input(ECHO) == 0:
        pulse_start = time.time()

    while GPIO.input(ECHO) == 1:
        pulse_end = time.time()

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
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame2, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.imshow("cam", frame2)
                cv2.imshow("contours", cv2.drawContours(frame2Copy, contours, -1, (0, 255, 0), 2))
                return True
        return False
    return None

def classify_object():
    # PUT AI CODE HERE
    return

def rotate_servo(degree):
    SERVO.value = degree / 180
    time.sleep(0.5)

def open_chute():
    print("Chute opened")

def close_chute():
    print("Chute closed")

def record_fill_amount(compartment):
    distance = get_distance()
    if distance < 4:
        containerFilled = True
    print("Recording fill amount for " + compartment)


try: 
    while True:
        if RESETBUTTON.is_active():
            containerFilled = False
            REDLED.off()
        if containerFilled:
            REDLED.on()
            print("bin is full")
        else:
            if detect_object():
                WHITELED.on()

                trash_type = classify_object()
                trash_type = "plastic" #test code, remove all placeholder code after testing
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
                    record_fill_amount("plastic")
                    rotate_servo(180)    
                
                elif trash_type == "aluminum":
                    rotate_servo(120)
                    open_chute()
                    time.sleep(1)
                    close_chute()
                    rotate_servo(180)
                    record_fill_amount("aluminum")
                    rotate_servo(60)
                
        time.sleep(1)

except KeyboardInterrupt:
    print("Exiting program...")
finally:
    GPIO.cleanup()
