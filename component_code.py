import RPi. GPIO as GPIO
import time
from gpiozero import LED, Button, Servo

GPIO.setmode(GPIO.BCM)

TRIG = 5
ECHO = 6
GPIO.setup(TRIG, GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)

redLED = LED(22)
whiteLED = LED(23)
servo = Servo(18)

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

def is_full():
    distance = get_distance()
    return distance < 4

def detect_object():
    // need to implement object detection logic later
    // also need to put camera ai code here as well
    return True

def rotate_servo(degree):
    servo.value = degree / 180
    time.sleep(0.5)

def open_chute():
    print("Chute opened")

def close_chute():
    print("Chute closed")

def indicate_full():
    redLED.on()
    whiteLED.off()
    print("Bin is full")

def clear_indication():
    redLED.off()
    whiteLED.on()
    print("Bin has space")

def cv_on():
    whiteLED.on()

def cv_off():
    whiteLED.off()

def record_fill_amount(bin_name):
    print(f"Recording fill amount for {bin_name}")


try: 
    while True:
        if not is_full():
            indicate_full()
            time.sleep(2)
            continue
        else:
            clear_indication()
        
        if detect_object():
            cv_on()

            //ai model output goes here
            // example trash_type
            trash_type = "plastic"
            cv_off()

            if trash_type == "plastic":
                rotate_servo(-60)
                open_chute()
                time.sleep(2)
                close_chute()
                rotate_servo(120)
                rotate_servo(-60)
                record_fill_amount("plastic")
            
            elif trash_type == "aluminum":
                rotate_servo(180)
                open_chute()
                time.sleep(2)
                close_chute()
                rotate_servo(-120)
                rotate_servo(120)
                record_fill_amount("aluminum")
            
            elif trash_type == "trash":
                rotate_servo(60)
                open_chute()
                time.sleep(2)
                close_chute()
                rotate_servo(120)
                rotate_servo(-60)
                record_fill_amount("trash")
            
        time.sleep(2)

except KeyboardInterrupt:
    print("Exiting program...")
finally:
    GPIO.cleanup()

