import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import time

#loading model
interpreter = tflite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

#opening camera on raspberry pi
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("camera not found")
    exit()
print("Camera opened successfully")

#No AI Computer Vision Object Detection Function (Shourya's Code)
# U CAN CALL THIS FUNCTION WHEREVER U WANT, ITS NOT CURRENTLY BEING CALLED
def detectObject():

    ret1, frame1 = cap.read()
    frame1 = frame1[100:500, 100:500]
    time.sleep(0.3)
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
        count = 0
        for contour in contours:
            if cv2.contourArea(contour) >= 600: #change to 500 or 550? Or maybe 650?
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame2, (x, y), (x+w, y+h), (0, 255, 0), 2)
                count+=1
                print(count)
        cv2.imshow("cam", frame2)
        cv2.imshow("contours", cv2.drawContours(frame2Copy, contours, -1, (0, 255, 0), 2))
        

while True:
    ret, frame = cap.read()
    if not ret:
        break
    #resizing 
    img = cv2.resize(frame, (width, height))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    # running ai
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])[0]

    #prediction from teachable
    class_id = np.argmax(output)
    confidence = output[class_id]

    # chatgpt formatting
    label = f"Class {class_id} : {confidence*100:.1f}%"
    cv2.putText(frame, label, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Camera LIVE", frame)

    # exiting for testing
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
