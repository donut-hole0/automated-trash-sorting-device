import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

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
