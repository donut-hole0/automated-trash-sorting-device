import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import time
import os
import subprocess
import threading

# Configuration
SHOW_WINDOW = True
INFERENCE_INTERVAL = 0.1

# labels
def load_labels(labels_path="labels.txt"):
    try:
        with open(labels_path, 'r') as f:
            return [line.strip() for line in f.readlines()]
    except FileNotFoundError:
        print("labels.txt not found, using class numbers instead")
        return None

labels = load_labels()

# model
interpreter = tflite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

print(f"Model loaded. Input size: {width}x{height}")
if labels:
    print(f"Labels: {labels}")

# chatgpt ahh
fifo_path = "/tmp/camera_stream.h264"
if os.path.exists(fifo_path):
    os.remove(fifo_path)
os.mkfifo(fifo_path)

print("Starting camera with rpicam-vid...")
os.system("pkill rpicam-vid")
time.sleep(1)

camera_cmd = f"rpicam-vid -t 0 --width 640 --height 480 --codec h264 -o {fifo_path} --inline --listen &"
os.system(camera_cmd)
time.sleep(2)

ffmpeg_cmd = [
    'ffmpeg',
    '-i', fifo_path,
    '-f', 'image2pipe',
    '-pix_fmt', 'bgr24',
    '-vcodec', 'rawvideo',
    '-'
]

ffmpeg_process = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10**8)

frame_size = 640 * 480 * 3

print("Camera ready!")

last_inference_time = 0

try:
    while True:
        raw_frame = ffmpeg_process.stdout.read(frame_size)
        if len(raw_frame) != frame_size:
            continue
        
        frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((480, 640, 3)).copy()
        
        current_time = time.time()
        if current_time - last_inference_time >= INFERENCE_INTERVAL:
            img = cv2.resize(frame, (width, height))
            img = img.astype(np.uint8)
            img = np.expand_dims(img, axis=0)
            
            interpreter.set_tensor(input_details[0]['index'], img)
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]['index'])[0]
            
            class_id = np.argmax(output)
            confidence = output[class_id]
            
            if labels:
                label = f"{labels[class_id]}: {confidence*100:.1f}%"
            else:
                label = f"Class {class_id}: {confidence*100:.1f}%"
            
            print(f">>> {label}")
            last_inference_time = current_time
        
        if SHOW_WINDOW:
            if 'label' in locals():
                cv2.putText(frame, label, (20, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Camera LIVE", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

except KeyboardInterrupt:
    print("\nStopping...")

finally:
    ffmpeg_process.terminate()
    os.system("pkill rpicam-vid")
    if os.path.exists(fifo_path):
        os.remove(fifo_path)
    if SHOW_WINDOW:
        cv2.destroyAllWindows()
    print("Camera stopped. Goodbye!")
