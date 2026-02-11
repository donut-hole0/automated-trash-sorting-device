import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import time
import os
import subprocess
import psutil

# ------------------ CONFIG ------------------
SHOW_WINDOW = True        # Set False for headless
INFERENCE_INTERVAL = 0.1  # Seconds between AI inferences
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
BACKGROUND_BLUR = (11, 11)
DIFF_THRESHOLD = 30
MIN_CONTOUR_AREA = 500
# --------------------------------------------

# RAM monitoring
process = psutil.Process()
peak_ram_mb = 0

def get_total_ram_usage():
    total_ram = process.memory_info().rss / 1024 / 1024
    try:
        for child in process.children(recursive=True):
            total_ram += child.memory_info().rss / 1024 / 1024
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    return total_ram

def get_system_ram_info():
    mem = psutil.virtual_memory()
    return {
        'total': mem.total / 1024 / 1024,
        'used': mem.used / 1024 / 1024,
        'available': mem.available / 1024 / 1024,
        'percent': mem.percent
    }

# Load labels
def load_labels(labels_path="labels.txt"):
    try:
        with open(labels_path, 'r') as f:
            return [line.strip() for line in f.readlines()]
    except FileNotFoundError:
        print("labels.txt not found, using class numbers instead")
        return None

# Softmax
def softmax(x):
    x = np.clip(x, -500, 500)
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()

labels = load_labels()

# Load TFLite model
interpreter = tflite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]
input_dtype = input_details[0]['dtype']

print(f"Model loaded: {width}x{height}, dtype={input_dtype}")
if labels:
    print(f"Labels: {labels}")

# RAM info
initial_script_ram = process.memory_info().rss / 1024 / 1024
sys_info = get_system_ram_info()
print(f"Initial RAM: {initial_script_ram:.1f} MB, System: {sys_info['percent']:.1f}% used")

# ------------------ CAMERA SETUP ------------------
fifo_path = "/tmp/camera_stream.h264"
if os.path.exists(fifo_path):
    os.remove(fifo_path)
os.mkfifo(fifo_path)

print("Starting rpicam-vid...")
os.system("pkill rpicam-vid")
time.sleep(1)
camera_cmd = f"rpicam-vid -t 0 --width {FRAME_WIDTH} --height {FRAME_HEIGHT} --codec h264 -o {fifo_path} --inline --listen &"
os.system(camera_cmd)
time.sleep(2)

ffmpeg_cmd = [
    'ffmpeg',
    '-i', fifo_path,
    '-f', 'image2pipe',
    '-pix_fmt', 'rgb24',
    '-vcodec', 'rawvideo',
    '-'
]

ffmpeg_process = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10**8)
frame_size = FRAME_WIDTH * FRAME_HEIGHT * 3

# ------------------ CAPTURE BACKGROUND ------------------
print("Capturing background frame...")
raw_bg = ffmpeg_process.stdout.read(frame_size)
bg_frame = np.frombuffer(raw_bg, dtype=np.uint8).reshape((FRAME_HEIGHT, FRAME_WIDTH, 3)).copy()
bg_gray = cv2.cvtColor(bg_frame, cv2.COLOR_RGB2GRAY)
bg_gray = cv2.GaussianBlur(bg_gray, BACKGROUND_BLUR, 0)
print("Background captured.")

# ------------------ MAIN LOOP ------------------
last_inference_time = 0
last_valid_label = "Waiting..."
frame_count = 0

try:
    while True:
        raw_frame = ffmpeg_process.stdout.read(frame_size)
        if len(raw_frame) != frame_size:
            continue

        frame_rgb = np.frombuffer(raw_frame, dtype=np.uint8).reshape((FRAME_HEIGHT, FRAME_WIDTH, 3)).copy()
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        # RAM usage
        script_ram = process.memory_info().rss / 1024 / 1024
        total_ram = get_total_ram_usage()
        peak_ram_mb = max(peak_ram_mb, total_ram)
        frame_count += 1

        current_time = time.time()
        if current_time - last_inference_time >= INFERENCE_INTERVAL:
            # --- Background subtraction ---
            frame_gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
            frame_gray = cv2.GaussianBlur(frame_gray, BACKGROUND_BLUR, 0)
            diff = cv2.absdiff(frame_gray, bg_gray)
            _, mask = cv2.threshold(diff, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Crop to largest object if present
            if contours:
                largest = max(contours, key=cv2.contourArea)
                if cv2.contourArea(largest) > MIN_CONTOUR_AREA:
                    x, y, w, h = cv2.boundingRect(largest)
                    obj_frame = frame_rgb[y:y+h, x:x+w]
                else:
                    obj_frame = frame_rgb
            else:
                obj_frame = frame_rgb

            # Resize for model
            img = cv2.resize(obj_frame, (width, height))
            if input_dtype != np.uint8:
                img = img.astype(np.float32) / 255.0
            img = np.expand_dims(img, axis=0)

            # --- TFLite inference ---
            try:
                interpreter.set_tensor(input_details[0]['index'], img)
                interpreter.invoke()
                output = interpreter.get_tensor(output_details[0]['index'])[0]
                probabilities = softmax(output)

                if probabilities is not None:
                    class_id = np.argmax(probabilities)
                    confidence = probabilities[class_id]
                    if labels:
                        label = f"{labels[class_id]}: {confidence*100:.1f}%"
                    else:
                        label = f"Class {class_id}: {confidence*100:.1f}%"

                    if confidence > 0.3:
                        last_valid_label = label
                        print(f">>> {label} | RAM: {script_ram:.1f} MB | Total: {total_ram:.1f} MB | Peak: {peak_ram_mb:.1f} MB")
            except Exception as e:
                print(f">>> Error during inference: {e}")

            last_inference_time = current_time

        # Optional display
        if SHOW_WINDOW:
            cv2.putText(frame_bgr, last_valid_label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Camera LIVE", frame_bgr)
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
    
    final_ram = get_total_ram_usage()
    sys_info = get_system_ram_info()
    print(f"\n=== RAM Usage Summary ===")
    print(f"Initial (script only): {initial_script_ram:.1f} MB")
    print(f"Final (total): {final_ram:.1f} MB | Peak: {peak_ram_mb:.1f} MB")
    print(f"Frames processed: {frame_count}")
    print("\nCamera stopped. Goodbye!")
