import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import time
import os
import subprocess
import psutil

# Configuration
SHOW_WINDOW = True
INFERENCE_INTERVAL = 0.1

# Get process for memory monitoring
process = psutil.Process()
peak_ram_mb = 0

def get_total_ram_usage():
    """Get RAM usage for this process and all child processes"""
    total_ram = process.memory_info().rss / 1024 / 1024  # Python script
    
    # Add all child processes (rpicam-vid, ffmpeg)
    try:
        for child in process.children(recursive=True):
            total_ram += child.memory_info().rss / 1024 / 1024
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    
    return total_ram

def get_system_ram_info():
    """Get overall system RAM usage"""
    mem = psutil.virtual_memory()
    return {
        'total': mem.total / 1024 / 1024,
        'used': mem.used / 1024 / 1024,
        'available': mem.available / 1024 / 1024,
        'percent': mem.percent
    }

# Loading labels
def load_labels(labels_path="labels.txt"):
    try:
        with open(labels_path, 'r') as f:
            return [line.strip() for line in f.readlines()]
    except FileNotFoundError:
        print("labels.txt not found, using class numbers instead")
        return None

# Improved softmax function
def softmax(x):
    if np.any(np.isnan(x)) or np.any(np.isinf(x)):
        return None
    
    x = np.clip(x, -500, 500)
    exp_x = np.exp(x - np.max(x))
    result = exp_x / exp_x.sum()
    
    if np.any(np.isnan(result)) or np.any(np.isinf(result)):
        return None
    
    return result

labels = load_labels()

# Loading model
interpreter = tflite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

# Check input type
input_dtype = input_details[0]['dtype']
print(f"Model loaded. Input size: {width}x{height}")
print(f"Input dtype: {input_dtype}")
if labels:
    print(f"Labels: {labels}")

# Check initial RAM usage
initial_script_ram = process.memory_info().rss / 1024 / 1024
sys_info = get_system_ram_info()
print(f"\n=== Initial RAM Usage ===")
print(f"Python script only: {initial_script_ram:.1f} MB")
print(f"System RAM: {sys_info['used']:.0f} MB / {sys_info['total']:.0f} MB ({sys_info['percent']:.1f}%)")

# Create a FIFO
fifo_path = "/tmp/camera_stream.h264"
if os.path.exists(fifo_path):
    os.remove(fifo_path)
os.mkfifo(fifo_path)

print("\nStarting camera with rpicam-vid...")
os.system("pkill rpicam-vid")
time.sleep(1)

# Start rpicam-vid in background
camera_cmd = f"rpicam-vid -t 0 --width 640 --height 480 --codec h264 -o {fifo_path} --inline --listen &"
os.system(camera_cmd)
time.sleep(2)

# Open FIFO with ffmpeg
ffmpeg_cmd = [
    'ffmpeg',
    '-i', fifo_path,
    '-f', 'image2pipe',
    '-pix_fmt', 'rgb24',
    '-vcodec', 'rawvideo',
    '-'
]

ffmpeg_process = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10**8)

frame_size = 640 * 480 * 3

print("Camera ready!\n")

# Check RAM after camera startup
time.sleep(1)
after_camera_ram = get_total_ram_usage()
print(f"After camera startup (script + subprocesses): {after_camera_ram:.1f} MB\n")

last_inference_time = 0
last_valid_label = "Waiting..."
frame_count = 0

try:
    while True:
        raw_frame = ffmpeg_process.stdout.read(frame_size)
        if len(raw_frame) != frame_size:
            continue
        
        frame_rgb = np.frombuffer(raw_frame, dtype=np.uint8).reshape((480, 640, 3)).copy()
        frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        
        # Get comprehensive RAM usage
        script_ram = process.memory_info().rss / 1024 / 1024
        total_ram = get_total_ram_usage()
        peak_ram_mb = max(peak_ram_mb, total_ram)
        
        frame_count += 1
        
        current_time = time.time()
        if current_time - last_inference_time >= INFERENCE_INTERVAL:
            img = cv2.resize(frame_rgb, (width, height))
            
            if input_dtype == np.uint8:
                img = img.astype(np.uint8)
            else:
                img = img.astype(np.float32) / 255.0
            
            img = np.expand_dims(img, axis=0)
            
            try:
                interpreter.set_tensor(input_details[0]['index'], img)
                interpreter.invoke()
                output = interpreter.get_tensor(output_details[0]['index'])[0]
                
                probabilities = softmax(output)
                
                if probabilities is None:
                    print(">>> Skipping frame (invalid output)")
                    last_inference_time = current_time
                    continue
                
                class_id = np.argmax(probabilities)
                confidence = probabilities[class_id]
                
                if labels:
                    label = f"{labels[class_id]}: {confidence*100:.1f}%"
                else:
                    label = f"Class {class_id}: {confidence*100:.1f}%"
                
                if confidence > 0.3:
                    last_valid_label = label
                    print(f">>> {label}")
                    print(f"    Script: {script_ram:.1f} MB | Total (w/ subprocesses): {total_ram:.1f} MB | Peak: {peak_ram_mb:.1f} MB")
                
            except Exception as e:
                print(f">>> Error during inference: {e}")
            
            last_inference_time = current_time
        
        if SHOW_WINDOW:
            cv2.putText(frame, last_valid_label, (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Show detailed RAM info
            cv2.putText(frame, f"Script: {script_ram:.0f} MB", (20, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(frame, f"Total: {total_ram:.0f} MB", (20, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(frame, f"Peak: {peak_ram_mb:.0f} MB", (20, 140),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
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
    
    # Final summary
    final_ram = get_total_ram_usage()
    sys_info = get_system_ram_info()
    print(f"\n=== RAM Usage Summary ===")
    print(f"Initial (script only): {initial_script_ram:.1f} MB")
    print(f"After camera startup: {after_camera_ram:.1f} MB")
    print(f"Final (total): {final_ram:.1f} MB")
    print(f"Peak (total): {peak_ram_mb:.1f} MB")
    print(f"\nSystem RAM usage: {sys_info['used']:.0f} MB / {sys_info['total']:.0f} MB ({sys_info['percent']:.1f}%)")
    print(f"Frames processed: {frame_count}")
    print("\nCamera stopped. Goodbye!")
