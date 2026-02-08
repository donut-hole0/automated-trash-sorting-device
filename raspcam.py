import time
import numpy as np
from picamera2 import Picamera2
from PIL import Image
import tflite_runtime.interpreter as tflite

MODEL_PATH = "model.tflite"  
LABELS_PATH = "labels.txt" 
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
INFERENCE_INTERVAL = 0.5 

class TeachableMachineClassifier:
    def __init__(self, model_path, labels_path):
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        self.input_shape = self.input_details[0]['shape']
        self.input_height = self.input_shape[1]
        self.input_width = self.input_shape[2]
        
        with open(labels_path, 'r') as f:
            self.labels = [line.strip() for line in f.readlines()]
        
        print(f"Model loaded. Input size: {self.input_width}x{self.input_height}")
        print(f"Labels: {self.labels}")
    
    def preprocess_image(self, image):
        """Prepare image for model input"""
        image = image.resize((self.input_width, self.input_height))
        
        input_data = np.array(image, dtype=np.float32)
        input_data = np.expand_dims(input_data, axis=0)

        input_data = input_data / 255.0
        
        return input_data
    
    def classify(self, image):
        """Run inference on an image"""
        input_data = self.preprocess_image(image)
        
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        results = output_data[0]
        
        predictions = [(self.labels[i], float(results[i])) for i in range(len(self.labels))]
        
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        return predictions

def main():
    print("Initializing Teachable Machine Classifier...")
    
    classifier = TeachableMachineClassifier(MODEL_PATH, LABELS_PATH)
    
    print("Initializing camera...")
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": (CAMERA_WIDTH, CAMERA_HEIGHT), "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()
    
    time.sleep(2)
    print("Camera ready!")
    print("\nStarting classification (Press Ctrl+C to stop)...\n")
    
    try:
        while True:
            frame = picam2.capture_array()
            
            image = Image.fromarray(frame)
            
            predictions = classifier.classify(image)
            
            print("\n" + "="*50)
            print(f"Classification results:")
            for label, confidence in predictions:
                print(f"  {label}: {confidence*100:.2f}%")
            
            top_label, top_confidence = predictions[0]
            print(f"\n>>> DETECTED: {top_label} ({top_confidence*100:.1f}%)")
            
            time.sleep(INFERENCE_INTERVAL)
            
    except KeyboardInterrupt:
        print("\n\nStopping...")
    finally:
        picam2.stop()
        print("Camera stopped. Goodbye!")

if __name__ == "__main__":
    main()
