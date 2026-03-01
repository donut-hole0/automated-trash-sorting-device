from picamera2 import Picamera2
import cv2

picam = Picamera2()
config = picam.create_preview_configuration(
    main={"format": "RGB888", "size": (1280, 720)}
)
picam.configure(config)

test_frame = picam.capture_array()
print(f"Frame shape: {test_frame.shape}")
while True:
  cv2.imshow("full_frame.jpg", test_frame)

except KeyboardInterrupt:
    print("Exiting program...")
