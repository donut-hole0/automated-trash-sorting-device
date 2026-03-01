from picamera2 import Picamera2
import cv2

picam = Picamera2()
config = picam.create_preview_configuration(
    main={"format": "RGB888", "size": (1280, 720)}
)
picam.configure(config)
picam.start()
try:
    while True:
        test_frame = picam.capture_array()
        cv2.flip(test_frame, -1)
        cv2.imshow("full_frame.jpg", test_frame)
        print(f"Frame shape: {test_frame.shape}")
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Exiting program...")
