# Libraries
from ultralytics import YOLO
import cv2
import time

# Capture the model's weights
model = YOLO('best.pt')

# Use the YOLO model to detect facial expressions
use_model = True

# Start capturing video
cap = cv2.VideoCapture(0)

# Record video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640,  480))

# Time before video starts
prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break
    
    # Track the amount of time it takes to perform inference on the 
    # captured frame if use_model is true, otherwise latency is 0
    latency_start = time.time()
    if use_model == True:
        results = model(frame, conf=0.4)
        result = results[0]
    latency = (time.time() - latency_start) * 1000
    
    # Track the fps of the video capture
    updated_time = time.time()
    fps = 1 / (updated_time - prev_time)
    prev_time = updated_time

    # Plot results
    try:
        if use_model == True:
            frame = result.plot()
    except AttributeError:
        print("Error: plot() method not available for results.")
        break
    
    # FPS and latency overlay
    cv2.putText(frame, f"FPS: {int(fps)}", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {int(fps)}", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Latency: {latency:.2f} ms", (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, f"Latency: {latency:.2f} ms", (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    # Save the video recording
    out.write(frame)

    # Display the output
    if use_model == True:
        cv2.imshow('Camera Display with YOLO', frame)
    else:
        cv2.imshow('Camera Display without YOLO', frame)
    
    # Stop video
    if cv2.waitKey(1) == 27:  # ESC key to exit
        break

cap.release()
cv2.destroyAllWindows()
