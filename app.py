import cv2
import numpy as np
from flask import Flask, Response, render_template
import time
import torch  # PyTorch for YOLOv5
from torchvision import transforms

app = Flask(_name_)

# IP Camera Stream (Ensure this URL works in a browser)
IP_CAMERA_URL = "http://10.10.212.96:8080/shot.jpg"

# Load YOLOv5 Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.eval()

# Define Image Transformations
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((640, 640)),  
    transforms.ToTensor()
])

def get_frame():
    """Fetch a frame from the IP camera."""
    cap = cv2.VideoCapture(IP_CAMERA_URL)
    ret, frame = cap.read()
    cap.release()
    
    if not ret or frame is None:
        print("⚠ Failed to capture frame from camera.")
        return None, None

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return gray, frame

def detect_objects(frame):
    """Detect objects in the frame using YOLOv5."""
    try:
        image = transform(frame).unsqueeze(0)  # Preprocess image
        results = model(image)  # Perform inference
        detections = results.xyxy[0].cpu().numpy()  # Extract bounding boxes

        objects = []
        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            objects.append({
                "box": (int(x1), int(y1), int(x2), int(y2)), 
                "label": model.names[int(cls)], 
                "conf": float(conf)
            })

        return objects
    except Exception as e:
        print(f"❌ Object detection error: {e}")
        return []

def detect_motion():
    """Generate video stream with motion and object detection."""
    prev_frame, _ = get_frame()
    if prev_frame is None:
        print("⚠ No initial frame. Motion detection disabled.")
        return  # Exit if no frame is available

    time.sleep(0.3)  # Small delay before processing

    while True:
        cur_frame, color_frame = get_frame()
        if cur_frame is None or color_frame is None:
            print("⚠ Skipping frame due to capture failure.")
            time.sleep(0.3)
            continue

        # Compute motion detection
        frame_diff = cv2.absdiff(prev_frame, cur_frame)
        _, thresh = cv2.threshold(frame_diff, 40, 255, cv2.THRESH_BINARY)
        motion_pixels = np.sum(thresh > 0)
        motion_threshold = cur_frame.size * 0.02  # 2% of pixels changed

        if motion_pixels > motion_threshold:
            cv2.putText(color_frame, "🚨 Motion Detected!", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Object detection
        objects = detect_objects(color_frame)
        for obj in objects:
            x1, y1, x2, y2 = obj["box"]
            label = f"{obj['label']} ({obj['conf']:.2f})"
            cv2.rectangle(color_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(color_frame, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        prev_frame = cur_frame  # Update previous frame

        # Encode frame for streaming
        ret, buffer = cv2.imencode('.jpg', color_frame)
        if not ret:
            print("⚠ Failed to encode frame.")
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               buffer.tobytes() + b'\r\n')

@app.route('/')
def index():
    """Render the homepage."""
    return render_template("index.html")

@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(detect_motion(), mimetype='multipart/x-mixed-replace; boundary=frame')

if _name_ == '_main_':
    app.run(host='0.0.0.0', port=5000, debug=True)


