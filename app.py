from flask import Flask, Response, request
import cv2
from ultralytics import YOLO
import supervision as sv
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Load the YOLOv8 model
model = YOLO('weights/last.pt')

# Open the default camera (index 0)
cap = cv2.VideoCapture(0)
streaming = False

def gen_frames():
    global streaming
    streaming = True
    while streaming:
        ret, frame = cap.read()
        if not ret:
            break

        # Apply YOLOv8 object detection
        results = model(frame)[0]
        detections = sv.Detections.from_ultralytics(results)
        bounding_box_annotator = sv.BoxAnnotator()
        annotated_image = bounding_box_annotator.annotate(scene=frame, detections=detections)

        # Add labels to the annotated image
        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, conf, cls = result[:6]
            label = model.names[int(cls)]
            cv2.putText(annotated_image, label, (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)

        # Encode the output frame as a JPEG image
        ret, jpeg = cv2.imencode('.jpg', annotated_image)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_video_feed', methods=['POST'])
def stop_video_feed():
    global streaming
    streaming = False
    return {"message": "Video feed stopped successfully"}

# Run the app with Render's specified port
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Use PORT environment variable or default to 5000
    app.run(host="0.0.0.0", port=port)
