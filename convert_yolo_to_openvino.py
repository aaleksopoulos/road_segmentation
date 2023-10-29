from ultralytics import YOLO

# Load the YOLOv8 model
detection_model = YOLO('yolov8n.pt')

#export the yolo model to openvino
detection_model.export(format='openvino')