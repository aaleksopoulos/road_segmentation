#setting up the imports
import cv2, time, os
import numpy as np
from openvino.runtime import Core
from PIL import Image
import ov_utils as f

from ultralytics import YOLO

def process_frame(frame_orig, compiled_model, model, input_layer, output_layer, colormap):

    frame = f.preprocess(frame_orig, input_layer)
    mask = f.inference(compiled_model, frame, output_layer)

    output_frame, output_frame_masked = f.output(mask, frame_orig, colormap, alpha)

    #yolo time
    # Run YOLOv8 tracking on the frame, persisting tracks between frames
    yolo_results = model.track(frame_orig, conf=0.3, iou=0.2, show=False)
    # yolo_results = model.track(frame_orig)

    # print(yolo_results[0][1])

    # print(results[0].boxes.data)
    for i in range(len(yolo_results[0])):

        # get the coordinates of the bounding box
        x1 = int(yolo_results[0][i].boxes.data[0][0].item())
        y1 = int(yolo_results[0][i].boxes.data[0][1].item())
        x4 = int(yolo_results[0][i].boxes.data[0][2].item())
        y4 = int(yolo_results[0][i].boxes.data[0][3].item())

        # print(f'x1, y1 = {x1, y1}') #top left
        # print(f'x2, y2 = {x4, y1}') #top right
        # print(f'x3, y3 = {x1, y4}') #bottom left
        # print(f'x4, y4 = {x4, y4}') #bottom right

        # Visualize the results on the frame
        annotated_frame = yolo_results[0][i].plot()

        # Display the annotated frame
        cv2.imshow("YOLOv8_Tracking.jpg", annotated_frame)
        # print('colormap : ',colormap[1])
        # wait until any key is pressed
        cv2.waitKey()

        # print(output_frame[y1][x1]) #H, W
        # print(output_frame[y1][x4]) #H, W
        # print(output_frame[y4][x1]) #H, W
        # print(output_frame[y4][x4]) #H, W
        # print(frame_orig.shape)
        count = 0
        if np.array_equal(output_frame[y1][x1], colormap[1]):
            count+=1

        if np.array_equal(output_frame[y1][x4], colormap[1]):
            count+=1

        if np.array_equal(output_frame[y4][x1], colormap[1]):
            count+=1

        if np.array_equal(output_frame[y4][x4], colormap[1]):
            count+=1

        # print('count : ', count)
        if count>=2:
            cv2.drawMarker(output_frame_masked, (x1, y1),(0,0,255), markerType=cv2.MARKER_STAR, markerSize=5, thickness=1, line_type=cv2.LINE_AA)
            cv2.drawMarker(output_frame_masked, (x4, y1),(0,0,255), markerType=cv2.MARKER_STAR, markerSize=5, thickness=1, line_type=cv2.LINE_AA)
            cv2.drawMarker(output_frame_masked, (x1, y4),(0,0,255), markerType=cv2.MARKER_STAR, markerSize=5, thickness=1, line_type=cv2.LINE_AA)
            cv2.drawMarker(output_frame_masked, (x4, y4),(0,0,255), markerType=cv2.MARKER_STAR, markerSize=5, thickness=1, line_type=cv2.LINE_AA)

    cv2.imshow('output', output_frame_masked)

    # wait until any key is pressed
    cv2.waitKey()

# Load the YOLOv8 model
detection_model = YOLO('yolov8n.pt')

#initiate openVino runtime
core = Core()

#set the model we are gonna use    
model_path = '/intel/road-segmentation-adas-0001/FP32/road-segmentation-adas-0001'
pwd = os.getcwd()

#set up the colormap, we only need to show the road 
colormap = np.array([[0, 0, 0], [48, 103, 141], [0, 0, 0], [0, 0, 0]])
alpha = 0.5

#get the device
selected_device = f.get_devices(core)

#instantiate the model
ir_model = core.read_model(model=pwd+model_path+'.xml', weights=pwd+model_path+'.bin')
compiled_model = core.compile_model(model=ir_model, device_name=selected_device)

#get some basic info about the model
input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)

print(f"Input layer shape: {input_layer.shape}")
print(f"Output layer shape: {output_layer.shape}")

frame_orig = cv2.imread('img1.jpg')

process_frame(frame_orig, compiled_model, detection_model, input_layer, output_layer, colormap)

# release resources
cv2.destroyAllWindows()