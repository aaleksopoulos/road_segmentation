#setting up the imports
import cv2, time, os
import numpy as np
from openvino.runtime import Core
from PIL import Image
import ov_utils as f

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

frame_orig = cv2.imread('img2.jpg')

#if indeed there is a frame
frame = f.preprocess(frame_orig, input_layer)
mask = f.inference(compiled_model, frame, output_layer)

output_frame, output_frame_masked = f.output(mask, frame_orig, colormap, alpha)
print(output_frame[209][316]) #H, W
print(output_frame[209][407]) #H, W
print(output_frame[358][316]) #H, W
print(output_frame[358][407]) #H, W
print(frame_orig.shape)
cv2.drawMarker(output_frame_masked, (316, 209),(0,0,255), markerType=cv2.MARKER_STAR, markerSize=5, thickness=1, line_type=cv2.LINE_AA)
cv2.drawMarker(output_frame_masked, (316, 358),(0,0,255), markerType=cv2.MARKER_STAR, markerSize=5, thickness=1, line_type=cv2.LINE_AA)
cv2.drawMarker(output_frame_masked, (407, 209),(0,0,255), markerType=cv2.MARKER_STAR, markerSize=5, thickness=1, line_type=cv2.LINE_AA)
cv2.drawMarker(output_frame_masked, (407, 358),(0,0,255), markerType=cv2.MARKER_STAR, markerSize=5, thickness=1, line_type=cv2.LINE_AA)

cv2.imshow('output', output_frame_masked)

# wait until any key is pressed
cv2.waitKey()

# release resources
cv2.destroyAllWindows()