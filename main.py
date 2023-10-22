#setting up the imports
import cv2
import numpy as np
import time
from openvino.runtime import Core
import torch
import os
from PIL import Image

#only download if file does not exist
if not os.path.isfile('notebook_utils.py'):
    import urllib.request
    urllib.request.urlretrieve(
        url='https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/main/notebooks/utils/notebook_utils.py',
        filename='notebook_utils.py'
    )

from notebook_utils import segmentation_map_to_image

#initiate openVino runtim
core = Core()

#check the available devices
devices = core.available_devices
device_names = []

for device in devices:
    device_name = core.get_property(device, "FULL_DEVICE_NAME")
    device_names.append(device)
    print(f"{device}: {device_name}")

#use CPU unless a GPU is available
selected_device = 'CPU'
# if ('GPU' in device_names): #my GPU sucks, will stick to the CPU
#     selected_device = 'GPU'
    
print(f'selected_device is: {selected_device}')
    
#set the model we are gonna use    
model_path = '/intel/road-segmentation-adas-0001/FP32/road-segmentation-adas-0001'
pwd = os.getcwd()

#set up the colormap, we only need to show the road 
colormap = np.array([[0, 0, 0], [48, 103, 141], [0, 0, 0], [0, 0, 0]])
alpha = 0.5

#instantiate the model
ir_model = core.read_model(model=pwd+model_path+'.xml', weights=pwd+model_path+'.bin')
compiled_model = core.compile_model(model=ir_model, device_name=selected_device)

#get some basic info about the model
input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)

print(f"Input layer shape: {input_layer.shape}")
print(f"Output layer shape: {output_layer.shape}")

def preprocess(frame):
    '''
    Preprocesses an image so it coult be used with the current selected model.
    Parameters : An image or a frame from a video
    Returns: the preprocessed image
    '''
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    N, C, H, W = input_layer.shape
    frame = cv2.resize(frame, (W, H))
    frame = np.expand_dims(frame.transpose(2,0,1),0)
    
    return frame

def inference(model, frame):
    '''
    Performs inference on an image
    Parameters:
        model: The loaded model
        frame : An image or a frame from a video
    Returns:
        The segmentation mask of the image
    '''
    result = model(frame)[output_layer]
    mask = np.argmax(result, axis=1)

    return mask

def output(mask, frame_orig):
    '''
    Outputs the inference result on an image
    Parameters:
        mask: The segmentation mask of the image (taken from inference)
        frame_orig : The original (unprocessed) image or a frame from a video
    Returns:
        converted_mask : The mask converted to an RGB image
        image_with_mask : The original image with the converted mask
    '''
    H, W, _ = frame_orig.shape
    #use the imported function to convert the mask to an RGB image
    converted_mask = segmentation_map_to_image(mask, colormap)
    converted_mask = cv2.resize(converted_mask, (W, H))
    #add the mask to the original image
    image_with_mask = cv2.addWeighted(converted_mask, alpha, frame_orig, 1 - alpha, 0)
    return converted_mask, image_with_mask

#get the input video
cap = cv2.VideoCapture('./input.mp4')
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_fps = int(cap.get(cv2.CAP_PROP_FPS))
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 300)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 300)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' codec for MP4 format
out = cv2.VideoWriter('output_video.mp4', fourcc, 10.0, (frame_width, frame_height))  # Output file will be in MP4 format

#to count the fps
frame_count = 0
start_time = time.time()

while True:
    ret, frame_orig = cap.read()
    if ret:

        #if indeed there is a frame
        frame = preprocess(frame_orig)
        mask = inference(compiled_model, frame)
        output_frame, output_frame_masked = output(mask, frame_orig)

        #counting the fps
        frame_count += 1
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time    

        #update the output image
        cv2.putText(output_frame_masked, f'FPS: {fps:.2f}', (frame_width - 125, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (38, 0, 255), 1, cv2.LINE_AA)
        cv2.imshow('output', output_frame_masked)
        out.write(output_frame_masked)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
    else:
        break

#clean up the mess
cap.release()
out.release()
cv2.destroyAllWindows()