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
        frame = f.preprocess(frame_orig, input_layer)
        mask = f.inference(compiled_model, frame, output_layer)
        output_frame, output_frame_masked = f.output(mask, frame_orig, colormap, alpha)

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