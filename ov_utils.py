import cv2, os
import numpy as np

#only download if file does not exist
if not os.path.isfile('notebook_utils.py'):
    import urllib.request
    urllib.request.urlretrieve(
        url='https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/main/notebooks/utils/notebook_utils.py',
        filename='notebook_utils.py'
    )
    print('downloaded notebook_utils')
else:
    print('file notebook_utils was found')

from notebook_utils import segmentation_map_to_image

def get_devices(core):
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
    return selected_device

def preprocess(frame, input_layer):
    '''
    Preprocesses an image so it coult be used with the current selected model.
    Parameters : An image or a frame from a video, and the associated input layer
    Returns: the preprocessed image
    '''
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    N, C, H, W = input_layer.shape
    frame = cv2.resize(frame, (W, H))
    frame = np.expand_dims(frame.transpose(2,0,1),0)
    
    return frame

def inference(model, frame, output_layer):
    '''
    Performs inference on an image
    Parameters:
        model: The loaded model
        frame : An image or a frame from a video
        output_layer : The associated output layer
    Returns:
        The segmentation mask of the image
    '''
    result = model(frame)[output_layer]
    mask = np.argmax(result, axis=1)
    return mask

def output(mask, frame_orig, colormap, alpha):
    '''
    Outputs the inference result on an image
    Parameters:
        mask: The segmentation mask of the image (taken from inference)
        frame_orig : The original (unprocessed) image or a frame from a video
        colormap : The associated colormap for the segmentation
        alpha : The alpha value for the segmentation
    Returns:
        origanal_img_class : The class of the pixels of the original image
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