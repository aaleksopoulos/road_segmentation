## Performing road segmentation on video footage

### Overview

This project was originally started to be my submission in the [Chip's Challenge](https://events.hackster.io/chips-challenge?utm_source=email&mc_cid=6e64f26212&mc_eid=7a4cf17b2b) but due to lack of time did not manage to submit it on time [too bad... would like to had a chance to win the NUC... anyway...]

The main concept is to use the [OpenVino](https://docs.openvino.ai/) framework to try to identify which parts of an input video contain road pavement. For that purpose a model from the OpenVino ModelZoo was used, the [road-segmentation-adas-0001](https://docs.openvino.ai/2023.1/omz_models_model_road_segmentation_adas_0001.html). This model classifies each pixel in four classes: BG, road, curb, mark.

The first step, after installing the openvino (based on the instruction that are provided in their webpage), is to download the model. This can be achieved using the following command  
```omz_downloader --name road-segmentation-adas-0001 ```  
which will download the model, in the OpenVino Intermediate Representation format.  

The code for performing the segmentation lies in the [main.py](./main.py) file, which is kinda self explanatory. It reads the file (should be named as ```input.mp4```) and outputs the result in a CV2 window, as well as in the ```output_video.mp4``` file.

### Future Work

There are a lot of improvements that could be done to that code. First - and more obvious - is to be able for the user to specify the file on which inference should be performed. 

Part from that, a useful option, would be to be able to perform also classification, in order to be able to track objects on the road.  