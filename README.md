# objectDetectionWithYOLO

In the yolov3.py we used OpenCv's line detection algorithm and DarkNet's pre-trained YOLO model. We have an object recognition and line detection application. You need to know something before work on this codes;
- There is both screen capture and webcam image codes in this project. Screen capture for testing different videos in your desktop easily. Webcam images for implementation real world problems, it is your choice.
- As you can see some videos have around 20 FPS but others have just 4 FPS the reason why the neural network model has different weights and we use two of them. If you use YOLOv3-tiny weights, you will get 20 FPS but this weights has restricted recognition ability. If you use YOLOv3-spp weights, you will get 4 FPS and more recognition ability, choose what you need.
- You can download weights, cfg and names files in DarkNet's official website. Here is the [link](https://pjreddie.com/darknet/yolo/)

yolov5.py file added later. This script contains yolov5 object detection model and how to use it in real time object detection problems. It's more accurate and fast compared to yolov3 but I assume you are using a GPU.

Photos are taken yolov3.
![od1](https://user-images.githubusercontent.com/79279694/109396292-da770380-7941-11eb-896c-bf71b4ce68a8.png)
![od2](https://user-images.githubusercontent.com/79279694/109396311-ed89d380-7941-11eb-8a3b-ab0340067675.png)
![od3](https://user-images.githubusercontent.com/79279694/109396313-ee226a00-7941-11eb-8b3c-45e484612d03.png)




