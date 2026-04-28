The datasets used for training the models are as follows:
for training the road_model.pth: https://www.kaggle.com/datasets/carlolepelaars/camvid
for training the mobilenetv3s.pt on GuideDog dataset: https://huggingface.co/datasets/kjunh/GuideDog

Initially, the project focused on geometric surface analysis—fitting a ground plane and walls to detect obstacles as 'deformations' in the user's path. While this provided a strong theoretical foundation, we transitioned to a deep-learning approach using MobileNetV3-Small trained on the GuideDog dataset. This transition allowed for more robust classification in the central walking region, improving real-world performance over the initial geometry-based model.

To test the model, please run  
"test_with_depth_original.py"

Choose 1 or 2 (description will be given in the terminal, what 1 does and what 2 does)

You can change the file name of the video you want to run the code on. 
Since I was running the code in Ubuntu wsl2, i couldn't access the laptop's camera. So instead I downloaded an app IPWebcam, that will take input from my mobile phone. You just need to type in the IP Address mentioned in the app after you click on start server button in the app. 


------------------------------------------------------------------------------------------------------------------------------------------------------------------

terrain_nav-dir.py  |  This is what we tried to implement previously but failed, hence this part is not included in the report. But the trained model and the source code are given just for reference in legacy_road_model

This code is essentially a navigation assistant for visually impaired people — it watches a live camera feed and tells you how to move, in plain language, in real time. It figures out what's around you, how far things are, and what kind of ground you're walking on, then distills all of that into simple commands like "Go straight", "Move left", "STOP" etc

To understand the environment, it runs three models at once. MiDaS looks at each camera frame and estimates how far away every surface is. A custom UNet model figures out which parts of the image are actually walkable ground — this "road mask" is what keeps everything else focused on the path ahead rather than wasting effort on walls or the sky. YOLOv8 handles object detection, spotting people, vehicles, and other things moving around in the scene. All three are run lean and fast — skipping frames where possible and using half-precision arithmetic on the GPU to keep up with real-time video.

Detected objects are tracked across frames using a simple overlap-based matcher, so the system knows when the same person or vehicle persists from one frame to the next. These tracked regions are then marked off as forbidden zones, making sure the obstacle detector doesn't confuse a passing pedestrian for something fixed in the ground.

Stair detection is where it gets clever. Rather than relying on a single signal, it looks for four things at once — whether horizontal edges in the depth map are evenly spaced like stair steps would be, whether depth values jump repeatedly as you scan downward, how strong the overall vertical gradient is, and how many horizontal lines appear in the scene. These four pieces of evidence are blended into a confidence score, and stairs are only announced if that confidence stays high across at least three frames in a row, with the up or down direction settled by majority vote over the last five.

On flat ground, the system works out what the floor normally looks like and flags anything sticking up above it. It splits the walkable path into left, center, and right thirds and measures how blocked each zone is. From there, the direction logic is straightforward — too much in the center means stop, one side clearer than the other means go that way, both sides open means go straight.

On top of all this, a separate edge detector runs on the raw camera image to catch things the depth map might miss, like thin poles or glass. Everything feeds into a three-state terrain classifier — stairs, flat, or irregular — that switches between modes carefully to avoid flip-flopping. Warnings are capped to once every two seconds so they don't become noise, and the whole annotated output is saved as a video you can review later.

OUTPUT of the previous model----------------------------------------------------------------------------
Due to file size limits, the initial road model output video is hosted externally:
https://drive.google.com/file/d/1XPANDxh6mvcevBYPPjB8qex8wZFtgpMY/view?usp=drive_link
