The datasets used for training the models are as follows:
for training the road_model.pth: https://www.kaggle.com/datasets/carlolepelaars/camvid
for training the mobilenetv3s.pt on GuideDog dataset: https://huggingface.co/datasets/kjunh/GuideDog

Initially, the project focused on geometric surface analysis—fitting a ground plane and walls to detect obstacles as 'deformations' in the user's path. While this provided a strong theoretical foundation, we transitioned to a deep-learning approach using MobileNetV3-Small trained on the GuideDog dataset. This transition allowed for more robust classification in the central walking region, improving real-world performance over the initial geometry-based model.

To test the model, please run  
"test_with_depth_original.py"

Choose 1 or 2 (description will be given in the terminal, what 1 does and what 2 does)

You can change the file name of the video you want to run the code on
Since I was running the code in Ubuntu wsl2, i couldn't access the laptop's camera. So instead I downloaded an app IPWebcam, that will take input from my mobile phone. You just need to type in the IP Address mentioned in the app after you click on start server button in the app.
