The datasets used for training the models are as follows:
for training the road_model.pth: https://www.kaggle.com/datasets/carlolepelaars/camvid
for training the mobilenetv3s.pt on GuideDog dataset: https://huggingface.co/datasets/kjunh/GuideDog

Initially, the project focused on geometric surface analysis—fitting a ground plane and walls to detect obstacles as 'deformations' in the user's path. While this provided a strong theoretical foundation, we transitioned to a deep-learning approach using MobileNetV3-Small trained on the GuideDog dataset. This transition allowed for more robust classification in the central walking region, improving real-world performance over the initial geometry-based model.
