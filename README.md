The datasets used for training the models are as follows:
for training the road_model.pth: https://www.kaggle.com/datasets/carlolepelaars/camvid
for training the mobilenetv3s.pt on GuideDog dataset: https://huggingface.co/datasets/kjunh/GuideDog
Initially, we trained a model to fit the ground plane and walls to see surface deformations and classify them as obstacles when they are in the central region of the frame, the region in which the user will be walking.
Then we left that code, we trained a mobilenetv3s.pt model on GuideDog dataset. That is the model being used in the final project we submitted.
