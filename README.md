# Face-Liveness-AntiSpoof-Detector

Face Liveness Detector is a neural network based system that takes images as input and predicts whether the face is real or fake. Fake images can be photographs, images on a phone, mask.

This Face Liveness Detector is based on two models:

1. Face Image Liveness Network
2. Eye Movement Liveness Network


## Face Image Liveness Network

Architecture: ResNet. Experimented with VGG16, DenseNet, Resnet
Dataset: NUAA Photo Imposter Database, Kaggle YouTube Photo Imposter Database
Dataset size: 12614, 20918
Train Accuracy: 0.9942, Validation Accuracy: 0.9606 (on Epoch 8)
Test Accuracy: 0.9729
The Network takes cropped image of the face as input.

![Face Image Liveness Network](/images/1.png)


## Eye Movement Liveness Network

Architecture: KNN. Experimented with Linear/ Logistic regression, Decision tree
Dataset: Custom made dataset
Dataset size: 200
Train Accuracy: 0.979, Validation Accuracy: 0.967 (on Epoch 3)
Test Accuracy: 0.9333
The Network takes the ratio of eye(roe) for 10 subsequent frames as input.
Ratio of eye = height of eye / width of eye

![Eye Movement Liveness Network](/images/2.png)


## The Overall Network:

Eye Movement Liveness Network is good at detecting if the input is a static image/ photograph.
Face liveness detection is good at detecting if the input is a video/image from a display.
A combination of these two networks give the best performance.
Overall, the network is more likely to output false positives than false negatives. That is, it is more likely to incorrectly classify a real face as a fake face.
Dlib library and OpenCV were used to extract faces landmarks and face images from video stream and build the system.


## Caveats:

Model sensitive to camera quality and resolution
Experimenting with different quality webcams showed that low quality webcam predicted better with only NUAA dataset, medium quality webcam predicted better with both NUAA and Kaggle YouTube dataset. The YouTube dataset contains high quality images.

Model always predicts face to be fake if there is no adequate lighting.
Possible reasons:
- Real images in dataset are well lit.
- Images in dataset are of people with fair complexion


Papers read:
- 2D Face Liveness Detection: an Overview.
- An Overview of Face Liveness Detection
- Anti-spoofing Face Databases
- Face De-Spoofing Anti-Spoofing via Noise Modeling
- Learn Convolutional Neural Network for Face Anti-Spoofin
