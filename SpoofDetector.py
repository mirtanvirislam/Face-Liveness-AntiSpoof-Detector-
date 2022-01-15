import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from torchvision.utils import make_grid
from torch.utils.data import Dataset, random_split, DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random
import pickle
from sklearn.neighbors import KNeighborsClassifier


class SpoofDetector:
    """
    SpoofDetector detects whether input image is real or fake
    """

    def __init__(self):
        """
        Initializes networks and loads saved models for predicting
        """
        self.face_liveness_network = models.resnet18(pretrained=False)
        self.face_liveness_network.load_state_dict(torch.load("./data/res18.pth", map_location=torch.device('cpu')))
        self.face_liveness_network.eval()
        self.eye_liveness_network = pickle.load(open('./data/eye_ratio_model', 'rb'))

    def predict(self, face_image, roe):
        """
        Argument type: Tensor, np array
        Output: is fake, confidence
        Returns False if input image is real
        Returns True if input image is fake
        """
        outputs = self.face_liveness_network(face_image)
        prediction = outputs.argmax(dim=1).item()
        confidence = (torch.nn.functional.softmax(outputs[:2], dim=1))[0][prediction].item()
        roe = roe.reshape(-1, 20)

        eye_liveness_predictions = self.eye_liveness_network.predict(roe)
        eye_liveness_prediction = eye_liveness_predictions[0]

        if eye_liveness_prediction == 'real':
            return (prediction != 1), confidence
        else:
            return True, 1
