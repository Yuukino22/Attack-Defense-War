"""
The template for the students to predict the result.
Please do not change LeNet, the name of batch_predict and predict function of the Prediction.

"""

import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim


class LeNet(nn.Module):
    """
    """
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x) -> torch.tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def layers(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x1 = F.relu(self.fc1(x))
        x2 = F.relu(self.fc2(x1))
        return x2

class LeN(nn.Module):
    """
    """
    def __init__(self):
        super(LeN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5 + 84, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x, y) -> torch.tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = torch.cat([x[0], y[0]])
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = torch.unsqueeze(x,0)
        return x
    
    def p(self, x, y):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = torch.cat([x[0], y[0]])
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = torch.unsqueeze(x,0)
        return torch.exp(x[0][1]) / (torch.exp(x[0][0]) + torch.exp(x[0][1]))

class LeNLayer(nn.Module):
    """
    """
    def __init__(self):
        super(LeNLayer, self).__init__()
        self.fc1 = nn.Linear(84, 120)
        self.fc2 = nn.Linear(120, 40)
        self.fc3 = nn.Linear(40, 2)

    def forward(self, x) -> torch.tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def p(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.exp(x[0][1]) / (torch.exp(x[0][0]) + torch.exp(x[0][1]))

class Prediction():
    """
    The Prediction class is used for evaluator to load the model and detect or classify the images. The output of the batch_predict function will be checked which is the label.
    If the label is the same as the ground truth, it means you predict the image successfully. If the label is -1 and the image is an adversarial examples, it means you predict the image successfully. Other situations will be decided as failure.
    You can use the preprocess function to clean or check the input data are benign or adversarial for later prediction.
    """
    def __init__(self, device, pretrained_file):
        self.device = device
        self.model = self.constructor(pretrained_file).to(device)
        self.clean_model = LeNet().to(device)
        self.clean_model.load_state_dict(torch.load(pretrained_file+'/war_defense-model-clean.pth', map_location=self.device))
        self.clean_model_layer = LeNet().to(device)
        self.clean_model_layer.load_state_dict(torch.load(pretrained_file+'/war_defense-model-clean-layer.pth', map_location=self.device))
        self.detector_LeNet = LeN().to(device)
        self.detector_LeNet.load_state_dict(torch.load(pretrained_file+'/war_project-model-detector-LeNet.pth', map_location=self.device))
        self.detector_only_layers = LeNLayer().to(device)
        self.detector_only_layers.load_state_dict(torch.load(pretrained_file+'//war_project-model-detector-only-layers.pth', map_location=self.device))

    def constructor(self, pretrained_file=None):
        model = LeNet()
        if pretrained_file != None:
            model.load_state_dict(torch.load(pretrained_file+'/war_defense-model-clean.pth', map_location=self.device))
        return model

    def preprocess(self, original_images):
        perturbed_image = original_images.unsqueeze(0)
        return perturbed_image
    
    def detect_attack_LeNet(self, original_image):
        original_image = torch.unsqueeze(original_image, 0).to(self.device)
        output = self.detector_LeNet.p(original_image, self.clean_model.layers(original_image))
        return output
        # _, predicted = torch.max(output.data, 1)
        # return predicted == 1


    def detect_attack_only_layers(self, original_image):
        original_image = torch.unsqueeze(original_image, 0).to(self.device)
        output = self.detector_only_layers.p(self.clean_model_layer.layers(original_image))
        return output
        # _, predicted = torch.max(output.data, 1)
        # return predicted == 1

    def batch_predict(self, images):
        predictions = []
        for image in images:
            if 0.23 * self.detect_attack_only_layers(image) + 1.77 * self.detect_attack_LeNet(image) > 0.95:
                predictions.append(-1)
            else:
                prediction = self.predict(image)
                predictions.append(prediction)
        predictions = torch.tensor(predictions).to(self.device)
        return predictions

    def predict(self, image):
        image = self.preprocess(image)
        output = self.model(image)
        _, predicted = torch.max(output.data, 1)
        return predicted
