import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import sys
sys.path.append(".")
import importlib.util
from typing import List, Iterator, Dict, Tuple, Any, Type
from torch.utils.data import DataLoader
import time
import requests
import pickle
from playground.defense_project.utils import *
from pathlib import Path
import datetime
import os
from torchvision import datasets, transforms
import collections
import random
from playground.defense_project.tasks.defense_project.predict import LeNet

class LeN(nn.Module):
    """
    """
    def __init__(self):
        super(LeN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x) -> torch.tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
        

class Det_Training():
    def __init__(self, device, epsilon=0.2, min_val=0, max_val=1):
        self.model = LeN().to(device)
        self.ori_model = LeNet().to(device)
        self.ori_model.load_state_dict(torch.load("/Volumes/test/maestro-class/playground/defense_project/models/defense_project-model-clean.pth"))
        self.epsilon = epsilon
        self.min_val = min_val
        self.max_val = max_val
        
    
    def perturb(self, original_images, labels):
        original_images.requires_grad = True
        self.ori_model.eval()
        outputs = self.ori_model(original_images)
        loss = F.nll_loss(outputs, labels)
        self.ori_model.zero_grad()
        loss.backward()
        data_grad = original_images.grad.data
        sign_data_grad = data_grad.sign()
        perturbed_image = original_images + self.epsilon*sign_data_grad
        perturbed_image = torch.clamp(perturbed_image, self.min_val, self.max_val)
        self.ori_model.zero_grad()
        return perturbed_image
    
    def train(self, model, trainset, valset, device, epoches=1):
        model.to(device)
        model.train()
        b_size = 100
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=b_size, shuffle=True, num_workers=10)
        dataset_size = len(trainset)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters())
        train_data = []
        for inputs, labels in trainloader:
            for data in inputs:
                train_data.append([data, torch.tensor([0])])
            pert_inputs = self.perturb(inputs, labels)
            for data in pert_inputs:
                train_data.append([data, torch.tensor([1])])
            
        #shuffle

        random.shuffle(train_data)

        for epoch in range(epoches):  # loop over the dataset multiple times
            running_loss = 0.0
            for inputs, labels in train_data:
                inputs = inputs.reshape([1,3,32,32])
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward(retain_graph=True)
                optimizer.step()
                running_loss += loss.item()
            print("epoch: ", epoch, ", loss: ", running_loss / dataset_size)
            
        
        # for epoch in range(epoches):  # loop over the dataset multiple times
        #     running_loss = 0.0
        #     for i, (inputs, labels) in enumerate(trainloader, 0):
        #         # get the inputs; data is a list of [inputs, labels]
        #         inputs = inputs.to(device)
        #         labels = torch.zeros((100,1)).to(device)
        #         # zero the parameter gradients
        #         optimizer.zero_grad()
        #         outputs = model(inputs)
        #         loss = criterion(outputs, labels)
        #         loss.backward()
        #         optimizer.step()
        #         running_loss += loss.item()
        #     print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / dataset_size))
        #     running_loss = 0.0
        
        model.eval()
        valloader = torch.utils.data.DataLoader(valset, batch_size=b_size, shuffle=True, num_workers=10)
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in valloader:
                # print(inputs.shape, labels.shape)
                inputs = inputs.to(device)
                labels = torch.zeros(b_size).to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print("Accuracy of the network on the val clean images: %.3f %%" % (100 * correct / total))

        correct = 0
        total = 0
        
        for inputs, labels in valloader:
            # print(inputs.shape, labels.shape)
            inputs = inputs.to(device)
            labels = labels.to(device)
            pert_inputs = self.perturb(inputs,labels)
            labels = torch.ones(b_size).to(device)
            outputs = model(pert_inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print("Accuracy of the network on the val adv images: %.3f %%" % (100 * correct / total))

        return model

def load_defender(task_folder, task, device):
    spec = importlib.util.spec_from_file_location(
        str(task) ,
        
        "detector.py",
    )
    foo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(foo)
    defender = foo.Det_Training(device)
    return defender


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_configs = {
                "name": "CIFAR10",
                "dataset_path": "/Volumes/test/maestro-class/playground/defense_project/datasets/CIFAR10/",
                "student_train_number": 10000,
                "server_train_number": 10000,
                "student_val_number": 1000,
                "student_test_number": 1000,
                "server_test_number": 1000
    }
    dataset = get_dataset("CIFAR10", dataset_configs)
    trainset = dataset['train']
    valset = dataset['val']
    testset = dataset['test']
    task = "defense_project"
    method = load_defender(task, task, device)
    model = method.train(method.model, trainset, valset, device)
    torch.save(model.state_dict(), "/Volumes/test/maestro-class/playground/defense_project/models/defense_project-model-detector.pth")

        

