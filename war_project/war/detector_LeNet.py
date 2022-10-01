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
#from playground.defense_project.utils import *
from playground.war_phase.utils import *
from pathlib import Path
import datetime
import os
from torchvision import datasets, transforms
import collections
import random
#from playground.defense_project.tasks.defense_project.predict import LeNet
from predict import LeNet

random.seed(42)

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
    

class Det_Training():
    def __init__(self, device, epsilon=0.2, min_val=0, max_val=1):
        self.model = LeN().to(device)
        self.ori_model = LeNet().to(device)
        self.ori_model.load_state_dict(torch.load("/Volumes/test/maestro-class/playground/war_phase/tasks/war_defense/war_defense-Jason-01/war_defense-model-clean.pth"))
        self.epsilon = epsilon
        self.min_val = min_val
        self.max_val = max_val
        self.device = device
        self.max_iters = 10
        self._type = 'linf'
        self.alpha = 0.1
        
    def project(self, x, original_x, epsilon, _type='linf'):
        if _type == 'linf':
            max_x = original_x + epsilon
            min_x = original_x - epsilon
            x = torch.max(torch.min(x, max_x), min_x)
        elif _type == 'l2':
            dist = (x - original_x)
            dist = dist.view(x.shape[0], -1)
            dist_norm = torch.norm(dist, dim=1, keepdim=True)
            mask = (dist_norm > epsilon).unsqueeze(2).unsqueeze(3)
            # dist = F.normalize(dist, p=2, dim=1)
            dist = dist / dist_norm
            dist *= epsilon
            dist = dist.view(x.shape)
            x = (original_x + dist) * mask.float() + x * (1 - mask.float())
        else:
            raise NotImplementedError
        return x

    def perturb_pgd(self, original_images, labels, reduction4loss='mean', random_start=False):
        original_images = torch.unsqueeze(original_images, 0).to(self.device)
        labels = torch.tensor([labels]).to(self.device)

        if random_start:
            rand_perturb = torch.FloatTensor(original_images.shape).uniform_(
                -self.epsilon, self.epsilon)
            rand_perturb = rand_perturb.to(self.device)
            x = original_images + rand_perturb
            x.clamp_(self.min_val, self.max_val)
        else:
            x = original_images.clone()

        x.requires_grad = True

        # max_x = original_images + self.epsilon
        # min_x = original_images - self.epsilon

        with torch.enable_grad():
            for _iter in range(self.max_iters):
                self.ori_model.eval()
                # print(x.shape)
                outputs = self.ori_model(x)
                loss = F.cross_entropy(outputs, labels, reduction=reduction4loss)

                if reduction4loss == 'none':
                    grad_outputs = torch.ones(loss.shape).to(self.device)

                else:
                    grad_outputs = None

                grads = torch.autograd.grad(loss, x, grad_outputs=grad_outputs,
                        only_inputs=True)[0]

                x.data += self.alpha * torch.sign(grads.data)
                x = self.project(x, original_images, self.epsilon, self._type)
                x.clamp_(self.min_val, self.max_val)

        final_pred = outputs.max(1, keepdim=True)[1]

        correct = 0
        if final_pred.item() != labels.item():
            correct = 1
        # return x
        return x

    def perturb_fgsm(self, original_images, labels):
        original_images = torch.unsqueeze(original_images, 0).to(self.device)
        labels = torch.tensor([labels]).to(self.device)
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

    def perturb_gibbon(self, original, labels):
        eps = 7.5/255.0
        eps_iter = 4.5/255.0
        steps = 100
        original = torch.unsqueeze(original, 0).to(self.device)
        labels = torch.tensor([labels]).to(self.device)
        original.requires_grad = True

        for i in range(10):
            if i > 0:
                eps += 0.04
                eps_iter += 0.0055
            adv_img = original.clone().detach()

            adv_img = adv_img + torch.empty_like(adv_img).uniform_(-eps, eps)
            adv_img = torch.clamp(adv_img, min=0, max=1).detach()

            for step in range(steps):
                self.ori_model.eval()
                outputs = self.ori_model(original)   
                loss = F.nll_loss(outputs, labels)
                self.ori_model.zero_grad()
                loss.backward()
                data_grad = original.grad.data
                data_grad =data_grad.to(self.device)
                adv_img = adv_img.to(self.device)
                grad_sign = data_grad.sign()
                adv_img = adv_img.detach() + eps_iter * grad_sign
                delta = torch.clamp(adv_img - original, min=-eps, max=eps)
                adv_img = torch.clamp(original + delta, min=0, max=1)
                logits = self.ori_model(adv_img)
                logits = torch.tensor(logits).to(self.device)
                probs = nn.Softmax(dim=1)(logits).cpu().detach().numpy()

                if (np.argmax(probs) != labels):
                    return adv_img

        return adv_img
    
    def compute_gradient(self, original_images, labels):
        original_images = torch.tensor(original_images).to(self.device)
        labels = torch.tensor(labels).to(self.device)
        original_images.requires_grad = True
        self.ori_model.eval()
        outputs = self.ori_model(original_images)
        loss = F.nll_loss(outputs, labels)
        self.ori_model.zero_grad()
        loss.backward()
        data_grad = original_images.grad.data

        return data_grad

    def deepfool(self, original_images, labels, num_classes=5, overshoot=0.02, max_iter=50):
        labels = torch.tensor([labels]).to(self.device)
        # original_images.shape = [1,3,32,32]
        perturbed_image = original_images.clone().detach()
        perturbed_image = perturbed_image.to(self.device)
        output = self.ori_model(perturbed_image)[0].to(self.device)
        output_ranks = torch.argsort(output, descending=True)
        output_ranks = output_ranks[0:num_classes].to(self.device)
        # if original prediction is wrong, no need to attack
        if labels != output_ranks[0]:
            return perturbed_image

        input_image_shape = perturbed_image.shape
        w = torch.zeros(input_image_shape).to(self.device)
        r_tot = torch.zeros(input_image_shape).to(self.device)
        loop_i = 0
        prediction = labels[0]
        while prediction == labels[0] and loop_i < max_iter:
            pert = np.inf
            orig_grad = self.compute_gradient(perturbed_image, torch.tensor([output_ranks[0]]))

            for k in range(1, num_classes):
                cur_grad = self.compute_gradient(perturbed_image, torch.tensor([output_ranks[k]]))
                w_k = orig_grad - cur_grad
                output_k = output[output_ranks[k]] - output[output_ranks[0]]
                pert_k = torch.abs(output_k) / torch.linalg.norm(w_k.flatten())
                if pert_k < pert:
                    pert = pert_k
                    w = w_k

            r_i = (pert + 1e-4) * w / torch.linalg.norm(w)
            r_tot = torch.add(r_tot, r_i)
            perturbed_image = torch.clamp(original_images + (1 + overshoot) * r_tot, self.min_val, self.max_val)
            output = self.ori_model(perturbed_image)[0]
            prediction = torch.argsort(output, descending=True)[0]
            loop_i += 1
        # print(f'ori:{labels[0]}, pert:{prediction}')
        return perturbed_image

    def perturb_Tian(self, original_images, labels):
        # original_images.shape = [3,32,32], need unsqueeze
        original_images = torch.tensor(original_images).to(self.device)
        original_images = torch.unsqueeze(original_images, 0)
        labels = torch.tensor(labels).to(self.device)
        correct = 0
        perturbed_image = self.deepfool(original_images, labels)
        adv_outputs = self.ori_model(perturbed_image)
        final_pred = adv_outputs.max(1, keepdim=True)[1]
        correct = 0
        if final_pred.item() != labels.item():
            correct = 1

        return perturbed_image
    
    def train(self, model, trainset, valset, device, epoches=7):
        model.to(device)
        model.train()
        b_size = 100
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=b_size, shuffle=True, num_workers=8)
        dataset_size = len(trainset)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters())
        train_data = []
        print("Generate Datasets!")
        total = 0
        for inputs, labels in trainloader:
            for data in inputs:
                train_data.append([data, torch.tensor([0])])
            if total < 800:
                for i in range(len(inputs)):
                    train_data.append([self.perturb_pgd(inputs[i], labels[i]), torch.tensor([1])])
            elif total < 7000:
                for i in range(len(inputs)):
                    train_data.append([self.perturb_Tian(inputs[i], labels[i]), torch.tensor([1])])
            if total > 7000:
                for i in range(len(inputs)):
                    train_data.append([self.perturb_gibbon(inputs[i], labels[i]), torch.tensor([1])])
            total += len(inputs)
            
        #shuffle

        random.shuffle(train_data)
        print("Start Training!")

        for epoch in range(epoches):  # loop over the dataset multiple times
            running_loss = 0.0
            for inputs, labels in train_data:
                inputs = inputs.reshape([1,3,32,32])
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs, self.ori_model.layers(inputs))
                loss = criterion(outputs, labels)
                loss.backward(retain_graph=True)
                optimizer.step()
                running_loss += loss.item()
            print("epoch: ", epoch, ", loss: ", running_loss / dataset_size)
            
        
        model.eval()
        valloader = torch.utils.data.DataLoader(valset, batch_size=b_size, shuffle=True, num_workers=8)
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in valloader:
                # print(inputs.shape, labels.shape)
                inputs = inputs.to(device)
                labels = torch.zeros(b_size).to(device)
                outputs = model(inputs, self.ori_model.layers(inputs))
                _, predicted = torch.max(outputs.data, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print("Accuracy of the network on the val clean images: %.3f %%" % (100 * correct / total))

        correct = 0
        total = 0

        for inputs, labels in valloader:
            # print(inputs.shape, labels.shape)
            for i in range(len(inputs)):
                input = inputs[i].to(device)
                label = labels[i].to(device)
                pert_input = self.perturb_pgd(input,label)
                label = torch.ones(1).to(device)
                output = model(pert_input, self.ori_model.layers(pert_input))
                _, predicted = torch.max(output.data, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()
        print("Accuracy of the network on the val adv pgd images: %.3f %%" % (100 * correct / total))

        correct = 0
        total = 0
        
        for inputs, labels in valloader:
            # print(inputs.shape, labels.shape)
            for i in range(len(inputs)):
                input = inputs[i].to(device)
                label = labels[i].to(device)
                pert_input = self.perturb_Tian(input,label)
                label = torch.ones(1).to(device)
                output = model(pert_input, self.ori_model.layers(pert_input))
                _, predicted = torch.max(output.data, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()
        print("Accuracy of the network on the val adv Tian images: %.3f %%" % (100 * correct / total))

        correct = 0
        total = 0

        for inputs, labels in valloader:
            # print(inputs.shape, labels.shape)
            for i in range(len(inputs)):
                input = inputs[i].to(device)
                label = labels[i].to(device)
                pert_input = self.perturb_gibbon(input,label)
                label = torch.ones(1).to(device)
                output = model(pert_input, self.ori_model.layers(pert_input))
                _, predicted = torch.max(output.data, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()
        print("Accuracy of the network on the val adv gibbon images: %.3f %%" % (100 * correct / total))
        


        return model

def load_defender(task_folder, task, device):
    spec = importlib.util.spec_from_file_location(
        str(task) ,
        
        "detector_LeNet.py",
    )
    foo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(foo)
    defender = foo.Det_Training(device)
    return defender


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_configs = {
                "name": "CIFAR10",
                "dataset_path": "/Volumes/test/maestro-class/playground/war_phase/datasets/CIFAR10/war",
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
    task = "war_phase"
    method = load_defender(task, task, device)
    model = method.train(method.model, trainset, valset, device)
    torch.save(model.state_dict(), "/Volumes/test/maestro-class/playground/war_phase/tasks/war_defense/war_defense-Jason-01/war_project-model-detector-LeNet.pth")

        

