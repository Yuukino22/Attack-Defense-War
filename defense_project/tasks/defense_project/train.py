"""
The template for the students to train the model.
Please do not change the name of the functions in Adv_Training.
"""
import sys
sys.path.append(".")
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from tasks.defense_project.predict import LeNet

class Adv_Training():
    """
    The class is used to set the defense related to adversarial training and adjust the loss function. 
    Please design your own training methods and add some adversarial examples for training.
    The perturb function is used to generate the adversarial examples for training.
    """
    def __init__(self, device, epsilon=0.3, min_val=0, max_val=1):
        self.model = LeNet().to(device)
        self.device = device
        self.epsilon = epsilon
        self.min_val = min_val
        self.max_val = max_val
        self.max_iters = 10
        self.alpha = 0.1
        self._type = 'linf'
    
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
        return x[0]

    def perturb(self, original_images, labels):
        original_images.requires_grad = True
        self.model.eval()
        outputs = self.model(original_images)
        loss = F.nll_loss(outputs, labels)
        self.model.zero_grad()
        loss.backward()
        data_grad = original_images.grad.data
        sign_data_grad = data_grad.sign()
        perturbed_image = original_images + self.epsilon*sign_data_grad
        perturbed_image = torch.clamp(perturbed_image, self.min_val, self.max_val)
        self.model.zero_grad()
        return perturbed_image
    
    # def perturb(self, original_images, labels, reduction4loss='mean', random_start=False):
    #     original_images = torch.unsqueeze(original_images, 0).to(self.device)
    #     labels = torch.tensor(labels).to(self.device)

    #     if random_start:
    #         rand_perturb = torch.FloatTensor(original_images.shape).uniform_(
    #             -self.epsilon, self.epsilon)
    #         rand_perturb = rand_perturb.to(self.device)
    #         x = original_images + rand_perturb
    #         x.clamp_(self.min_val, self.max_val)
    #     else:
    #         x = original_images.clone()

    #     x.requires_grad = True
    #     # max_x = original_images + self.epsilon
    #     # min_x = original_images - self.epsilon
    #     x = x[0]
    #     with torch.enable_grad():
    #         for _iter in range(self.max_iters):
    #             self.model.eval()
    #             outputs = self.model(x)

    #             loss = F.cross_entropy(outputs, labels, reduction=reduction4loss)

    #             if reduction4loss == 'none':
    #                 grad_outputs = torch.ones(loss.shape).to(self.device)

    #             else:
    #                 grad_outputs = None

    #             grads = torch.autograd.grad(loss, x, grad_outputs=grad_outputs,
    #                     only_inputs=True)[0]

    #             x.data += self.alpha * torch.sign(grads.data)
    #             x = self.project(x, original_images, self.epsilon, self._type)
    #             x.clamp_(self.min_val, self.max_val)
                

    #     return x

    def train(self, model, trainset, valset, device, epoches=20):
        model.to(device)
        model.train()
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=10)
        dataset_size = len(trainset)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters())
        for epoch in range(epoches):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, (inputs, labels) in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs = inputs.to(device)
                labels = labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                pertubed = self.perturb(inputs, labels)
                pert_output = model(pertubed)
                pert_loss = criterion(pert_output, labels)
                loss += pert_loss
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / (dataset_size)))
            running_loss = 0.0

        valloader = torch.utils.data.DataLoader(valset, batch_size=100, shuffle=True, num_workers=10)
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in valloader:
                # print(inputs.shape, labels.shape)
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print("Accuracy of the network on the val images: %.3f %%" % (100 * correct / total))

        return model


