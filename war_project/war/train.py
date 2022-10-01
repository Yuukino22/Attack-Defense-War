"""
The template for the students to train the model.
Please do not change the name of the functions in Adv_Training.
"""
import sys
sys.path.append("../../../")
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from utils import get_dataset

class Adv_Training():
    """
    The class is used to set the defense related to adversarial training and adjust the loss function. Please design your own training methods and add some adversarial examples for training.
    The perturb function is used to generate the adversarial examples for training.
    """
    def __init__(self, device, file_path, epsilon=0.3, min_val=0, max_val=1):
        sys.path.append(file_path)
        from predict import LeNet
        self.model = LeNet().to(device)
        self.epsilon = epsilon
        self.min_val = min_val
        self.max_val = max_val

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
        return perturbed_image

    def train(self, trainset, valset, device, epoches=30):
        self.model.to(device)
        self.model.train()
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=10)
        dataset_size = len(trainset)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters())
        for epoch in range(epoches):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, (inputs, labels) in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs = inputs.to(device)
                labels = labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / dataset_size))
            running_loss = 0.0

        valloader = torch.utils.data.DataLoader(valset, batch_size=100, shuffle=True, num_workers=10)
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in valloader:
                # print(inputs.shape, labels.shape)
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print("Accuracy of the network on the val images: %.3f %%" % (100 * correct / total))

        return


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adv_training = Adv_Training(device, file_path='.')
    dataset_configs = {
                "name": "CIFAR10",
                "dataset_path": "../../../datasets/CIFAR10/war/",
                "student_train_number": 10000,
                "student_val_number": 1000,
                "student_test_number": 100,
    }

    dataset = get_dataset("CIFAR10", dataset_configs, False)
    trainset = dataset['train']
    valset = dataset['val']
    adv_training.train(trainset, valset, device)
    torch.save(adv_training.model.state_dict(), "war_defense-model.pth")


if __name__ == "__main__":
    main()
