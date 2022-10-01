from typing import List, Iterator, Dict, Tuple, Any, Type
import numpy as np
import torch
from copy import deepcopy
from torch.utils.data import DataLoader
from Maestro.attacker_helper.attacker_request_helper import virtual_model
from transformers.data.data_collator import default_data_collator
from Maestro.evaluator.Evaluator import get_data

class ProjectAttack:
    def __init__(
        self,
        vm,
        image_size: List[int],
        l2_threshold=7.5

    ):
        self.vm = vm
        self.image_size = image_size
        self.l2_threshold = l2_threshold

    def attack(
        self,
        original_image:  np.ndarray,
        labels: List[int],
        target_label: int,
        epsilon = 0.214

    ):
        """
        args:
            original_image: a numpy ndarray images, [1,3,32,32]
            labels: label of the image, a list of size 1
            target_label: target label we want the image to be classified, int
        return:
            the perturbed image
            label of that perturbed iamge
            success: whether the attack succeds
        """

        
        # --------------TODO--------------
        X = deepcopy(original_image)
        target_labels = np.ones_like(labels) * target_label
        n = 500
        max_iter = 100
        alpha = 0.05
        #eps = 5 / 255
        #X = X + np.random.uniform(-eps, eps)
        for _ in range(max_iter):
            data_grad = self.vm.get_batch_input_gradient(X, target_labels)
            grad_flat = data_grad.flatten()
            grad_ranks = np.argsort(data_grad.flatten())[::-1]
            ma = grad_ranks[:n]
            mi = grad_ranks[-n:]
            X_ = X.flatten()
            X_[ma] = X_[ma] - alpha * grad_flat[ma]
            X_[mi] = X_[mi] - alpha * grad_flat[mi]
            X_ = X_.reshape(X.shape)
            X = X_
            X = np.clip(X,0,1)
            if np.argmax(self.vm.get_batch_output(X)[0]) == target_label:
                break
        # ------------END TODO-------------
        return X


