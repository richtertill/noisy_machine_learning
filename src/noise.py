#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-
import os
import sys
import torch
from torch.distributions.poisson import Poisson
import numpy as np
from scipy import stats


def add_data_noise(data_noise, tensor, prob, mean=0, std=1):
    """
    adds noise to the image
    :param std:
    :param mean:
    :param prob: for salt_pepper loss, default 5% of pixels are set to black or white
    :param data_noise: str, type of noise to be added
    'gauss'         Gaussian-distributed additive noise
    'salt_pepper'   Replaces random pixels with 0 or 1
    'poisson'       Posson-distributed noise generated from the data
    'speckle'       Multiplicative noise
    :param tensor: torch tensor, image
    :return: torch tensor, image with added noise
    """
    # we need torch tensor, dataloader might return numpy array
    if not torch.is_tensor(tensor):
        tensor = torch.from_numpy(tensor)
    if data_noise == 'no':
        return tensor

    if data_noise == 'gauss':
        """
        1. L2-norm of the signal/image using functions like 
            numpy.linalg.norm —> Signal Power P_s (Scalar)
        2. L2-norm of the noise using the same function above 
            —> Noise Power P_n (Scalar)
        3. Compute the SNR = P_s / P_n —> Scalar 
        4. Noisy signal = Signal + \alpha * SNR * Noise. 
            —> signal or image, and \alpha controls how much noise you would like to add. 
            For example, when you set \alpha > 1 then you amplify the noise (dominant),   
            however, when you set \alpha < 1 (the signal will be dominant). 
            Suggested start with 0.1 (10%). 
        """
        gaussian = torch.randn(size=tensor.shape)
        if tensor.dim() == 2:  # shape batch x dim x dim
            # get norm for each image
            signal_norm = np.linalg.norm(tensor, axis=(-1, -2))
            noise_norm = np.linalg.norm(gaussian, axis=(-1, -2))
        elif tensor.dim() == 3 and tensor.shape[-1] == 3:  # cifar10 case
            signal_norm = np.linalg.norm(tensor, axis=(0, 1))
            noise_norm = np.linalg.norm(gaussian, axis=(0, 1))
        elif tensor.dim() == 3 and tensor.shape[0] == 3:  # cifar10 new transforms
            signal_norm = np.linalg.norm(tensor, axis=(1,2))
            noise_norm = np.linalg.norm(gaussian, axis=(1,2))
            signal_norm = np.mean(signal_norm, axis=-1)
            noise_norm = np.mean(noise_norm, axis=-1)
        elif tensor.dim() == 4:  # shape batch x channels x dim x dim
            signal_norm = np.linalg.norm(tensor, axis=(-1, -2))
            signal_norm = np.mean(signal_norm, axis=-1)
            noise_norm = np.linalg.norm(gaussian, axis=(-1, -2))
            noise_norm = np.mean(noise_norm, axis=-1)

        snr = (signal_norm / noise_norm)  # -> shape batch
        if type(snr) != torch.tensor:
            snr = torch.tensor(snr)
        out = tensor + prob * snr * gaussian
        return tensor + prob * snr * gaussian

    if data_noise == 'salt_pepper':
        rnd = torch.FloatTensor(tensor.size()).uniform_(0, 1)
        noisy = tensor
        noisy[rnd < prob / 2] = 0.
        noisy[rnd > 1 - prob / 2] = 1.
        return noisy

    if data_noise == 'poisson':
        np_tensor = tensor.numpy()
        val = len(np.unique(np_tensor))
        val = 2 ** np.ceil(np.log2(val))
        np_noisy = np.random.poisson(np_tensor * val) / float(val)
        return torch.from_numpy(np_noisy)

    if data_noise == 'speckle':
        if tensor.dim() == 4:
            gauss = torch.randn(tensor.shape[0], tensor.shape[1], tensor.shape[2], tensor.shape[3])
            return torch.Tensor(tensor + tensor * gauss)
        elif tensor.dim() == 3:
            gauss = torch.randn(tensor.shape[0], tensor.shape[1], tensor.shape[2])
            return torch.Tensor(tensor + tensor * gauss)
        elif tensor.dim() == 2:
            gauss = torch.randn(tensor.shape[0], tensor.shape[1])
            return torch.Tensor(tensor + tensor * gauss)

    if data_noise == 'open_set':
        '''
        Open set noise as in Yu et al (2019)
        Replace prob % of images with images of another dataset
        '''
        print('Open Set noise from Yu et al (2019) is not implemented yet')
        exit()
        return tensor


def add_label_noise(label_noise, tensor, labels, prob, classes, indices):
    """
    adds noise to the labels, currently only for MNIST
    :param labels: array, all possible labels
    :param label_noise: str, indicates type of noise to be added
    :param tensor: torch tensor, true labels from the dataset
    :param prob: percentage of labels to be corrupted
    :param classes: index of classes that should be corrupted
    :return: label, torch tensor of labels with noise
    """
    if label_noise == 'no':
        return tensor

    if label_noise == 'deterministic':
        '''
        Switch labels of first prob% 
        '''
        if type(tensor) != torch.Tensor:
            tensor = torch.Tensor(tensor)
        ind_a_corrupted = indices[0]
        ind_b_corrupted = indices[1]
        tensor[ind_a_corrupted] = classes[1]
        tensor[ind_b_corrupted] = classes[0]
        
        return tensor



    if label_noise == 'vanilla':
        '''
        Switch labels of prob % of label a and b
        '''
        if type(tensor) != torch.Tensor:
            tensor = torch.Tensor(tensor)
        tensor = tensor.long()

        class_a = classes[0]  # class that should be corrupted with class b, int
        class_b = classes[1]  # class that should be corrupted with class a, int
        if tensor.dim() == 2:
            ind_a = tensor[:, class_a].nonzero()
            ind_b = tensor[:, class_b].nonzero()

            count_a = ind_a.shape[0]
            count_b = ind_b.shape[0]

            min_total = min(count_a, count_b)  # get the min of both to have enough true labels left, int
            n_corrupted = int(prob * min_total)  # number of labels to be corrupted, int

            indices_a = np.random.choice(ind_a.squeeze(1), size=n_corrupted)  # get n% of ind_a
            indices_b = np.random.choice(ind_b.squeeze(1), size=n_corrupted)  # get n% of ind_b

            tensor[indices_a, class_a] = 0
            tensor[indices_b, class_b] = 0
            tensor[indices_a, class_b] = 1
            tensor[indices_b, class_a] = 1

        else:
            ind_a = (tensor == class_a).nonzero()  # ind of occurences for a, tensor with [row1, col1], ..., tensor Nx1
            ind_b = (tensor == class_b).nonzero()  # ind of occurences for b, tensor with [row1, col1], ..., tensor Nx1

            count_a = (tensor == class_a).sum().item()  # total number of class a, int
            count_b = (tensor == class_b).sum().item()  # total number of class b, int

            min_total = min(count_a, count_b)  # get the min of both to have enough true labels left, int
            n_corrupted = int(prob * min_total)  # number of labels to be corrupted, int

            indices_a = np.random.choice(ind_a.squeeze(1), size=n_corrupted)  # get n% of ind_a
            indices_b = np.random.choice(ind_b.squeeze(1), size=n_corrupted)  # get n% of ind_b

            tensor[indices_a] = class_b
            tensor[indices_b] = class_a
        return tensor

    """
    Possible label noise, doesn't yield much information gain
    if label_noise == 'symmetric_incl':
        '''
        Switch labels of prob % of all classes, including switching to original number
        '''
        tensor = tensor.long()

        # get n indices that should be corrupted
        indices = torch.randperm(tensor.shape[0])[:int(prob * tensor.size(0))]
        # workaround to be generalized with all types of labels
        length = labels.shape[0]
        ind = torch.arange(0, length)
        # new_label = np.random.choice(ind, size=int(prob * tensor.size(0)), replace=False)
        new_label = torch.randperm(ind.shape[0])[:int(prob * tensor.size(0))].type(torch.FloatTensor)
        tensor[indices] = new_label
        return tensor
    """

    if label_noise == 'symmetric':
        '''
        Switch labels of prob % of class a, excluding switching to original number
        '''
        if type(tensor) != torch.Tensor:
            tensor = torch.Tensor(tensor)
        class_a = classes[0]  # class that should be corrupted with class b, int
        tensor = tensor.long()
        if tensor.dim() == 2:  # multi label case
            ind_a = (tensor[:,
                     class_a] == 1).nonzero()  # ind of occurences for a, tensor with [row1, col1], ..., tensor Nx1
            count_a = ind_a.shape[0]
            n_corrupted = int(prob * count_a)  # number of labels to be corrupted, 
            indices_a = np.random.choice(ind_a.squeeze(1), size=n_corrupted)
            new_label = np.random.choice(np.setdiff1d(range(0, tensor.shape[-1]), int(class_a)),
                                         size=n_corrupted).astype(np.int)
            for i in range(new_label.shape[0]):
                tensor[indices_a[i], new_label[i]] = 1
                tensor[indices_a[i], class_a] = 0

        else:
            ind_a = (tensor == class_a).nonzero()  # ind of occurences for a, tensor with [row1, col1], ..., tensor Nx1
            count_a = (tensor == class_a).sum().item()  # total number of class a, int
            n_corrupted = int(prob * count_a)  # number of labels to be corrupted, 
            indices_a = np.random.choice(ind_a.squeeze(1), size=n_corrupted)  # get n% of ind a      
            length = labels.shape[0]
            ind = torch.arange(0, length)
            new_label = np.empty(indices_a.shape[0])
            for i in range(indices_a.shape[0]):
                new_label[i] = np.random.choice(np.setdiff1d(range(0, length), int(class_a))).astype(np.int)
            tensor[indices_a] = torch.from_numpy(new_label).type(torch.LongTensor)

        return tensor

    if label_noise == 'symmetrc_multi':  # symmetric for multi-label case, eg CheXPert
        if type(tensor) != torch.Tensor:
            tensor = torch.Tensor(tensor)
        class_a = classes[0]  # class that should be corrupted with class b, int
        tensor = tensor.long()
        if tensor.dim() < 2:
            print('This label noise is only designed for multi-label scenario, you have single label')
        else:
            ind_a = (tensor[:,
                     class_a] == 1).nonzero()  # ind of occurences for a, tensor with [row1, col1], ..., tensor Nx1
            count_a = ind_a.shape[0]
            n_corrupted = int(prob * count_a)  # number of labels to be corrupted,
            indices_a = np.random.choice(ind_a.squeeze(1), size=n_corrupted)
            new_label = np.random.choice(np.setdiff1d(range(0, tensor.shape[-1]), int(class_a)),
                                         size=n_corrupted).astype(np.int)
            for i in range(new_label.shape[0]):
                tensor[indices_a[i], new_label[i]] = 1
        return tensor

    if label_noise == 'assymetric_single':
        '''
        Switch labels of prob % or class i to the neighbouring class
        From Want et al 2018a
        Change class i to class i+1
        '''
        if type(tensor) != torch.Tensor:
            tensor = torch.Tensor(tensor)
        class_a = classes[0]  # class that should be corrupted with class b, int
        tensor = tensor.long()
        if class_a == len(labels) - 1:
            new_class = 0
        else:
            new_class = class_a + 1

        if tensor.dim() == 2:  # multi label case
            ind_a = (tensor[:,
                     class_a] == 1).nonzero()  # ind of occurences for a, tensor with [row1, col1], ..., tensor Nx1
            count_a = ind_a.shape[0]
            n_corrupted = int(prob * count_a)  # number of labels to be corrupted, 
            indices_a = np.random.choice(ind_a.squeeze(1), size=n_corrupted)
            indices = torch.randperm(ind_a.shape[0])[:int(prob * ind_a.size(0))]
            tensor_check = tensor
            for i in range(indices.shape[0]):
                tensor[indices[i], class_a] = 0
                tensor[indices[i], class_a + 1] = 1


        else:
            ind_a = (tensor == class_a).nonzero()  # ind of occurences for a, tensor with [row1, col1], ..., tensor Nx1
            # get n indices that should be corrupted
            n_corrupted = int(prob * ind_a.shape[0])  # how many labels should be corrupted
            indices_a = np.random.choice(ind_a.squeeze(1), size=n_corrupted)  # get n% of ind_a

            indices = torch.randperm(ind_a.shape[0])[:int(prob * ind_a.size(0))]
            tensor[indices] = new_class
            tensor.type(torch.FloatTensor)

            count_a = (tensor == class_a).sum().item()  # total number of class a, int
            n_corrupted = int(prob * count_a)  # number of labels to be corrupted, 
            indices_a = np.random.choice(ind_a.squeeze(1), size=n_corrupted)  # get n% of ind a      
            length = labels.shape[0]
            ind = torch.arange(0, length)
            new_label = np.empty(indices_a.shape[0])
            for i in range(indices_a.shape[0]):
                new_label[i] = np.random.choice(np.setdiff1d(range(0, length), int(class_a))).astype(np.int)
            tensor[indices_a] = torch.from_numpy(new_label).type(torch.LongTensor)

        return tensor

    if label_noise == 'assymetric':
        '''
        Switch labels of prob % of all classes to specific incorrect class
        From Wang et al 2018a
        Change class i to class i+1
        '''
        if type(tensor) != torch.Tensor:
            tensor = torch.Tensor(tensor)
        tensor = tensor.long()
        indices = torch.randperm(tensor.shape[0])[:int(prob * tensor.size(0))]  # indices to be permuted

        if tensor.dim() == 2:  # multi label case
            for image in range(indices.shape[0]):
                labels = (tensor[image, :] == 1).nonzero()  # get true labels, ie where the label is 1 for the image

                # 1st set all old labels to 0
                for j in range(labels.shape[0]):
                    tensor[image, labels[j]] = 0

                # 2nd set all new labels to 1
                for j in range(labels.shape[0]):
                    if labels[j] == tensor.shape[-1]:
                        tensor[image, 0] = 1
                    else:
                        tensor[image, labels[j]] = 1

        else:

            length = labels.shape[0]
            new_label = np.empty(indices.shape)
            for i in range(len(indices)):
                if tensor[i] == length - 1:  # old label = 9 -> new label = 0
                    new_label[i] = 0
                else:
                    new_label[i] = tensor[i] + 1
            tensor[indices] = torch.from_numpy(new_label).type(torch.LongTensor)
            tensor.type(torch.FloatTensor)
        return tensor

    if label_noise == 'semantic':
        '''
        Switch labels from false negative samples to the erroneously classified label
        '''
        return tensor

    if label_noise == 'adversarial':
        '''
        Add noise on labels such that the decision boundary is yet not crossed
        '''
        print('Adversarial Label Noise is a future research direction!')
        return tensor
