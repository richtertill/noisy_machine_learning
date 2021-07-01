"""

Dataloader

main function: dataloader(dataset, params)
returns: trainloader, valloader, testloader

"""

from torchvision import datasets
from pathlib import Path
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision
from torchvision import transforms, utils
from torchvision.transforms import *
from datetime import date
import csv
import sys
import os

project_root_dir = os.getcwd()
"""
def get_project_root():
    return Path(__file__).parent.parent

sys.path.append(get_project_root())
project_root_dir = get_project_root()
"""
from src.noise import add_data_noise,add_label_noise

# label noise profiles to choose from
label_noise_profiles = ['no', 'deterministic', 'vanilla', 'symmetric', 'symmetrc_multi', 'assymetric_single', 'assymetric']
# data noise profiles to choose from
data_noise_profiles = ['no', 'gauss', 'salt_pepper', 'poisson', 'speckle', 'open_set']

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




def dataloader(params):
    """
    Dataloader, that is called to return batch of images and labels
    :param params: Parameters, see params folder
    :return: train data loader, validation data loader, test data loader -> torch data loader
    """
    dataset = params.dataset_class_name

    if dataset == 'SVHN':
        data_train = SVHNDataset(params, split='train')

        data_val = SVHNDataset(params, split='val')

        data_test = SVHNDataset(params, split='test')

        train_loader = DataLoader(data_train, batch_size=params.batch_size, shuffle=True,
                                  num_workers=params.num_workers, pin_memory=True)

        val_loader = DataLoader(data_val, batch_size=params.batch_size, shuffle=True, num_workers=params.num_workers,
                                pin_memory=True)

        test_loader = DataLoader(data_test, batch_size=params.batch_size, num_workers=params.num_workers,
                                 pin_memory=True)

    if dataset == 'CIFAR10':
        data_train = CIFAR10Dataset(params, split='train')

        data_val = CIFAR10Dataset(params, split='val')

        data_test = CIFAR10Dataset(params, split='test')

        train_loader = DataLoader(data_train, batch_size=params.batch_size, shuffle=True, drop_last=True,
                                  num_workers=params.num_workers)

        val_loader = DataLoader(data_val, batch_size=params.batch_size, shuffle=True, drop_last=True,
                                num_workers=params.num_workers)

        test_loader = DataLoader(data_test, batch_size=params.batch_size, num_workers=params.num_workers)

    if dataset == 'FashionMNIST':
        data_train = FashionMNIST(params, split='train')

        data_val = FashionMNIST(params, split='val')

        data_test = FashionMNIST(params, split='test')

        train_loader = DataLoader(data_train, batch_size=params.batch_size, shuffle=False,
                                  num_workers=params.num_workers)

        val_loader = DataLoader(data_val, batch_size=params.batch_size, shuffle=False, num_workers=params.num_workers)

        test_loader = DataLoader(data_test, batch_size=params.batch_size, num_workers=params.num_workers)

    if dataset == 'MNIST':
        data_train = MNIST(params, split='train')

        data_val = MNIST(params, split='val')

        data_test = MNIST(params, split='test')

        train_loader = DataLoader(data_train, batch_size=params.batch_size, shuffle=False,
                                  num_workers=params.num_workers)

        val_loader = DataLoader(data_val, batch_size=params.batch_size, shuffle=False, num_workers=params.num_workers)

        test_loader = DataLoader(data_test, batch_size=params.batch_size, num_workers=params.num_workers)

    return train_loader, val_loader, test_loader


class SVHNDataset(Dataset):
    def __init__(self, params, split):
        """
        SVHN Dataset
        :param params: Parameter file, see params folder
        :param split: String, checks if currently the train, validation or test set is called
        """
        self.params = params
        self.split = split
        SVHN_labels = torch.arange(0, 10)  # possible labels of SVHN

        mean = [0.4380, 0.4440, 0.4730]
        std = [0.1751, 0.1771, 0.1744]
        normalize = transforms.Normalize(mean=mean,
                                         std=std)
        self.transform_train = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize])

        self.transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                normalize,
            ]
        )

        download_dir = str(project_root_dir) + '/data/svhn'
        dataset = datasets.SVHN(root=str(download_dir) , download=True, transform=ToTensor())

        # get data and labels, possibly perturbed labels
        data = torch.Tensor(dataset.data)  # 73257 x 3 x 32 x 32
        labels = torch.Tensor(dataset.labels)
        length = 73257
        PATH = str(project_root_dir) + "/data/semantic/"  # semantic experiment
        
        if any(elem in label_noise_profiles for elem in params.experts["labels_noise"]) and not params.use_semantic_labels:
            # concatenate all data and labels that the experts see, incl. their
            # pertubated expert labels
            perturbed_classes = [int(params.experts["classes"][0]), int(params.experts["classes"][1])]
            ind_a = (labels == perturbed_classes[
                0]).nonzero().flatten()  # get indices of ground truth class to be perturbed
            ind_b = (labels == perturbed_classes[
                1]).nonzero().flatten()  # get indices of ground truth class to be perturbed
            n_corrupted = int(params.intensity * min(ind_a.shape[0], ind_b.shape[0]))
            ind_a_corrupted = ind_a[:n_corrupted]
            ind_b_corrupted = ind_b[:n_corrupted]

            indices = [ind_a_corrupted, ind_b_corrupted]

            for expert in range(len(params.experts["labels_noise"])):
                expert_data = add_data_noise(params.train_data_noise, data, prob=params.intensity)
                expert_labels = add_label_noise(params.experts["labels_noise"][expert], labels, labels=SVHN_labels,
                                                prob=params.intensity, classes=params.experts["classes"], indices=indices)
                if expert == 0:
                    all_data = expert_data
                    all_labels = expert_labels
                else:
                    all_data = torch.cat((all_data, expert_data), dim=0)
                    all_labels = torch.cat((all_labels, expert_labels), dim=0)

            # choose 73257 images to be comparable
            rand_ind = torch.randperm(all_labels.shape[0])[:73257]
            used_data = all_data[rand_ind]
            used_labels = all_labels[rand_ind]
            self.val_data = used_data[int(params.sample_sizes['train'] * length):, :, :]
            self.train_data = used_data[:int(params.sample_sizes['train'] * length), :, :]
            # self.test_data = torch.reshape(torch.Tensor(test_set.data), (torch.Tensor(test_set.data).shape[0], 3, 32, 32))
            
            self.val_labels = used_labels[int(params.sample_sizes['train'] * length):]
            self.train_labels = used_labels[:int(params.sample_sizes['train'] * length)]
            
        else:
            train_data = data[:int(params.sample_sizes['train'] * length), :, :]
            val_data = data[int(params.sample_sizes['train'] * length):, :, :]
            train_labels = labels[:int(params.sample_sizes['train'] * length)]
            val_labels = labels[int(params.sample_sizes['train'] * length):]
            self.train_data = add_data_noise(params.train_data_noise, train_data, prob=params.intensity)
            self.val_data = add_data_noise(params.train_data_noise, val_data, prob=params.intensity)
            self.train_labels = train_labels
            self.val_labels = val_labels

            # for training semantic dataset: train data is only x% of the training set
            # for training semantic dataset: test data is remaining training set
        test_set = datasets.SVHN(root=str(download_dir), split='test', download=True)
        self.test_data = torch.Tensor(test_set.data)
        self.test_labels = torch.Tensor(test_set.labels)

        if params.use_semantic_labels:  # use the semantic labels
            # get the val and test data & labels
            self.val_data = torch.load(PATH + "SVHN_val_data.pt", map_location='cpu')  # 12000xdimxdim
            self.test_data = torch.load(PATH + "SVHN_test_data.pt", map_location='cpu')  # 10000xdimxdim
            self.val_labels = torch.load(PATH + "SVHN_val_labels.pt", map_location='cpu')  # 12000
            self.test_labels = torch.load(PATH + "SVHN_test_labels.pt", map_location='cpu')  # 10000

            semantic_train_data = torch.load(PATH + "SVHN_train_train_data.pt", map_location='cpu')
            semantic_test_data = torch.load(PATH + "SVHN_train_test_data.pt", map_location='cpu')
            semantic_train_labels = torch.load(PATH + "SVHN_train_train_labels.pt", map_location='cpu')  # 2400

            if params.intensity == 0.1:
                perturbed_labels = torch.load(PATH + "semantic_svhn_labels_10.pt", map_location='cpu')  # 45600
            elif params.intensity == 0.2:
                perturbed_labels = torch.load(PATH + "semantic_svhn_labels_20.pt", map_location='cpu')  # 45600
            elif params.intensity == 0.3:
                perturbed_labels = torch.load(PATH + "semantic_svhn_labels_30.pt", map_location='cpu')  # 45600
            elif params.intensity == 0.4:
                perturbed_labels = torch.load(PATH + "semantic_svhn_labels_40.pt", map_location='cpu')  # 45600
            else:
                print(
                    'Only 10%, 20%, 30% and 40% semantic noise is implemented. This will cause an error from data_load.py')
            semantic_test_data = torch.reshape(semantic_test_data, (semantic_test_data.shape[0], 3, 32, 32))
            self.train_data = torch.cat((semantic_train_data, semantic_test_data), dim=0)  # 48000xdimxdim
            self.train_labels = torch.cat((semantic_train_labels, perturbed_labels), dim=0)  # 48000

            # for training semantic dataset: train data is only x% of the training set, test data is remaining training set
        if params.semantic_labels_noise:
            torch.save(self.test_data, PATH + "SVHN_test_data.pt")
            torch.save(self.val_data, PATH + "SVHN_val_data.pt")
            torch.save(self.val_labels, PATH + "SVHN_val_labels.pt")
            torch.save(self.test_labels, PATH + "SVHN_test_labels.pt")
            torch.save(self.train_data, PATH + "SVHN_train_data.pt")

            self.test_data = self.train_data[int(length * params.semantic_set):, ]
            self.train_data = self.train_data[:int(length * params.semantic_set), ]
            self.val_data = self.val_data[:int(self.val_data.shape[0] * params.semantic_set), ]

            self.test_labels = self.train_labels[int(length * params.semantic_set):, ]
            self.train_labels = self.train_labels[:int(length * params.semantic_set), ]
            self.val_labels = self.val_labels[:int(self.val_labels.shape[0] * params.semantic_set), ]

            # save the data and labels for semantic set
            torch.save(self.train_data, PATH + "SVHN_train_train_data.pt")
            torch.save(self.test_labels, PATH + "SVHN_train_test_labels.pt")
            torch.save(self.train_labels, PATH + "SVHN_train_train_labels.pt")
            # shape: batch x 3 x 32 x 32
        self.train_data = torch.reshape(self.train_data, (self.train_data.shape[0], 32, 32, 3))
        self.val_data = torch.reshape(self.val_data, (self.val_data.shape[0], 32, 32, 3))
        self.test_data = torch.reshape(self.test_data, (self.test_data.shape[0], 32, 32, 3))

    def __len__(self):
        """
        Get lengths of the dataset
        :return: length
        """
        if self.split == 'train':
            return len(self.train_data)
        if self.split == 'val':
            return len(self.val_data)
        if self.split == 'test':
            return len(self.test_data)

    def __getitem__(self, index):
        """
        Returns a single item from the train loader
        :param index: index of the image
        :return: image, label at position index
        Transforms doesn't work in __init__ if the tensors are perturbed afterwards, so it's included individually here
        """
        if self.split == 'train':
            x_train = self.train_data[index]  # PIL only works with numpy
            if torch.is_tensor(x_train):
                x_train = x_train.numpy().astype(np.uint8)
            y_train = self.train_labels[index]
            if self.transform_train:
                x_train = self.transform_train(transforms.ToPILImage()(x_train).convert("RGB"))
            return x_train, y_train.type(torch.LongTensor), index

        if self.split == 'val':
            x_val = self.val_data[index]
            if torch.is_tensor(x_val):
                x_val = x_val.numpy().astype(np.uint8)
            y_val = self.val_labels[index]
            if self.transform_test:
                x_val = self.transform_test(transforms.ToPILImage()(x_val).convert("RGB"))
            return x_val, y_val.type(torch.LongTensor), index

        if self.split == 'test':
            x_test = self.test_data[index]
            if torch.is_tensor(x_test):
                x_test = x_test.numpy().astype(np.uint8)
            y_test = self.test_labels[index]
            if self.transform_test:
                x_test = self.transform_test(transforms.ToPILImage()(x_test).convert("RGB"))
            x_test = add_data_noise(self.params.test_data_noise, x_test, prob=self.params.intensity)
            return x_test, y_test.type(torch.LongTensor), index

class CIFAR10Dataset(Dataset):
    def __init__(self, params, split):
        """
        CIFAR 10 Dataset
        :param params: Parameter file, see params folder
        :param split: String, checks if currently the train, validation or test set is called
        """
        self.params = params
        self.split = split
        cifar10_labels = torch.arange(0, 10)  # possible labels of CIFAR10
        mean, std = self._get_statistics()

        self.transform_train = transforms.Compose(
            [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            ])

        self.transform_test = transforms.Compose(
            [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            ])

        dataset = datasets.CIFAR10(root = str(project_root_dir) + '/data/cifar10', train=True, download=True)

        # get data and labels, perhaps perturbed labels
        data = torch.Tensor(dataset.data)  # 50.000 x 32 x 32 x 3
        labels = torch.Tensor(dataset.targets)
        length = 40000  # length of training set in mnist

        # get semantic labels path
        PATH = str(project_root_dir) + "/data/semantic/"

        if any(elem in label_noise_profiles  for elem in params.experts["labels_noise"]) and not params.use_semantic_labels:
            # concatenate all data and labels that the experts see, incl. their
            # pertubated expert labels
            perturbed_classes = [int(params.experts["classes"][0]), int(params.experts["classes"][1])]
            ind_1 = (labels == perturbed_classes[
                0]).nonzero().flatten()  # get indices of ground truth class to be perturbed
            ind_2 = (labels == perturbed_classes[
                1]).nonzero().flatten()  # get indices of ground truth class to be perturbed
            indices = [ind_1, ind_2]

            for expert in range(len(params.experts["labels_noise"])):
                expert_data = add_data_noise(params.train_data_noise, data, prob=params.intensity)
                expert_labels = add_label_noise(params.experts["labels_noise"][expert], labels, labels=cifar10_labels,
                                                prob=params.intensity, classes=params.experts["classes"], indices=indices)
                if expert == 0:
                    all_data = expert_data
                    all_labels = expert_labels
                else:
                    all_data = torch.cat((all_data, expert_data), dim=0)
                    all_labels = torch.cat((all_labels, expert_labels), dim=0)

            # choose 40000 images to be comparable
            rand_ind = torch.randperm(all_labels.shape[0])[:50000]
            used_data = all_data[rand_ind]
            used_labels = all_labels[rand_ind]

            # test set is exclusive
            download_dir = str(project_root_dir) + '/data/cifar10'
            test_set = datasets.CIFAR10(root=str(project_root_dir) + '/data/cifar10', train=False, download=True)

            # train val test split
            # used data: 50000 x 32 x 32 x 3
            # test set 10000 x 32 x 32 x 3
            self.val_data = used_data[int(params.sample_sizes['train'] * len(dataset)):, :, :]
            self.train_data = used_data[:int(params.sample_sizes['train'] * len(dataset)), :, :]
            self.test_data = torch.Tensor(test_set.data)
            self.val_labels = used_labels[int(params.sample_sizes['train'] * len(dataset)):]
            self.train_labels = used_labels[:int(params.sample_sizes['train'] * len(dataset))]
            self.test_labels = torch.Tensor(test_set.targets)

        else:
            train_data = data[:int(params.sample_sizes['train'] * len(dataset)), :, :]
            val_data = data[int(params.sample_sizes['train'] * len(dataset)):, :, :]
            train_labels = labels[:int(params.sample_sizes['train'] * len(dataset))]
            val_labels = labels[int(params.sample_sizes['train'] * len(dataset)):]
            download_dir = str(project_root_dir) + '/data/cifar10'
            test_set = datasets.CIFAR10(root=str(project_root_dir) + '/data/cifar10', train=False, download=True)
            # adapt train data, i.e. include data and label noise
            self.train_data = add_data_noise(params.train_data_noise, train_data, prob=params.intensity)
            self.val_data = add_data_noise(params.train_data_noise, val_data, prob=params.intensity)
            self.test_data = add_data_noise(params.train_data_noise, test_set.data, prob=params.intensity)

            self.train_labels = train_labels
            self.val_labels = val_labels
            self.test_labels = torch.tensor(test_set.targets)

        # for training semantic dataset: train data is only x% of the training set
        # for training semantic dataset: test data is remaining training set
        if params.use_semantic_labels:  # use the semantic labels
            # get the val and test data & labels
            self.val_data = torch.load(PATH + "CIFAR10_val_data.pt", map_location='cpu')  # 12000xdimxdim
            self.test_data = torch.load(PATH + "CIFAR10_test_data.pt", map_location='cpu')  # 10000xdimxdim
            self.val_labels = torch.load(PATH + "CIFAR10_val_labels.pt", map_location='cpu')  # 12000
            self.test_labels = torch.load(PATH + "CIFAR10_test_labels.pt", map_location='cpu')  # 10000

            semantic_train_data = torch.load(PATH + "CIFAR10_train_train_data.pt", map_location='cpu')
            semantic_test_data = torch.load(PATH + "CIFAR10_train_test_data.pt", map_location='cpu')
            semantic_train_labels = torch.load(PATH + "CIFAR10_train_train_labels.pt", map_location='cpu')  # 2400

            if params.intensity == 0.1:
                perturbed_labels = torch.load(PATH + "semantic_cifar10_labels_10.pt", map_location='cpu')  # 45600
            elif params.intensity == 0.2:
                perturbed_labels = torch.load(PATH + "semantic_cifar10_labels_20.pt", map_location='cpu')  # 45600
            elif params.intensity == 0.3:
                perturbed_labels = torch.load(PATH + "semantic_cifar10_labels_30.pt", map_location='cpu')  # 45600
            elif params.intensity == 0.4:
                perturbed_labels = torch.load(PATH + "semantic_cifar10_labels_40.pt", map_location='cpu')  # 45600
            else:
                print(
                    'Only 10%, 20%, 30% and 40% semantic noise is implemented. This will cause an error from data_load.py')

            semantic_test_data = torch.reshape(semantic_test_data, (semantic_test_data.shape[0], 32, 32, 3))

            self.train_data = torch.cat((semantic_train_data, semantic_test_data), dim=0)  # 48000xdimxdim
            self.train_labels = torch.cat((semantic_train_labels, perturbed_labels), dim=0)  # 48000

        # for training semantic dataset: train data is only x% of the training set, test data is remaining training set
        if params.semantic_labels_noise:
            torch.save(self.test_data, PATH + "CIFAR10_test_data.pt")
            torch.save(self.val_data, PATH + "CIFAR10_val_data.pt")
            torch.save(self.val_labels, PATH + "CIFAR10_val_labels.pt")
            torch.save(self.test_labels, PATH + "CIFAR10_test_labels.pt")
            torch.save(self.train_data, PATH + "CIFAR10_train_data.pt")

            self.test_data = self.train_data[int(length * params.semantic_set):, ]
            self.train_data = self.train_data[:int(length * params.semantic_set), ]
            self.val_data = self.val_data[:int(self.val_data.shape[0] * params.semantic_set), ]

            self.test_labels = self.train_labels[int(length * params.semantic_set):, ]
            self.train_labels = self.train_labels[:int(length * params.semantic_set), ]
            self.val_labels = self.val_labels[:int(self.val_labels.shape[0] * params.semantic_set), ]

            # save the data and labels for semantic set
            torch.save(self.train_data, PATH + "CIFAR10_train_train_data.pt")
            torch.save(self.test_labels, PATH + "CIFAR10_train_test_labels.pt")
            torch.save(self.train_labels, PATH + "CIFAR10_train_train_labels.pt")
        # shape: batch x 32 x 32 x 3

    def __len__(self):
        """
        Get lengths of the dataset
        :return: length
        """
        if self.split == 'train':
            return len(self.train_data)
        if self.split == 'val':
            return len(self.val_data)
        if self.split == 'test':
            return len(self.test_data)

    def __getitem__(self, index):
        """
        Returns a single item from the train loader
        :param index: index of the image
        :return: image, label at position index
        """
        if self.split == 'train':
            x_train = self.train_data[index]  # somehow to PIL only works with numpy
            if torch.is_tensor(x_train):
                x_train = x_train.numpy().astype(np.uint8)
            y_train = self.train_labels[index]
            if self.transform_train:
                x_train = self.transform_train(transforms.ToPILImage()(x_train).convert("RGB"))
            return x_train, y_train.type(torch.LongTensor), index

        if self.split == 'val':
            x_val = self.val_data[index]
            if torch.is_tensor(x_val):
                x_val = x_val.numpy().astype(np.uint8)
            y_val = self.val_labels[index]
            if self.transform_test:
                x_val = self.transform_test(transforms.ToPILImage()(x_val).convert("RGB"))
            return x_val, y_val.type(torch.LongTensor), index

        if self.split == 'test':
            x_test = self.test_data[index]
            if torch.is_tensor(x_test):
                x_test = x_test.numpy().astype(np.uint8)
            y_test = self.test_labels[index]
            if self.transform_test:
                x_test = self.transform_test(transforms.ToPILImage()(x_test).convert("RGB"))
            x_test = add_data_noise(self.params.test_data_noise, x_test, prob=self.params.intensity)
            return x_test, y_test.type(torch.LongTensor), index

    def _get_statistics(self):
        """
        Get mean and std of the images for transformation
        :return: Mean, Std of images
        """
        train_set = torchvision.datasets.CIFAR10(root=str(project_root_dir) + '/data/cifar10', train=True, download=True,
                                                 transform=transforms.ToTensor())
        test_set = torchvision.datasets.CIFAR10(root=str(project_root_dir) + '/data/cifar10', train=False, download=True,
                                                transform=transforms.ToTensor())

        data = torch.cat([d[0] for d in DataLoader(train_set)] + [d[0] for d in DataLoader(test_set)])
        return data.mean(dim=[0, 2, 3]), data.std(dim=[0, 2, 3])


class FashionMNIST(Dataset):
    def __init__(self, params, split):
        """
        MNIST Dataset
        :param params: Parameter file, see params folder
        :param split: String, checks if currently the train, validation or test set is called
        """
        self.params = params
        self.split = split
        mnist_labels = torch.arange(0, 10)  # possible labels of MNIST
        self.transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.RandomCrop(28, padding=4),
            transforms.RandomErasing(
                p=0.5,
                scale=(0.02, 0.33),
                ratio=(0.3, 3.3),
                value="random",
                inplace=False,
            ),
            transforms.Lambda(lambda x: x.repeat(1, 1, 1)),
        ])

        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(lambda x: x.repeat(1, 1, 1)),
        ])
        dataset = datasets.FashionMNIST(root = project_root_dir + '/data/fashionmnist', download=True)
        # get data and labels, perhaps perturbed labels
        data = dataset.data
        labels = dataset.targets
        length = 48000  # length of training set in mnist

        PATH = str(project_root_dir) + "/data/semantic/"

        if any(elem in label_noise_profiles  for elem in params.experts["labels_noise"]) and not params.use_semantic_labels:

            # concatenate all data and labels that the experts see, incl. their
            # pertubated expert labels
            perturbed_classes = [int(params.experts["classes"][0]), int(params.experts["classes"][1])]
            ind_1 = (labels == perturbed_classes[
                0]).nonzero().flatten()  # get indices of ground truth class to be perturbed
            ind_2 = (labels == perturbed_classes[
                1]).nonzero().flatten()  # get indices of ground truth class to be perturbed
            indices = [ind_1, ind_2]

            for expert in range(len(params.experts["labels_noise"])):
                expert_data = add_data_noise(params.train_data_noise, data, prob=params.intensity)
                expert_labels = add_label_noise(params.experts["labels_noise"][expert], labels, labels=mnist_labels,
                                                prob=params.intensity, classes=params.experts["classes"], indices=indices)
                if expert == 0:
                    all_data = expert_data
                    all_labels = expert_labels
                else:
                    all_data = torch.cat((all_data, expert_data), dim=0)
                    all_labels = torch.cat((all_labels, expert_labels), dim=0)

            # choose 60000 images to be comparable
            rand_ind = torch.randperm(all_labels.shape[0])[:60000]
            used_data = all_data[rand_ind]
            used_labels = all_labels[rand_ind]

            # test set is exclusive
            download_dir = str(project_root_dir) + '/data/fashionmnist'
            test_set = datasets.FashionMNIST(root=project_root_dir + '/data/fashionmnist', train=False, download=True)

            # train val test split
            self.val_data = used_data[int(params.sample_sizes['train'] * len(dataset)):, :, :]
            self.train_data = used_data[:int(params.sample_sizes['train'] * len(dataset)), :, :]
            self.test_data = test_set.data

            self.val_labels = used_labels[int(params.sample_sizes['train'] * len(dataset)):]
            self.train_labels = used_labels[:int(params.sample_sizes['train'] * len(dataset))]
            self.test_labels = test_set.targets
        else:
            train_data = data[:int(params.sample_sizes['train'] * len(dataset)), :, :]
            val_data = data[int(params.sample_sizes['train'] * len(dataset)):, :, :]
            train_labels = labels[:int(params.sample_sizes['train'] * len(dataset))]
            val_labels = labels[int(params.sample_sizes['train'] * len(dataset)):]
            download_dir = str(project_root_dir) + '/data/fashionmnist'
            test_set = datasets.FashionMNIST(root=project_root_dir + '/data/fashionmnist', train=False, download=True)

            # adapt train data, i.e. include data and label noise
            self.train_data = add_data_noise(params.train_data_noise, train_data, prob=params.intensity)
            self.val_data = add_data_noise(params.train_data_noise, val_data, prob=params.intensity)
            self.test_data = add_data_noise(params.train_data_noise, test_set.data, prob=params.intensity)

            self.train_labels = train_labels
            self.val_labels = val_labels
            self.test_labels = test_set.targets

        if params.use_semantic_labels:  # use the semantic labels
            # get the val and test data & labels
            self.val_data = torch.load(PATH + "FashionMNIST_val_data.pt", map_location='cpu')  # 12000x28x28
            self.test_data = torch.load(PATH + "FashionMNIST_test_data.pt", map_location='cpu')  # 10000x28x28
            self.val_labels = torch.load(PATH + "FashionMNIST_val_labels.pt", map_location='cpu')  # 12000
            self.test_labels = torch.load(PATH + "FashionMNIST_test_labels.pt", map_location='cpu')  # 10000

            semantic_train_data = torch.load(PATH + "FashionMNIST_train_train_data.pt", map_location='cpu')
            semantic_test_data = torch.load(PATH + "FashionMNIST_train_test_data.pt", map_location='cpu')
            semantic_train_labels = torch.load(PATH + "FashionMNIST_train_train_labels.pt", map_location='cpu')  # 2400

            perturbed_labels = torch.load(PATH + "FashionMNIST4semantic_create.pt", map_location='cpu')  # 45600
            self.train_data = torch.cat((semantic_train_data, semantic_test_data), dim=0)  # 48000x28x28
            self.train_labels = torch.cat((semantic_train_labels, perturbed_labels), dim=0)  # 48000

        # for training semantic dataset: train data is only x% of the training set
        # for training semantic dataset: test data is remaining training set
        if params.semantic_labels_noise:
            torch.save(self.test_data, PATH + "FashionMNIST_test_data.pt")
            torch.save(self.val_data, PATH + "FashionMNIST_val_data.pt")
            torch.save(self.val_labels, PATH + "FashionMNIST_val_labels.pt")
            torch.save(self.test_labels, PATH + "FashionMNIST_test_labels.pt")
            torch.save(self.train_data, PATH + "FashionMNIST_train_data.pt")

            self.test_data = self.train_data[int(length * params.semantic_set):, ]
            self.train_data = self.train_data[:int(length * params.semantic_set), ]
            self.val_data = self.val_data[:int(self.val_data.shape[0] * params.semantic_set), ]

            self.test_labels = self.train_labels[int(length * params.semantic_set):, ]
            self.train_labels = self.train_labels[:int(length * params.semantic_set), ]
            self.val_labels = self.val_labels[:int(self.val_labels.shape[0] * params.semantic_set), ]

            # save the data and labels for semantic set
            torch.save(self.train_data, PATH + "FashionMNIST_train_train_data.pt")
            torch.save(self.train_labels, PATH + "FashionMNIST_train_train_labels.pt")
            torch.save(self.test_labels, PATH + "FashionMNIST_train_test_labels.pt")

    def __len__(self):
        """
        Get length of the dataloaders
        :return: Length
        """
        if self.split == 'train':
            return len(self.train_data)
        if self.split == 'val':
            return len(self.val_data)
        if self.split == 'test':
            return len(self.test_data)

    def __getitem__(self, index):
        """
        Returns a single item from the train loader
        :param index: index of the image
        :return: image, label at position index
        """
        if self.split == 'train':
            x_train = self.train_data[index]  # PIL only works with numpy
            if torch.is_tensor(x_train):
                x_train = x_train.numpy().astype(np.uint8)
            y_train = self.train_labels[index]
            if self.transform_train:
                x_train = self.transform_train(transforms.ToPILImage()(x_train).convert("RGB"))
            return x_train, y_train.type(torch.LongTensor), index

        if self.split == 'val':
            x_val = self.val_data[index]
            if torch.is_tensor(x_val):
                x_val = x_val.numpy().astype(np.uint8)
            y_val = self.val_labels[index]
            if self.transform_train:
                x_val = self.transform_train(transforms.ToPILImage()(x_val).convert("RGB"))
            return x_val, y_val.type(torch.LongTensor), index

        if self.split == 'test':
            x_test = self.test_data[index]
            if torch.is_tensor(x_test):
                x_test = x_test.numpy().astype(np.uint8)
            y_test = self.test_labels[index]
            if self.transform_test:
                x_test = self.transform_test(transforms.ToPILImage()(x_test).convert("RGB"))
            x_test = add_data_noise(self.params.test_data_noise, x_test, prob=self.params.intensity)
            return x_test, y_test.type(torch.LongTensor), index


class MNIST(Dataset):
    def __init__(self, params, split):
        """
        MNIST Dataset
        :param params: Parameter file, see params folder
        :param split: String, checks if currently the train, validation or test set is called
        """
        self.params = params
        self.split = split
        mnist_labels = torch.arange(0, 10)  # possible labels of MNIST
        self.transform_train = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
        self.transform_test = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
        download_dir = str(project_root_dir) + '/data/mnist'
        dataset = datasets.MNIST(root = project_root_dir + '/data/mnist', download=True, transform=self.transform_train)
        # get data and labels, perhaps perturbed labels
        data = dataset.data
        labels = dataset.targets
        length = 48000  # length of training set in mnist
        PATH = str(project_root_dir) + "/data/semantic/"

        if any(elem in label_noise_profiles  for elem in params.experts["labels_noise"]) and not params.use_semantic_labels:
            # concatenate all data and labels that the experts see, incl. their
            # pertubated expert labels
            perturbed_classes = [int(params.experts["classes"][0]), int(params.experts["classes"][1])]
            ind_a = (labels==perturbed_classes[0]).nonzero().flatten()  # get indices of ground truth class to be perturbed
            ind_b = (labels==perturbed_classes[1]).nonzero().flatten()  # get indices of ground truth class to be perturbed
            n_corrupted = int(params.intensity * min(ind_a.shape[0], ind_b.shape[0]))
            ind_a_corrupted = ind_a[:n_corrupted]
            ind_b_corrupted = ind_b[:n_corrupted]
            
            indices = [ind_a_corrupted, ind_b_corrupted]
            perturbed_ind = torch.cat((ind_a_corrupted, ind_b_corrupted), dim=0)
            non_perturbed_ind = torch.arange(labels.shape[0])
            for ind in range(perturbed_ind.shape[0]):
                non_perturbed_ind = non_perturbed_ind[non_perturbed_ind != perturbed_ind[ind]]
            for expert in range(len(params.experts["labels_noise"])):
                expert_data = add_data_noise(params.train_data_noise, data, prob=params.intensity)
                expert_labels = add_label_noise(params.experts["labels_noise"][expert], labels, labels=mnist_labels,
                                                prob=params.intensity, classes=params.experts["classes"], indices=indices)
                if expert == 0:
                    all_data = expert_data
                    all_labels = expert_labels
                else:
                    all_data = torch.cat((all_data, expert_data), dim=0)
                    all_labels = torch.cat((all_labels, expert_labels), dim=0)
            
            # test set is exclusive
            download_dir = str(project_root_dir) + '/data/mnist'
            test_set = datasets.MNIST(root=project_root_dir + '/data/mnist', train=False, download=True, transform=self.transform_test)

            # train val test split
            self.val_data = all_data[int(params.sample_sizes['train'] * all_data.shape[0]):, :, :]  # old with used_data
            self.train_data = all_data[:int(params.sample_sizes['train'] * all_data.shape[0]), :, :]  # old with used_data
            self.test_data = test_set.data
            # self.test_data = self.train_data[perturbed_ind, ]
            self.val_labels = all_labels[int(params.sample_sizes['train'] * all_data.shape[0]):]
            self.train_labels = all_labels[:int(params.sample_sizes['train'] * all_data.shape[0])]
            self.test_labels = test_set.targets
            # self.test_labels = self.train_labels[perturbed_ind]
        else:
            train_data = data[:int(params.sample_sizes['train'] * len(dataset)), :, :]
            val_data = data[int(params.sample_sizes['train'] * len(dataset)):, :, :]
            train_labels = labels[:int(params.sample_sizes['train'] * len(dataset))]
            val_labels = labels[int(params.sample_sizes['train'] * len(dataset)):]
            download_dir = str(project_root_dir) + '/data/mnist'
            test_set = datasets.MNIST(root=project_root_dir + '/data/mnist', train=False, download=True, transform=self.transform_test)

            # adapt train data, i.e. include data and label noise
            self.train_data = add_data_noise(params.train_data_noise, train_data, prob=params.intensity)
            self.val_data = add_data_noise(params.train_data_noise, val_data, prob=params.intensity)
            self.test_data = add_data_noise(params.train_data_noise, test_set.data, prob=params.intensity)

            self.train_labels = train_labels
            self.val_labels = val_labels
            self.test_labels = test_set.targets

        if params.use_semantic_labels:  # use the semantic labels
            # get the val and test data & labels
            self.val_data = torch.load(PATH + "MNIST_val_data.pt", map_location='cpu')  # 12000x28x28
            self.test_data = torch.load(PATH + "MNIST_test_data.pt", map_location='cpu')  # 10000x28x28
            self.val_labels = torch.load(PATH + "MNIST_val_labels.pt", map_location='cpu')  # 12000
            self.test_labels = torch.load(PATH + "MNIST_test_labels.pt", map_location='cpu')  # 10000

            semantic_train_data = torch.load(PATH + "MNIST_train_train_data.pt", map_location='cpu')
            semantic_test_data = torch.load(PATH + "MNIST_train_test_data.pt", map_location='cpu')
            semantic_train_labels = torch.load(PATH + "MNIST_train_train_labels.pt", map_location='cpu')  # 2400

            if params.intensity == 0.1:
                perturbed_labels = torch.load(PATH + "semantic_mnist_labels_10.pt", map_location='cpu')  # 45600
            elif params.intensity == 0.2:
                perturbed_labels = torch.load(PATH + "semantic_mnist_labels_20.pt", map_location='cpu')  # 45600
            elif params.intensity == 0.3:
                perturbed_labels = torch.load(PATH + "semantic_mnist_labels_30.pt", map_location='cpu')  # 45600
            elif params.intensity == 0.4:
                perturbed_labels = torch.load(PATH + "semantic_mnist_labels_40.pt", map_location='cpu')  # 45600
            else:
                print(
                    'Only 10%, 20%, 30% and 40% semantic noise is implemented. This will cause an error from data_load.py')
            self.train_data = torch.cat((semantic_train_data, semantic_test_data), dim=0)  # 48000x28x28
            self.train_labels = torch.cat((semantic_train_labels, perturbed_labels), dim=0)  # 48000
            
        # for training semantic dataset: train data is only x% of the training set
        # for training semantic dataset: test data is remaining training set
        if params.semantic_labels_noise:
            torch.save(self.test_data, PATH + "MNIST_test_data.pt")
            torch.save(self.val_data, PATH + "MNIST_val_data.pt")
            torch.save(self.val_labels, PATH + "MNIST_val_labels.pt")
            torch.save(self.test_labels, PATH + "MNIST_test_labels.pt")
            torch.save(self.train_data, PATH + "MNIST_train_data.pt")

            self.test_data = self.train_data[int(length * params.semantic_set):, ]
            self.train_data = self.train_data[:int(length * params.semantic_set), ]
            self.val_data = self.val_data[:int(self.val_data.shape[0] * params.semantic_set), ]

            self.test_labels = self.train_labels[int(length * params.semantic_set):, ]
            self.train_labels = self.train_labels[:int(length * params.semantic_set), ]
            self.val_labels = self.val_labels[:int(self.val_labels.shape[0] * params.semantic_set), ]

            # save the data and labels for semantic set
            torch.save(self.train_data, PATH + "MNIST_train_train_data.pt")
            torch.save(self.train_labels, PATH + "MNIST_train_train_labels.pt")
            torch.save(self.test_labels, PATH + "MNIST_train_test_labels.pt")

    def __len__(self):
        """
        Get length of the dataloaders
        :return: Length
        """
        if self.split == 'train':
            return len(self.train_data)
        if self.split == 'val':
            return len(self.val_data)
        if self.split == 'test':
            return len(self.test_data)

    def __getitem__(self, index):
        """
        Returns a single item from the train loader
        :param index: index of the image
        :return: image, label at position index
        """
        if self.split == 'train':
            return self.train_data[index], self.train_labels[index], index
        if self.split == 'val':
            return self.val_data[index], self.val_labels[index], index
        if self.split == 'test':
            self.test_data[index] = add_data_noise(self.params.test_data_noise, self.test_data[index],
                                                   prob=self.params.intensity)
            return self.test_data[index], self.test_labels[index], index