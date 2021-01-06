import os
import numpy as np
import torch
import torchvision
import sklearn.datasets

class Dataset:
    def __init__(self):
        assert len(self.x_train) == len(self.y_train)
        assert len(self.x_test) == len(self.y_test)
        self.n_trainb = len(self.x_train)
        self.n_testb = len(self.x_test)
        self.train_idx = 0
        self.test_idx = 0
    def next_training_batch(self):
        rv = (self.x_train[self.train_idx], self.y_train[self.train_idx])
        self.train_idx = (self.train_idx+1)%(self.n_trainb)
        return rv
    def next_test_batch(self):
        rv = (self.x_test[self.test_idx], self.y_test[self.test_idx])
        self.test_idx = (self.test_idx+1)%(self.n_testb)
        return rv

class MNIST(Dataset):
    def __init__(self, batch_size, device):
        self.name = 'MNIST'
        self.n_in = 28**2
        self.n_out = 10
        self.batch_size = batch_size
        training_examples = torchvision.datasets.MNIST(
                root=os.path.join(os.getcwd(), 'datasets'),
                train=True,
                download=True,
                transform=torchvision.transforms.ToTensor(),
                target_transform=None)
        training_loader = torch.utils.data.DataLoader(dataset=training_examples, batch_size=batch_size, shuffle=False)
        raw_data = [out for out in training_loader]
        np.random.shuffle(raw_data)
        self.x_train = [d[0].to(device).reshape([batch_size, 28**2]) for d in raw_data]
        self.y_train = []
        for [_, y] in raw_data:
            self.y_train.append(torch.zeros([batch_size, 10]))
            for n in range(len(y)):
                self.y_train[-1][n, y[n]] = 1
        test_examples = torchvision.datasets.MNIST(
                root=os.path.join(os.getcwd(), 'datasets'),
                train=False,
                download=True,
                transform=torchvision.transforms.ToTensor(),
                target_transform=None)
        test_loader = torch.utils.data.DataLoader(dataset=test_examples, batch_size=batch_size, shuffle=False)
        raw_data = [out for out in test_loader]
        self.x_test = [d[0].to(device).reshape([batch_size, 28**2]) for d in raw_data]
        self.y_test = []
        for [_, y] in raw_data:
            self.y_test.append(torch.zeros([batch_size, 10]).to(device))
            for n in range(len(y)):
                self.y_test[-1][n, y[n]] = 1
        Dataset.__init__(self)

class FashionMNIST(Dataset):
    def __init__(self, batch_size, device):
        self.name = 'Fashion MNIST'
        self.n_in = 28**2
        self.n_out = 10
        self.batch_size = batch_size
        training_examples = torchvision.datasets.FashionMNIST(
                root=os.path.join(os.getcwd(), 'datasets'),
                train=True,
                download=True,
                transform=torchvision.transforms.ToTensor(),
                target_transform=None)
        training_loader = torch.utils.data.DataLoader(dataset=training_examples, batch_size=batch_size, shuffle=False)
        raw_data = [out for out in training_loader]
        np.random.shuffle(raw_data)
        self.x_train = [d[0].to(device).reshape([batch_size, 28**2]) for d in raw_data]
        self.y_train = []
        for [_, y] in raw_data:
            self.y_train.append(torch.zeros([batch_size, 10]))
            for n in range(len(y)):
                self.y_train[-1][n, y[n]] = 1
        test_examples = torchvision.datasets.FashionMNIST(
                root=os.path.join(os.getcwd(), 'datasets'),
                train=False,
                download=True,
                transform=torchvision.transforms.ToTensor(),
                target_transform=None)
        test_loader = torch.utils.data.DataLoader(dataset=test_examples, batch_size=batch_size, shuffle=False)
        raw_data = [out for out in test_loader]
        self.x_test = [d[0].to(device).reshape([batch_size, 28**2]) for d in raw_data]
        self.y_test = []
        for [_, y] in raw_data:
            self.y_test.append(torch.zeros([batch_size, 10]).to(device))
            for n in range(len(y)):
                self.y_test[-1][n, y[n]] = 1
        Dataset.__init__(self)

class Diabetes(Dataset):
    def __init__(self, batch_size, device):
        assert batch_size <= 42
        self.name = 'Diabetes'
        self.n_in = 10
        self.n_out = 1
        self.batch_size = batch_size
        (x_raw, y_raw) = sklearn.datasets.load_diabetes(return_X_y=True, as_frame=False)
        np.random.shuffle(x_raw)
        np.random.shuffle(y_raw)
        self.x_train = []
        self.y_train = []
        for idx in np.arange(int(400/batch_size)):
            x_batch = np.stack(x_raw[batch_size*idx:batch_size*(idx+1)], axis=0)
            y_batch = np.stack(y_raw[batch_size*idx:batch_size*(idx+1)], axis=0)
            self.x_train.append(torch.from_numpy(x_batch).to(device))
            self.y_train.append(torch.from_numpy(y_batch).to(device))
        self.x_test = []
        self.y_test = []
        for idx in np.arange(int(400/batch_size), int(442/batch_size)):
            x_batch = np.stack(x_raw[batch_size*idx:batch_size*(idx+1)], axis=0)
            y_batch = np.stack(y_raw[batch_size*idx:batch_size*(idx+1)], axis=0)
            self.x_test.append(x_batch)
            self.y_test.append(y_batch)
        Dataset.__init__(self)

class Wine(Dataset):
    def __init__(self, batch_size, device):
        assert batch_size <= 28
        self.name = 'Wine'
        self.n_in = 13
        self.n_out = 3
        self.batch_size = batch_size
        (x_raw, y_raw) = sklearn.datasets.load_wine(return_X_y=True, as_frame=False)
        np.random.shuffle(x_raw)
        np.random.shuffle(y_raw)
        self.x_train = []
        self.y_train = []
        for idx in np.arange(int(150/batch_size)):
            x_batch = np.stack(x_raw[batch_size*idx:batch_size*(idx+1)], axis=0)
            y_batch = np.stack(y_raw[batch_size*idx:batch_size*(idx+1)], axis=0)
            self.x_train.append(torch.from_numpy(x_batch).to(device))
            self.y_train.append(torch.from_numpy(y_batch).to(device))
        self.x_test = []
        self.y_test = []
        for idx in np.arange(int(150/batch_size), int(178/batch_size)):
            x_batch = np.stack(x_raw[batch_size*idx:batch_size*(idx+1)], axis=0)
            y_batch = np.stack(y_raw[batch_size*idx:batch_size*(idx+1)], axis=0)
            self.x_test.append(x_batch)
            self.y_test.append(y_batch)
        Dataset.__init__(self)
        