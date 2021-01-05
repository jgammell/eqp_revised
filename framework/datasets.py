import urllib
import gzip
import os
import pickle
import numpy as np
import torch

class MNIST:
    def __init__(self, batch_size, device):
        self.name = 'MNIST'
        self.dimensions = (28**2, 10)
        self.n_batches_train = int(60000/batch_size)
        self.n_batches_test = int(10000/batch_size)
        filepath = os.path.join(os.getcwd(), '..', 'datasets', 'mnist')
        filename = 'mnist.pkl.gz'
        if not(os.path.exists(os.path.join(filepath, filename))):
            urllib.request.urlretrieve('http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz', os.path.join(filepath, filename))
        with gzip.open(os.path.join(filepath, filename), 'rb') as F:
            (x_train, y_train), (x_validate, y_validate), (x_test, y_test) = pickle.load(F, encoding='latin1')
        x = list(x_train) + list(x_validate) + list(x_test)
        y = list(y_train) + list(y_validate) + list(y_test)
        for i, yy in zip(range(len(y)), y):
            v = np.zeros((1, 10))
            v[0][yy] = 1
            y[i] = v
        x = [torch.from_numpy(xx).squeeze().to(device) for xx in x]
        y = [torch.from_numpy(yy).squeeze().to(device) for yy in y]
        self.training_batches = []
        self.test_batches = []
        self.training_index = 0
        self.test_index = 0
        for batch in range(self.n_batches_train):
            self.training_batches.append(
                [[torch.stack(x[batch_size*batch:batch_size*(batch+1)], dim=0).float(),
                  torch.stack(y[batch_size*batch:batch_size*(batch+1)], dim=0).float()],
                  batch])
        for batch in range(self.n_batches_train, self.n_batches_train+self.n_batches_test):
            self.test_batches.append(
                [[torch.stack(x[batch_size*batch:batch_size*(batch+1)], dim=0).float(),
                  torch.stack(y[batch_size*batch:batch_size*(batch+1)], dim=0).float()],
                  batch])
    
    def get_training_batch(self):
        rv = self.training_batches[self.training_index]
        self.training_index = (self.training_index+1)%(self.n_batches_train)
        return rv
    
    def get_test_batch(self):
        rv = self.test_batches[self.test_index]
        self.test_index = (self.test_index+1)%(self.n_batches_test)
        return rv

class Fashion_MNIST:
    def __init__(self, batch_size, device):
        self.name = 'MNIST'
        self.dimensions = (28**2, 10)
        self.n_batches_train = int(60000/batch_size)
        self.n_batches_test = int(10000/batch_size)
        filepath = os.path.join(os.getcwd(), '..', 'datasets', 'fashion_mnist')
        filenames = ['train-images-idx3-ubyte.gz',
                     'train-labels-idx1-ubyte.gz', 
                     't10k-images-idx3-ubyte.gz', 
                     't10k-labels-idx1-ubyte.gz']
        urls = ['http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz',
                'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz',
                'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz',
                'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz']
        for filename, url in zip(filenames, urls):
            if not(os.path.exists(os.path.join(filepath, filename))):
                urllib.request.urlretrieve(url, os.path.join(filepath, filename))
            with gzip.open(
        if not(os.path.exists(os.path.join(filepath, filename))):
            urllib.request.urlretrieve('http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz', os.path.join(filepath, filename))
        with gzip.open(os.path.join(filepath, filename), 'rb') as F:
            (x_train, y_train), (x_validate, y_validate), (x_test, y_test) = pickle.load(F, encoding='latin1')
        x = list(x_train) + list(x_validate) + list(x_test)
        y = list(y_train) + list(y_validate) + list(y_test)
        for i, yy in zip(range(len(y)), y):
            v = np.zeros((1, 10))
            v[0][yy] = 1
            y[i] = v
        x = [torch.from_numpy(xx).squeeze().to(device) for xx in x]
        y = [torch.from_numpy(yy).squeeze().to(device) for yy in y]
        self.training_batches = []
        self.test_batches = []
        self.training_index = 0
        self.test_index = 0
        for batch in range(self.n_batches_train):
            self.training_batches.append(
                [[torch.stack(x[batch_size*batch:batch_size*(batch+1)], dim=0).float(),
                  torch.stack(y[batch_size*batch:batch_size*(batch+1)], dim=0).float()],
                  batch])
        for batch in range(self.n_batches_train, self.n_batches_train+self.n_batches_test):
            self.test_batches.append(
                [[torch.stack(x[batch_size*batch:batch_size*(batch+1)], dim=0).float(),
                  torch.stack(y[batch_size*batch:batch_size*(batch+1)], dim=0).float()],
                  batch])
    
    def get_training_batch(self):
        rv = self.training_batches[self.training_index]
        self.training_index = (self.training_index+1)%(self.n_batches_train)
        return rv
    
    def get_test_batch(self):
        rv = self.test_batches[self.test_index]
        self.test_index = (self.test_index+1)%(self.n_batches_test)
        return rv
            
            
            