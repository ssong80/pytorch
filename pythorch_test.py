import itertools
from IPython.display import Image
from IPython import display
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

trn_dataset = datasets.MNIST('../mnist_data/', download=True, train=True,
transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.137,), (0.3081,))
]))

val_dataset = datasets.MNIST('../mnist_data/', download=False, train=True,
transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,),(0.3081,))
]))

batch_size = 64

trn_loader = torch.utils.data.DataLoader(trn_dataset,
batch_size=batch_size, shuffle=True)

val_loader = torch.utils.data.DataLoader(val_dataset,
batch_size=batch_size, shuffle=True)

use_cuda = torch.cuda.is_available()

class CNNClassifier(nn.Module):

    def __init__(self):
        super(CNNClassifier, self).__init__()
        conv1 = nn.Conv2d(1,6,5,1)
        pool1 = nn.MaxPool2d(2)
        conv2 = nn.Conv2d(6,16, 5,1)
        pool2 = nn.MaxPool2d(2)

        self.conv_module = nn.Sequential(
            conv1,
            nn.ReLU(),
            pool1,
            conv2,
            nn.ReLU(),
            pool2
        )

        fc1 = nn.Linear(16*4*4, 120)
        fc2 = nn.Linear(120,84)
        fc3 = nn.Linear(84, 10)

        self.fc_module = nn.Sequential(
            fc1,
            nn.ReLU(),
            fc2,
            nn.ReLU(),
            fc3
        )

        if use_cuda:
            self.conv_module = self.conv_module.cuda()
            self.fc_module = self.fc_module.cuda()

    def forward(self, x):
        out = self.conv_module(x)
        dim = 1
        for d in out.size()[1:]:
            dim = dim *d
        out = out.view(-1, dim)
        out = self.fc_module(out)
        return F.softmax(out, dim=1)
    
cnn = CNNClassifier()

criterion = nn.CrossEntropyLoss()

learning_rate = 1e-3
optimizer = optim.Adam(cnn.parameters(), lr=learning_rate)
num_epochs = 2
num_batches = len(trn_loader)

trn_loss_list = []
val_loss_list = []

for epoch in range(num_epochs):
    trn_loss = 0.0
    for i, data in enumerate(trn_loader):
        x, label = data
        if use_cuda:
            x = x.cuda()
            label = label.cuda()
        optimizer.zero_grad()
        model_output = cnn(x)
        loss = criterion(model_output, label)
        loss.backward()
        optimizer.step()

        trn_loss += loss.item()

        del loss
        del model_output

        if(i+1)% 100 == 0:
            with torch.no_grad():
                val_loss = 0.0
                for j, val in enumerate(val_loader):
                    val_x, val_label = val
                    if use_cuda:
                        val_x = val_x.cuda()
                        val_label = val_label.cuda()
                    val_output = cnn(val_x)
                    v_loss = criterion(val_output, val_label)
                    val_loss += v_loss
            print("epoch: {}/{} | step: {}/{} | trn loss: {:.4f} | val loss: {:.4f}".format(
                epoch+1, num_epochs, i+1, num_batches, trn_loss / 100, val_loss / len(val_loader)
            ))

            trn_loss_list.append(trn_loss/100)
            val_loss_list.append(val_loss/len(val_loader))
            trn_loss = 0.0

with torch.no_grad():
    corr_num = 0
    total_num = 0
    for j, val in enumerate(val_loader):
        val_x, val_label = val
        if use_cuda:
            val_x = val_x.cuda()
            val_label =val_label.cuda()
        val_output = cnn(val_x)
        model_label = val_output.argmax(dim=1)
        corr = val_label[val_label == model_label].size(0)
        corr_num += corr
        total_num += val_label.size(0)

print("acc: {:.2f}".format(corr_num / total_num * 100))

