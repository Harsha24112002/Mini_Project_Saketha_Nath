#!/usr/bin/env python
# coding: utf-8

# ### Frogner Classification

from torchvision import datasets, transforms
import torch
import torch.nn as nn
import torch.optim as optim
import ot
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
torch.autograd.set_detect_anomaly(True)

transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])

device = "cuda:0"


trainset = datasets.MNIST('.', download=True, train=True, transform=transform)
valset = datasets.MNIST('.', download=True, train=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=256, shuffle=True)

def sinkhorn_fixedCost(batch_a, batch_b, M, reg, stopThr, numItermax):
    """
    Sinkhorn Knopp when Cost matrix is common across datapoints
    """
    batch = batch_a.shape[0]
    classes = batch_a.shape[1]

    u = torch.ones((batch, classes)).to(device=device) / classes
    v = torch.ones((batch, classes)).to(device=device) / classes
    K = torch.empty(M.shape, dtype=M.dtype).to(device=device)
    torch.divide(M.to(device=device), -reg, out=K)
    torch.exp(K, out=K)

    # print(batch_b)
    for torcht in range(numItermax):
        KtransposeU = torch.einsum('ij,bi->bj', K, u).to(device=device,dtype=torch.float32)
        v = torch.divide(batch_b, KtransposeU).to(device=device)
        u = 1. / ((1. / (batch_a+1e-6)) * torch.einsum('ij,bj->bi', K, v)).to(device=device)

    gamma = torch.einsum('bi,ij,bj->bij', u, K, v).to(device=device)
    loss = torch.sum(torch.einsum('ijk,jk->i', gamma, M)).to(device=device)
    return loss, gamma, u

input_size = 784
hidden_sizes = [128, 64]
output_size = 10

model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[1],
                                 output_size),
                      nn.Softmax(dim=1)).to(device=device)
# model.load_state_dict(torch.load("model.pt", map_location=torch.device(device=device)))

optimizer = optim.Adam(model.parameters(),lr=1e-4)
epochs = 20


one2ten = torch.arange(10).reshape((10,1)).to(dtype=torch.float32)
C = torch.cdist(one2ten,one2ten,p=1.0).to(device=device)


def WassersteinLoss(xv,yv,C,reg):
    loss = 0
    for i in range(xv.shape[0]):
        loss += ot.sinkhorn2(xv[i],yv[i],C,reg)
    return loss

criterion = WassersteinLoss


f = open('sinkhornot.txt','w')
losses = []

for e in range(epochs):
    running_loss = 0
    for i, (images, labels) in enumerate(trainloader):
        images = images.view(images.shape[0], -1).to(device=device) 
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, torch.eye(10)[labels].to(device=device),C,0.2)
        loss.backward()
        optimizer.step()
        running_loss += loss.data
        # print(e,i, loss)
    print(e,running_loss,file=f)
    print(e)
    losses.append(running_loss.to(device='cpu').detach().numpy())

plt.plot(np.arange(epochs),losses)
plt.savefig("EPOCHVSLOSSES.png")
torch.save(model.state_dict(),"sinkhornmodel.pt")
preds=0
for i, (images, labels) in enumerate(valloader):
    preds = 0
    for j in range(len(labels)):
        img = images[j].view(1, 784).to(device=device)
        with torch.no_grad():
            ps = model(img)
        preds += not( torch.argmax(ps) == labels[j] )

    print(preds,file=f)


