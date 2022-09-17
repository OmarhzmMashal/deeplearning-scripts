import torch
from torch import nn
import torchvision
from torchvision import datasets
from torchvision import transforms
import numpy as np
from tqdm import tqdm
from itertools import cycle

from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import cv2
from numpy import expand_dims
from numpy import moveaxis
import math
from torch.utils.data import DataLoader
from torch.autograd import Function
from sklearn.decomposition import PCA


path = "/home/omar/Desktop/images"
device='cuda'
folder = path + "/mnist-m"

# UNSUPERVISED ADVERSRIAL DOMAIN ADAPTATION
# SOURCE DOMAIN -> MNISE
# TARGET DOMAIN -> MNIST-M
#images_channel_first = moveaxis(images, 3, 1)

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        img = cv2.resize(img, (28,28), interpolation=cv2.INTER_AREA)
        if img is not None:
            images.append(img)
    return np.array(images)

class GradientReversalFn(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x

    @staticmethod
    def backward(ctx, grad_output):
        output = - ctx.alpha * grad_output
        return output, None

#im = torch.ones((1,1,28,28))
#imm = convs(im)
#imm.shape


image_size=28
class DACNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, padding=1, stride=1),
            nn.BatchNorm2d(64), nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=5, padding=1, stride=1),
            nn.BatchNorm2d(128), nn.ReLU(True), nn.MaxPool2d(2),
            nn.Dropout2d(),
        )

        self.cnn_num_features = 128 * 5 * 5

        self.class_clsf = nn.Sequential(
            nn.Linear(self.cnn_num_features, 128),
            nn.BatchNorm1d(128), nn.Dropout2d(), nn.ReLU(True),
            nn.Linear(128,64), nn.BatchNorm1d(64), nn.ReLU(True),
            nn.Linear(64,10),
            nn.LogSoftmax(dim=1),
        )

        self.domain_clsf = nn.Sequential(
           nn.Linear(self.cnn_num_features, 128),
           nn.BatchNorm1d(128),nn.ReLU(True),
           nn.Linear(128, 10), nn.ReLU(True),
           nn.Linear(10, 1),
           nn.Sigmoid()
        )

    def forward(self, x, grl_lambda=1.0):
        #x = x.expand(x.data.shape[0], 3, image_size, image_size)

        features = self.feature_extraction(x)
        features = features.view(-1,self.cnn_num_features)
        features_grl = GradientReversalFn.apply(features, grl_lambda)
        class_pred = self.class_clsf(features)
        domain_pred = self.domain_clsf(features_grl)

        return class_pred, domain_pred, features

model = DACNN().to(device)

# loading target dataset, target domain -> 0
target_images = load_images_from_folder(folder)
target_images = np.moveaxis(target_images, 3,1).astype('float64')

# batch it
batch_target = 16
batch_source = 128

target_dataset=[]
for i in range(len(target_images)):
    target_images[i][0] = target_images[i][0]/255
    target_dataset.append([target_images[i][0].reshape(1,28,28), 0])
target_loader =  DataLoader(target_dataset, batch_size=batch_target, shuffle = True)
# Download the MNIST Dataset
source_dataset = datasets.MNIST(root = "./data", train = True, transform = transforms.ToTensor(), download=True)
source_loader = DataLoader(dataset = source_dataset, batch_size = batch_source, shuffle = True)

loss_fn_class = torch.nn.CrossEntropyLoss()
loss_fn_domain = torch.nn.BCELoss()

epochs=10
lr= 1e-2
optimizer = torch.optim.SGD(model.parameters(), lr =lr)
domain_loss = []
class_loss = []
class_acc = []
num_batches_source = len(source_loader)/batch_source
num_batches_target= len(target_loader)/batch_target

correct=0
total=1

print(f'num of source batches = {num_batches_source}')
print(f'num of source batches = {num_batches_target}')

for epoch in tqdm(range(epochs)):
    i=0

    for (x_source, y_source), (x_target, _) in zip(source_loader, cycle(target_loader)) if len(source_loader) > len(target_loader) else zip(cycle(source_loader), target_loader):

        p = float(i + epoch * num_batches_source) / (epochs * num_batches_source)
        grl_lambda = (2. / (1. + np.exp(-10*p))-1)
        i+=1

        # Train on source domain #
        x_source = x_source.to(device)
        y_source = y_source.to(device)
        current_batches = x_source.shape[0]
        domain_source = torch.zeros(current_batches, dtype=torch.float64).to(device) # source domain label = 0

        # predict
        pred_class_source, pred_domain_source, _ = model(x_source, grl_lambda)

        # calc loss
        loss_source_label = loss_fn_class(pred_class_source, y_source)
        loss_source_domain = loss_fn_domain(pred_domain_source.view(-1), domain_source.float())
        loss_source = loss_source_label + loss_source_domain

        optimizer.zero_grad()
        loss_source.backward(retain_graph=True)
        optimizer.step()

        _, predicted = torch.max(pred_class_source.data, 1)
        total += y_source.shape[0]
        correct += (predicted == y_source).sum().item()

        # Train on target domain #
        x_target = x_target.float().to(device)
        current_batches = x_target.shape[0]
        domain_target = torch.ones(current_batches, dtype=torch.float64).to(device) # target domain label = 1

        # predict
        _, pred_domain_target, _ = model(x_target, grl_lambda)

        # loss
        loss_target_domain = loss_fn_domain(pred_domain_target.view(-1), domain_target.float())
        optimizer.zero_grad()
        loss_target_domain.backward(retain_graph=True)
        optimizer.step()


    with torch.no_grad():

        domain_loss.append((loss_source_domain.cpu().detach().numpy() + loss_target_domain.cpu().detach().numpy())/2)
        class_loss.append(loss_source_label.cpu().detach().numpy())
        class_acc.append(correct/total)

        print(f'source class loss: {loss_source_label} \
                source domain loss {loss_source_domain} \
                target domain loss {loss_target_domain} \
                accuracy sourc {correct/total}')


c=0
for (x_source, y_source), (x_target, _) in zip(source_loader, target_loader):
    x_source = x_source.to(device)
    x_target = x_target.float().to(device)

    _, _,source_features = model(x_source, grl_lambda)
    _, _,target_features = model(x_target, grl_lambda)

    plt.scatter(source_features[:,0].cpu().detach().numpy(),source_features[:,1].cpu().detach().numpy(), c='r')
    plt.scatter(target_features[:,0].cpu().detach().numpy(),target_features[:,1].cpu().detach().numpy(), c='b')

    c+=1
    if c == 5:
        plt.show()
        break

#for i in range(32):
    #plt.subplot(8,4,i+1)
    #plt.imshow(features[0][i].cpu().detach().numpy())
plt.plot([i for i in range(epochs)],domain_loss)
plt.plot([i for i in range(epochs)],class_loss)
plt.show()
plt.plot([i for i in range(epochs)],class_acc)
plt.show()
