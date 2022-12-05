import torch
import torch.nn as nn
from torch.nn import functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.norm = nn.InstanceNorm3d(out_channels, affine=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = F.relu(x, inplace=True)

        return x

class ConvTransBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.convTransposed = nn.ConvTranspose3d(
            in_channels=in_channels,
            out_channels=out_channels,
            bias=False,
            **kwargs,
        )

        # self.norm = nn.InstanceNorm3d(out_channels, affine=True)

    def forward(self, x):
        x = self.convTransposed(x)
        # x = self.norm(x)
        # x = F.relu(x, inplace=True)

        return x

class SEBlock(nn.Module):
    class SEWeights(nn.Module):
        def __init__(self, in_channels, reduction=4):
            super().__init__()
            self.conv1 = nn.Conv3d(in_channels, in_channels // reduction, kernel_size=1, stride=1, padding=0, bias=True)
            self.conv2 = nn.Conv3d(in_channels // reduction, in_channels, kernel_size=1, stride=1, padding=0, bias=True)

        def forward(self, x):
            b, c, d, h, w = x.size()
            out = torch.mean(x.view(b, c, -1), dim=-1).view(b, c, 1, 1, 1)  # output_shape: in_channels x (1, 1, 1)
            out = F.relu(self.conv1(out))
            out = self.conv2(out)
            return out

    def __init__(self, in_channels, reduction=2):
        super().__init__()
        self.norm = nn.InstanceNorm3d(in_channels, affine=False)
        self.gamma = self.SEWeights(in_channels, reduction)
        self.beta = self.SEWeights(in_channels, reduction)

    def forward(self, x):
        gamma = torch.sigmoid(self.gamma(x))
        beta = torch.tanh(self.beta(x))
        x = self.norm(x)
        return gamma * x + beta

class SEConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, reduction=32):
        super().__init__()

        self.conv1 = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.norm1 = nn.InstanceNorm3d(out_channels, affine=True)
        self.conv2 = nn.Conv3d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.norm2 = nn.InstanceNorm3d(out_channels, affine=True)
        self.senorm = SEBlock(in_channels=out_channels, reduction=reduction)

        if in_channels != out_channels:
            self.projection = nn.Sequential(
                ConvBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=0),
                nn.InstanceNorm3d(out_channels, affine=True)
            )
        else:
            self.projection = nn.Sequential()

    def forward(self, x):
        x = F.relu(self.norm1(self.conv1(x)), inplace=True)
        x = self.norm2(self.conv2(x))
        x = self.senorm(x)
        x = F.relu(x, inplace=True)

        return x

class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale=2, reduction=2):
        super().__init__()
        self.scale = scale
        self.conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.norm = nn.InstanceNorm3d(out_channels, affine=True)
        self.senorm = SEBlock(in_channels=out_channels, reduction=reduction)
        self.upsample = nn.Upsample(scale_factor=scale, mode='trilinear', align_corners=True)

    def forward(self, x):
        x = F.relu(self.norm(self.conv(x)), inplace=True)
        x = self.upsample(x)
        return x

class UNet(nn.Module):
    def __init__(self, in_channels=1, n_cls=1, n_filters=16, reduction=16, deep_supervision=True):
        super().__init__()

        assert n_filters >= reduction

        self.n_cls = n_cls
        self.deep_supervision = deep_supervision

        ## encoder
        self.left1 = SEConvBlock(in_channels=in_channels, out_channels=n_filters, reduction=reduction)
        self.left2 = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=2),
            SEConvBlock(in_channels=n_filters, out_channels=n_filters * 2, reduction=reduction)
        )
        self.left3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=2),
            SEConvBlock(in_channels=n_filters * 2, out_channels=n_filters * 4, reduction=reduction)
        )

        ## center
        self.center = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=2),
            SEConvBlock(in_channels=n_filters * 4, out_channels=n_filters * 8, reduction=reduction)
        )

        ## decoder
        self.upconv3 = ConvTransBlock(
            in_channels=n_filters * 8,
            out_channels=n_filters * 4,
            kernel_size=4,
            stride=2,
            padding=1,
            output_padding=1,
        )
        self.right3 = SEConvBlock(in_channels=n_filters * 8, out_channels=n_filters * 4, reduction=reduction)

        self.upconv2 = ConvTransBlock(
            in_channels=n_filters * 4,
            out_channels=n_filters * 2,
            kernel_size=4,
            stride=2,
            padding=1,
            output_padding=0,
        )
        self.right2 = SEConvBlock(in_channels=n_filters * 4, out_channels=n_filters * 2, reduction=reduction)

        self.upconv1 = ConvTransBlock(
            in_channels=n_filters * 2,
            out_channels=n_filters * 1,
            kernel_size=4,
            stride=2,
            padding=1,
            output_padding=0,
        )
        self.right1 = SEConvBlock(in_channels=n_filters * 2, out_channels=n_filters * 1, reduction=reduction)

        if self.deep_supervision:
            ## upsample
            self.upsample2 = UpsampleBlock(in_channels=n_filters * 4, out_channels=n_filters * 1, scale=4,
                                           reduction=reduction)
            self.upsample1 = UpsampleBlock(in_channels=n_filters * 2, out_channels=n_filters * 1, scale=2,
                                           reduction=reduction)

            self.score = nn.Sequential(
                nn.Conv3d(n_filters * 3, n_filters, kernel_size=3, stride=1, padding=1, bias=False),
                nn.ELU(inplace=True),
                nn.Conv3d(n_filters, self.n_cls, kernel_size=1, stride=1, padding=0, bias=False),
            )
        else:
            self.score = nn.Conv3d(n_filters, self.n_cls, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        left1 = self.left1(x)
        print('left3 : {}'.format(left1.shape))
        left2 = self.left2(left1)
        print('left3 : {}'.format(left2.shape))
        left3 = self.left3(left2)
        print('left3 : {}'.format(left3.shape))
        center = self.center(left3)
        print('center size : {}'.format(center.shape))
        if self.deep_supervision:
            print('self.upconv3(center) size : {}'.format(self.upconv3(center).shape))
            x = self.right3(torch.cat([self.upconv3(center), left3], 1))
            print('x size : {}'.format(x.shape))
            up2 = self.upsample2(x)
            print('up2 size : {}'.format(up2.shape))
            x = self.right2(torch.cat([self.upconv2(x), left2], 1))
            print('x : {}')
            up1 = self.upsample1(x)
            x = self.right1(torch.cat([self.upconv1(x), left1], 1))
            hypercol = torch.cat([x, up1, up2], dim=1)  ## n_filters * 3, 160, 160, 160
            x = self.score(hypercol)
        else:
            x = self.right3(torch.cat([self.upconv3(center), left3], 1))
            x = self.right2(torch.cat([self.upconv2(x), left2], 1))
            x = self.right1(torch.cat([self.upconv1(x), left1], 1))
            x = self.score(x)

        if self.n_cls == 1:
            return torch.sigmoid(x)
        else:
            return F.softmax(x, dim=1)

'''
import nibabel as nib
import torchvision.datasets
import PIL.Image as Image

epi_img = nib.load('dataset/train/PA000005/image/PA000005.nii')
epi_img_data = epi_img.get_fdata()
img3 = Image.fromarray(epi_img_data.astype('uint8'), 'RGB')
b=torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

import matplotlib.pyplot as plt
test = epi_img_data[:,:,]

for i in range(10):
    test = epi_img_data[:,:,i]
    plt.imshow(test)
    plt.show()
'''
# https://discuss.pytorch.org/t/how-to-load-nib-to-pytorch/40947 <- 참고함
bs = 1
num_epochs = 100
learning_rate = 1e-3
mom = 0.9
import torch
import torch.nn as nn
import numpy as np
# import pandas as pd
import torchvision
import torchvision.transforms as tfms
import torch.utils.data as data
import matplotlib.pyplot as plt
import torch.nn.functional as F
import scipy as sc
import os
import PIL
import PIL.Image as Image
# import seaborn as sns
import warnings
import nibabel as nib  # http://nipy.org/nibabel/gettingstarted.html

import os,sys
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from numpy import *
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torchvision
from torchvision import transforms, datasets, models
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import random
import numpy as np

# [mytransforms.ToPILImage()

class Dataloder_img(data.Dataset):
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir  # dataset/train
        self.names_list = os.listdir(dataset_dir)  # PA000005, PA000016...
        self.images_dir = [os.path.join(dataset_dir, i, 'image') for i in self.names_list]
        self.labels_dir = [os.path.join(dataset_dir, i, 'label') for i in self.names_list]
        self.transforms = tfms.Compose([tfms.Resize((256, 256)),
                                        tfms.ToTensor()])
        # self.files = [os.path.join(dataset_dir, i, 'image') for i in os.listdir(dataset_dir)] # dataset/train/name/image
        # self.lables = [os.path.join(dataset_dir, i, 'label') for i in os.listdir(dataset_dir)] # dataset/train/name/label

    def __len__(self):
        return len(self.names_list)

    def __getitem__(self, idx):
        img_name = self.images_dir[idx]
        label_name = self.labels_dir[idx]
        image_set = None
        label_set = None

        # load image.nii
        img = nib.load(os.path.join(img_name, self.names_list[idx]+'.nii'))  # !Image.open(os.path.join(self.root_dir,img_name))
        # change to numpy0
        img = np.array(img.get_fdata())
        for i in img.T[:,:,:]:
            img = Image.fromarray(i.astype('uint8'))
            if self.transforms:
                img = self.transforms(img)
                if image_set == None:
                    image_set = torch.tensor(img)
                else:
                    image_set = torch.cat((image_set, img), dim=0)
                    if len(image_set) == 128:
                        break
            else:
                img = tfms.Compose([tfms.Resize((256, 256)),
                                    tfms.ToTensor()])(img)
                if image_set == None:
                    image_set = torch.tensor(img)
                else: image_set = torch.cat((image_set, img), dim=0)


        # load label.nii
        label = nib.load(os.path.join(label_name, self.names_list[idx]+'.nii'))
        # change to numpy0
        labels = np.array(label.dataobj)
        for i in labels.T[:,:,:]:
            labels = Image.fromarray(i.astype('uint8'))
            if self.transforms:
                labels = self.transforms(labels)
                if label_set == None:
                    label_set = torch.tensor(labels)
                else:
                    label_set = torch.cat((label_set, labels), dim=0)
                    if len(label_set) == 128:
                        break
            else:
                labels = tfms.Compose([tfms.Resize((256, 256)),
                                       tfms.ToTensor()])(labels)
                if label_set == None:
                    label_set = torch.tensor(labels)
                else: label_set = torch.cat((label_set, labels), dim=0)
        image_set = image_set.unsqueeze(dim=0).unsqueeze(dim=0)
        label_set = label_set.unsqueeze(dim=0).unsqueeze(dim=0)
        print(image_set.shape)
        print(label_set.shape)

        return image_set, label_set
        # change to PIL
        # img = Image.fromarray(img.astype('uint8'), 'RGB')

        # print(img.size)

        # label = nib.load(os.path.join(label_name, self.names_list[idx]+'.nii'))  # !Image.open(os.path.join(self.seg_dir,label_name))
        # change to numpy
        # label = np.array(label.get_fdata())
        # change to PIL
        # label = Image.fromarray(label.astype('uint8'), 'RGB')

        # print(label.size)
        #
        # if self.transforms:
        #     img = self.transforms(img)
        #     label = self.transforms(label)
        #     return img, label
        # else:
        #     return img, label

full_dataset = Dataloder_img(dataset_dir='E:/Parse2022/dataset/train')  #
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
train_loader = data.DataLoader(train_dataset, shuffle=False, batch_size=bs)
val_loader = data.DataLoader(val_dataset, shuffle=False, batch_size=bs)

def L1_loss(pred, target):
    loss = torch.mean(torch.abs(pred - target))
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)
    return loss
def L2_loss(pred, target):
    loss = torch.mean(torch.pow((pred - target), 2))
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)
    return loss
def HtoA(x):
    x=x.view(num_class, -1)
    return x

from collections import defaultdict
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy

class CosineAnnealingWarmUpRestarts(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, T_up=0, gamma=1., last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError("Expected positive integer T_up, but got {}".format(T_up))
        self.T_0 = T_0
        self.T_mult = T_mult
        self.base_eta_max = eta_max
        self.eta_max = eta_max
        self.T_up = T_up
        self.T_i = T_0
        self.gamma = gamma
        self.cycle = 0
        self.T_cur = last_epoch
        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_up:
            return [(self.eta_max - base_lr) * self.T_cur / self.T_up + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.eta_max - base_lr) * (
                        1 + math.cos(math.pi * (self.T_cur - self.T_up) / (self.T_i - self.T_up))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch

        self.eta_max = self.base_eta_max * (self.gamma ** self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

device_txt = "cuda:1"
device = torch.device(device_txt if torch.cuda.is_available() else "cpu")
num_class = 19

if __name__ == "__main__":
    model = UNet(1).to(device)
    num_epochs = 1000
    # optimizer = optim.Adam(model.parameters(), lr=1e-5)
    optimizer = optim.Adam(model.parameters(), lr=1e-10)
    scheduler = CosineAnnealingWarmUpRestarts(optimizer=optimizer, T_0=150, T_mult=1, eta_max=1e-4, T_up=10, gamma=0.5)
    print("****************************GPU : ", device)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_train_loss = 1e10
    best_val_loss = 1e10
    valtest = 10
    train_losses = []
    val_losses = []
    test_losses = []
    train_loss_ = 0
    val_loss_ = 0
    test_loss_ = 0


    for epoch in range(num_epochs):
        print('========================' * 9)
        print('Epoch {}/{}, learning_rate {}'.format(epoch, num_epochs - 1, scheduler.get_lr()))
        print('------------------------' * 9)
        now = time.time()

        uu= ['train', 'val', 'test'] if (epoch + 1) % valtest == 0 else ['train']

        for phase in uu:
            # since = time.time()
            if phase == 'train': scheduler.step(); model.train()  # Set model to training mode
            else: model.eval()  # Set model to evaluate mode

            metrics = defaultdict(float) # 성능 값 중첩
            epoch_samples = 0

            for inputs, labels in train_dataset:
                print('a')
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                # forward
                with torch.set_grad_enabled(phase == 'train'):
                    # Forward computation
                    outputs = model(inputs) #
                    #outputs = torch.sigmoid(outputs)
                    LOSS = L2_loss(outputs, labels)
                    metrics['Jointloss'] += LOSS
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        LOSS.backward()
                        optimizer.step()
                # # statistics
                # if num_ % 10000 == 0:
                #     plt.title("H=200, W=160")
                #     plt.imshow(outputs[0][1].detach().cpu());
                #     plt.show()

                epoch_samples += inputs.size(0)



            # print_metrics(metrics, epoch_samples, phase)

            epoch_Jointloss = metrics['Jointloss'] / epoch_samples
            epoch_Jointloss_cpu = epoch_Jointloss.cpu().detach().numpy()

            if phase == 'train':
                train_loss_ = epoch_Jointloss_cpu
            else:
                if phase == 'val':
                    val_loss_ = epoch_Jointloss_cpu
                else:
                    test_loss_ = epoch_Jointloss_cpu
            train_losses.append(train_loss_)
            val_losses.append(val_loss_)
            test_losses.append(test_loss_)

            print(phase,"Joint loss :", epoch_Jointloss )
            # deep copy the model