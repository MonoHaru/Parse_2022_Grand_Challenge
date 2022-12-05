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

# [mytransforms.ToPILImage()

class Dataloder_img(data.Dataset):
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir  # dataset/train
        self.names_list = os.listdir(dataset_dir)  # PA000005, PA000016...
        self.images_dir = [os.path.join(dataset_dir, i, 'image') for i in self.names_list]
        self.labels_dir = [os.path.join(dataset_dir, i, 'label') for i in self.names_list]
        self.transforms = tfms.Compose([tfms.Resize((512, 512)),
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
        print(img.shape)
        for i in img.T[:,:,:]:
            img = Image.fromarray(i.astype('uint8'))
            if self.transforms:
                img = self.transforms(img)
                if image_set == None:
                    image_set = torch.tensor(img)
                else: image_set = torch.cat((image_set, img), dim=0)
            else:
                img = tfms.Compose([tfms.Resize((512, 512)),
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
                else: label_set = torch.cat((label_set, labels), dim=0)
            else:
                labels = tfms.Compose([tfms.Resize((512, 512)),
                                       tfms.ToTensor()])(labels)
                if label_set == None:
                    label_set = torch.tensor(labels)
                else: label_set = torch.cat((label_set, labels), dim=0)

        image_set = image_set.unsqueeze(dim=0).unsqueeze(dim=0)
        label_set = label_set.unsqueeze(dim=0).unsqueeze(dim=0)

        # print(image_set.shape)
        # print(label_set.shape)
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

test_img, test_lb = next(iter(full_dataset))
print(test_img[0].shape)
plt.imshow(test_img[0])
plt.show()

for i in train_loader:
    print(i[0].shape)
    print(i[1].shape)


# transforms = tfms.Compose([
#     tfms.ToTensor(),  # 무조건 ToTensor를 처음으로 사용해야 함. 안 그러면 shape가 안 맞음
#     tfms.ZNormalizationIntensity(),  # z 정규화
#     tfms.RandomCrop3D(sample['input'].shape, (512, 512, 228)),  # 3D 랜덤 크롭, all shape h,w=512,512 and 제일 작은 깊이=228
# ])
#
# args = dict()
# args['base_path'] = './train'
#
# fn_list = sorted(os.listdir(args['base_path']))
# fn = random.choice(fn_list)
# print(fn)
#
# img_fn = os.path.join(args['base_path'], fn, 'image', fn + '.nii.gz')
# label_fn = os.path.join(args['base_path'], fn, 'label', fn + '.nii.gz')
#
# img = read_nifti(img_fn)
# label = read_nifti(label_fn)
#
# transforms =  tio.Compose([
#     tio.ZNormalization(p=1),
#     tio.RandomFlip(
#         axes=(0, 1, 2),
#         p=0,
#     ),
#     tio.RandomNoise(
#         mean=0,
#         std=(0, 0.25),
#         p=0,
#     ),
#     tio.RandomAffine(
#         scales=0.0,
#         degrees=15,
#         translation=5,
#         isotropic=True,
#         center='image' ## rotation around ccenter of image
#     )
# ])
#
# img_t = torch.as_tensor(img, dtype=torch.float, device='cpu').unsqueeze(dim=0)
# label_t = torch.as_tensor(label, dtype=torch.float, device='cpu').unsqueeze(dim=0)
#
# subject = tio.Subject(
#     image=tio.ScalarImage(tensor=img_t),
#     mask=tio.LabelMap(tensor=label_t)
# )
#
# result = transforms(subject)
#
# tf_img = result['image'][tio.DATA].squeeze().detach().cpu().numpy()
# tf_label = result['mask'][tio.DATA].squeeze().detach().cpu().numpy()
#
# def vis_t(s):
#     plt.figure(figsize=(14, 10))
#     plt.imshow(tf_img[:, :, 0], cmap='gray')
#     plt.show()