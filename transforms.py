import random
import numpy as np
import torch
from skimage.transform import rotate
import matplotlib.pyplot as plt

class Compose:
    def __init__(self, transforms=None):
        self.transforms = transforms

    def __call__(self, sample):
        for transform in self.transforms:
            sample = transform(sample)

        return sample

# Z 정규화
class ZNormalizationIntensity:
    def __init__(self, mode='train'):
        if mode not in ['train', 'test']:
            raise ValueError(f"Argument 'mode' must be 'train' or 'test'. Received {mode}")
        self.mode = mode

    def __call__(self, sample):
        if self.mode == 'train':
            img = sample['input']
            mean, std = img.mean(), img.std()
            if std == 0:
                return sample
            img -= mean
            img /= std
            sample['input'] = img
            print(sample)
            return sample

# To 텐서
class ToTensor:
    def __init__(self, mode='train'):
        if mode not in ['train', 'test']:
            raise ValueError(f"Argument 'mode' must be 'train' or 'test'. Received {mode}")
        self.mode = mode

    def __call__(self, sample):
        if self.mode == 'train':
            img, mask = sample['input'], sample['target']
            img = np.transpose(img, axes=[0, 1, 2])
            mask = np.transpose(mask, axes=[0, 1, 2])
            img = torch.from_numpy(img).float()
            mask = torch.from_numpy(mask).float()
            sample['input'], sample['target'] = img, mask

        else:  # if self.mode == 'test'
            img = sample['input']
            img = np.transpose(img, axes=[3, 0, 1, 2])
            img = torch.from_numpy(img).float()
            sample['input'] = img

        return sample


# 3D 크롭
class RandomCrop3D():
    '''
        volume_3d = torch.rand(3, 100, 100, 100)  # volume_3d.shape[0]은 무조건 1로 하기 때문에 상관없음
        rand_crop = RandomCrop3D(volume_3d.shape, (64, 64, 64))
        rand_crop(volume_3d)
    '''
    def __init__(self, img_sz, crop_sz, p=0.5):
        h, w, d = img_sz  # 원래 c,h,w,d = img_sz 이지만, c을 1로 고정했기 때문에 제거함.
        assert (h, w, d) > crop_sz
        self.img_sz = tuple((h, w, d))
        self.crop_sz = tuple(crop_sz)
        self.p = p

    def __call__(self, sample):
        print('@@@@@@@@@@ Crop 3D @@@@@@@@@@')
        if random.random() < self.p:
            slice_hwd = [self._get_slice(i, k) for i, k in zip(self.img_sz, self.crop_sz)]
            return self._crop(sample, *slice_hwd)
        return sample

    @staticmethod
    def _get_slice(sz, crop_sz):
        try:
            lower_bound = torch.randint(sz - crop_sz, (1,)).item()
            print('lower_bount, loswer_bount + crop_sz: ', lower_bound, lower_bound+crop_sz)
            return lower_bound, lower_bound + crop_sz
        except:
            return (None, None)

    @staticmethod
    def _crop(sample, slice_h, slice_w, slice_d):
        img = sample['input'].squeeze()
        mask = sample['target'].squeeze()
        print(img.shape)
        print(mask.shape)
        img = img[slice_h[0]:slice_h[1], slice_w[0]:slice_w[1], slice_d[0]:slice_d[1]]
        mask = mask[slice_h[0]:slice_h[1], slice_w[0]:slice_w[1], slice_d[0]:slice_d[1]]
        sample['input'], sample['target'] = img, mask
        return sample
# angle_range=[5, 15]
# 랜덤 3D 로테이션
class RandomRotation:
    def __init__(self, p=0.5, angle_range=[5, 15]):
        self.p = p
        self.angle_range = angle_range

    def __call__(self, sample):
        if random.random() < self.p:
            print('Random', sample['input'].shape)
            print('Random', sample['target'].shape)
            img, mask = sample['input'].unsqueeze(dim=0).numpy(), sample['target'].unsqueeze(dim=0).numpy()

            num_of_seqs = img.shape[-1]
            n_axes = random.randint(1, 3)
            random_axes = random.sample([0, 1, 2], n_axes)

            for axis in random_axes:

                angle = random.randrange(*self.angle_range)
                angle = -angle if random.random() < 0.5 else angle
                print('befor img[:,:,i]', img.shape, img)
                for i in range(num_of_seqs):
                    img[:, :, :, i] = RandomRotation.rotate_3d_along_axis(img[:, :, :, i], angle, axis, 1)
                print('after img[:,:,i]', img.shape, img)
                print('before mask[:,:,i]', mask.shape, mask)
                mask[:, :, :, 0] = RandomRotation.rotate_3d_along_axis(mask[:, :, :, 0], angle, axis, 0)
                print('after', mask.shape, mask)

            plt.imshow(img[0, :, :, 1])
            plt.show()
            plt.imshow(mask[0, :, :, 1])
            plt.show()
            sample['input'], sample['target'] = img.squeeze(), mask.squeeze()
            print('@@@@@@@@@@ RandomRotation @@@@@@@@@@')
            print('sample[input]', sample['input'].shape)
            print('sample[target]', sample['target'].shape)
        return sample

    @staticmethod
    def rotate_3d_along_axis(img, angle, axis, order):
        # print('rotate axis 3d: ', img.shape)
        if axis == 0:
            # print('rotate img axis = 0: ', img.shape)
            rot_img = rotate(img, angle, order=order, preserve_range=True)
            # print('result axis = 0:', rot_img.shape)
        if axis == 1:
            # print('rotate img axis = 1: ', img.shape)
            rot_img = np.transpose(img, axes=(1, 2, 0))
            # print('rotate rot_img axis = 1: ', rot_img.shape)
            rot_img = rotate(rot_img, angle, order=order, preserve_range=True)
            # print('result axis = 1:', rot_img.shape)
            rot_img = np.transpose(rot_img, axes=(2, 0, 1))

        if axis == 2:
            # print('rotate img axis = 2: ', img.shape)
            rot_img = np.transpose(img, axes=(2, 0, 1))
            # print('rotate rot_img axis = 2: ', rot_img.shape)
            rot_img = rotate(rot_img, angle, order=order, preserve_range=True)
            # print('result axis = 2:', rot_img.shape)
            rot_img = np.transpose(rot_img, axes=(1, 2, 0))

        # print(rot_img.shape)
        return rot_img