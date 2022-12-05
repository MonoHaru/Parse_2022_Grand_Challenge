import os
import nibabel as nib
from torch.utils.data import Dataset
import transforms as tfms


class dataloader(Dataset):
    def __init__(self, path='./train', mode='train'):
        self.base_path = path
        self.fn_list = sorted(os.listdir(self.base_path))
        self.mode = mode

    @staticmethod
    def __read_nifti__(path_to_nifti):
        return nib.load(path_to_nifti).get_fdata()

    def __len__(self):
        return len(self.fn_list)

    def __getitem__(self, index):
        sample = dict()
        fn = self.fn_list[index]

        img_fn = os.path.join(self.base_path, fn, 'image', fn + '.nii.gz')
        assert os.path.isfile(img_fn), '현재 img_fn 경로에 파일이 없습니다.'

        if self.mode == 'train':
            label_fn = os.path.join(self.base_path, fn, 'label', fn +  '.nii.gz')
            assert os.path.isfile(img_fn), '현재 label_fn 경로에 파일이 없습니다.'

            img = self.__read_nifti__(img_fn)
            # print('img size: ', img.shape)
            label = self.__read_nifti__(label_fn)
            print('*' * 10)
            print(label)
            sample['input'] = img
            sample['target'] = label

        elif self.mode == 'test':
            pass

        transforms = tfms.Compose([
            tfms.ToTensor(),  # 무조건 ToTensor를 처음으로 사용해야 함. 안 그러면 shape가 안 맞음
            tfms.ZNormalizationIntensity(),  # z 정규화
            tfms.RandomRotation(),
            tfms.RandomCrop3D(sample['input'].shape, (512, 512, 120)),
            # 3D 랜덤 크롭, all shape h,w=512,512 and 제일 작은 깊이=228
        ])

        if transforms:
            sample = transforms(sample)

        return sample['input'], sample['target']
