from torch.utils.data import DataLoader
import dataloader as dt
import transforms as tfms

dataloaders = {
    'train': DataLoader(dt.dataloader(path='E:/Parse2022/train'))
}

for inputs, outputs in dataloaders['train']:
    print('inputs.shape:', inputs.shape)
    print('outputs.shape', outputs.shape)