import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import argparse
import torch
import pandas as pd

import src.transforms as T
import src.loss as losses
import src.metrics as metrics
import src.MODELS as models
from src import (
    ParseDataset,
    ModelTrainer,
    seed_everything,
    str2bool,
)

_CROP_SZ_ = (320, 320, 200)


def get_train_transforms():
    return T.Compose(
        [
            # T.CenterCrop(crop_sz=_CROP_SZ_),
            T.RandCropNearCenter(p=0.5, crop_sz=_CROP_SZ_, offset=(20, 20, 10)),  # 먼저 위치해야 함
            T.RandomRotation(p=0.5, angle_range=[5, 15]),
            T.Mirroring(p=0.5),
            T.NormalizeIntensity(),
            T.ToTensor(),
        ]
    )


def get_valid_transforms():
    return T.Compose(
        [
            T.CenterCrop(crop_sz=_CROP_SZ_),
            T.NormalizeIntensity(),
            T.ToTensor(),
        ]
    )


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    seed_everything(42)

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default=device)
    parser.add_argument('--parallel', type=str2bool, default=False)

    parser.add_argument('--base_folder', type=str, default='./data')
    parser.add_argument('--save_folder', type=str, default='./checkpoint')

    parser.add_argument('--label_fn', type=str, default='./data/data_split.csv')
    parser.add_argument('--kfold_idx', type=int, choices=[0, 1, 2, 3, 4], default=0)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--use_wandb', type=str2bool, default=True)

    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)

    parser.add_argument('--coord_conv', type=str2bool, default=False)
    parser.add_argument('--ds_conv', type=str2bool, default=False)
    parser.add_argument('--gsop', type=str2bool, default=False)
    parser.add_argument('--self_distillation', type=str2bool, choices=[True, False], default=False)
    parser.add_argument('--deep_supervision', type=str, choices=['sum', 'concat'], default=False)
    parser.add_argument('--att_type', type=str, choices=['SE', 'scSE', 'PE', 'CBAM'], default=False)
    parser.add_argument('--reduction_ratio', type=int, default=4)

    parser.add_argument('--comments', type=str, default=None)

    args = parser.parse_args()

    assert os.path.isdir(args.base_folder), 'wrong path'

    print('=' * 50)
    print('[info msg] arguments')
    for key, value in vars(args).items():
        print(key, ":", value)

    label_df = pd.read_csv(args.label_fn)
    train_fns = label_df[(label_df['kfold_idx'] == args.kfold_idx) & (label_df['mode'] == 'train')]['fn'].values
    valid_fns = label_df[(label_df['kfold_idx'] == args.kfold_idx) & (label_df['mode'] == 'valid')]['fn'].values

    train_dataset = ParseDataset(
        base_path=os.path.join(args.base_folder, 'train'),
        fns=train_fns,
        transforms=get_train_transforms(),
    )

    valid_dataset = ParseDataset(
        base_path=os.path.join(args.base_folder, 'train'),
        fns=valid_fns,
        transforms=get_valid_transforms(),
    )

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = models.AttentionUNet(
        in_channels=1,
        n_cls=1,
        n_filters=8,
        coord_conv=args.coord_conv,
        ds_conv=args.ds_conv,
        gsop=args.gsop,
        att_type=args.att_type,
        reduction_ratio=args.reduction_ratio,
        deep_supervision=args.deep_supervision,
        self_distillation=args.self_distillation,
    )

    loss = losses.Dice_and_FocalLoss()
    metric = metrics.dice
    optimizer = torch.optim.Adam(model.parameters(), args.lr, betas=(0.9, 0.99))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50)

    trainer = ModelTrainer(
        model=model,
        train_loader=train_data_loader,
        valid_loader=valid_data_loader,
        loss_func=loss,
        metric_func=metric,
        optimizer=optimizer,
        device=args.device,
        save_dir=args.save_folder,
        mode='max',
        scheduler=scheduler,
        num_epochs=args.epochs,
        parallel=args.parallel,
        # snapshot_period=scheduler.T_0,
        snapshot_period=None,
        use_wandb=args.use_wandb,
    )

    if args.use_wandb:
        trainer.initWandb(
            project_name='parse2022',
            run_name=args.comments,
            args=args,
        )

    trainer.train()

    with open(os.path.join(trainer.save_dir, 'config.txt'), 'w') as f:
        for key, value in vars(args).items():
            f.write('{} : {}\n'.format(key, value))


if __name__ == '__main__':
    main()