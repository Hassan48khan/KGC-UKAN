"""Training entry point for KGC-UKAN.

Example
-------
    python train.py --dataset busi --data_dir inputs \
        --input_w 512 --input_h 512 --aspp_rates 6 12 18 \
        --batch_size 16 --epochs 300 --lr 1e-4

For 256x256 inputs use: --input_w 256 --input_h 256 --aspp_rates 2 4 6
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import train_test_split

from config import parse_args, save_config
from dataset import MedicalSegDataset, list_image_ids
from losses import CombinedLoss
from metrics import iou_score
from utils import AverageMeter, set_seed, count_parameters, get_main_output
from kgc_ukan import KGC_UKAN


def build_model(args):
    return KGC_UKAN(
        num_classes=args.num_classes,
        input_channels=args.input_channels,
        deep_supervision=args.deep_supervision,
        img_size=args.input_w,
        embed_dims=args.embed_dims,
        no_kan=args.no_kan,
        drop_path_rate=args.drop_path_rate,
        use_edge=args.use_edge,
        use_pulask=args.use_pulask,
        aspp_rates=tuple(args.aspp_rates),
    )


def train_one_epoch(model, loader, criterion, optimizer, device, args):
    model.train()
    meters = {'loss': AverageMeter(), 'iou': AverageMeter(), 'dice': AverageMeter()}
    for img, mask, _ in loader:
        img, mask = img.to(device), mask.to(device)

        if args.use_pulask and args.use_uncertainty_loss:
            outputs, unc = model(img, return_uncertainty=True)
        else:
            outputs, unc = model(img), None

        loss, _ = criterion(outputs, mask, uncertainty_maps=unc)
        if args.kan_reg_weight > 0:
            loss = loss + args.kan_reg_weight * model.regularization_loss()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        main = get_main_output(outputs)
        iou, dice, _, _ = iou_score(main, mask)
        meters['loss'].update(loss.item(), img.size(0))
        meters['iou'].update(iou, img.size(0))
        meters['dice'].update(dice, img.size(0))
    return {k: m.avg for k, m in meters.items()}


@torch.no_grad()
def validate(model, loader, criterion, device, args):
    model.eval()
    meters = {'loss': AverageMeter(), 'iou': AverageMeter(), 'dice': AverageMeter()}
    for img, mask, _ in loader:
        img, mask = img.to(device), mask.to(device)
        outputs = model(img)                      # eval: single tensor
        loss, _ = criterion(outputs, mask, uncertainty_maps=None)
        main = get_main_output(outputs)
        iou, dice, _, _ = iou_score(main, mask)
        meters['loss'].update(loss.item(), img.size(0))
        meters['iou'].update(iou, img.size(0))
        meters['dice'].update(dice, img.size(0))
    return {k: m.avg for k, m in meters.items()}


def main():
    args = parse_args()
    set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    out_dir = os.path.join('models', args.name)
    os.makedirs(out_dir, exist_ok=True)
    save_config(args, out_dir)

    # data
    img_dir = os.path.join(args.data_dir, args.dataset, 'images')
    mask_dir = os.path.join(args.data_dir, args.dataset, 'masks')
    ids = list_image_ids(img_dir, args.img_ext)
    train_ids, val_ids = train_test_split(ids, test_size=args.val_split, random_state=args.seed)

    train_set = MedicalSegDataset(train_ids, img_dir, mask_dir, args.img_ext, args.mask_ext,
                                  args.input_w, args.input_h, args.num_classes, augment=True)
    val_set = MedicalSegDataset(val_ids, img_dir, mask_dir, args.img_ext, args.mask_ext,
                                args.input_w, args.input_h, args.num_classes, augment=False)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)

    # model / loss / optim
    model = build_model(args).to(device)
    print(f'Model: KGC-UKAN | params = {count_parameters(model) / 1e6:.2f} M')

    criterion = CombinedLoss(
        boundary_weight=args.boundary_weight,
        use_boundary=args.use_boundary,
        use_uncertainty_loss=args.use_uncertainty_loss,
        uncertainty_loss_weight=args.uncertainty_loss_weight,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)

    log = {'epoch': [], 'lr': [], 'train_loss': [], 'train_iou': [],
           'val_loss': [], 'val_iou': [], 'val_dice': []}
    best_iou = 0.0

    for epoch in range(args.epochs):
        tr = train_one_epoch(model, train_loader, criterion, optimizer, device, args)
        va = validate(model, val_loader, criterion, device, args)
        scheduler.step()

        lr_now = optimizer.param_groups[0]['lr']
        print(f'Epoch [{epoch + 1}/{args.epochs}] '
              f'train_loss {tr["loss"]:.4f} train_iou {tr["iou"]:.4f} | '
              f'val_loss {va["loss"]:.4f} val_iou {va["iou"]:.4f} val_dice {va["dice"]:.4f}')

        log['epoch'].append(epoch + 1); log['lr'].append(lr_now)
        log['train_loss'].append(tr['loss']); log['train_iou'].append(tr['iou'])
        log['val_loss'].append(va['loss']); log['val_iou'].append(va['iou'])
        log['val_dice'].append(va['dice'])
        pd.DataFrame(log).to_csv(os.path.join(out_dir, 'log.csv'), index=False)

        if va['iou'] > best_iou:
            best_iou = va['iou']
            torch.save(model.state_dict(), os.path.join(out_dir, 'model_best.pth'))
            print(f'  -> saved best (val_iou={best_iou:.4f})')

    torch.save(model.state_dict(), os.path.join(out_dir, 'model_last.pth'))
    print(f'Done. Best val IoU = {best_iou:.4f}')


if __name__ == '__main__':
    main()
