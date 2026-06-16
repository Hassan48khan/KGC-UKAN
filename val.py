"""Evaluation for KGC-UKAN.

In-domain:
    python val.py --name kgc_ukan --dataset busi --data_dir inputs \
        --input_w 512 --input_h 512 --aspp_rates 6 12 18

Cross-dataset (train source's checkpoint, evaluate on target's full set):
    python val.py --name kgc_ukan --dataset cvc --data_dir inputs \
        --checkpoint models/kgc_ukan_busi/model_best.pth --eval_all True
"""
import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from config import str2bool
from dataset import MedicalSegDataset, list_image_ids
from metrics import iou_score
from utils import AverageMeter, get_main_output
from kgc_ukan import KGC_UKAN


def parse_eval_args():
    p = argparse.ArgumentParser()
    p.add_argument('--name', default='kgc_ukan')
    p.add_argument('--dataset', default='busi')
    p.add_argument('--data_dir', default='inputs')
    p.add_argument('--checkpoint', default=None, help='explicit checkpoint path; '
                   'defaults to models/<name>/model_best.pth')
    p.add_argument('--img_ext', default='.png')
    p.add_argument('--mask_ext', default='.png')
    p.add_argument('--input_w', default=512, type=int)
    p.add_argument('--input_h', default=512, type=int)
    p.add_argument('--num_classes', default=1, type=int)
    p.add_argument('--input_channels', default=3, type=int)
    p.add_argument('--embed_dims', default=[128, 160, 256], type=int, nargs='+')
    p.add_argument('--aspp_rates', default=[6, 12, 18], type=int, nargs='+')
    p.add_argument('--use_edge', default=True, type=str2bool)
    p.add_argument('--use_pulask', default=True, type=str2bool)
    p.add_argument('--batch_size', default=16, type=int)
    p.add_argument('--num_workers', default=4, type=int)
    p.add_argument('--eval_all', default=False, type=str2bool,
                   help='evaluate on the entire dataset (used for cross-dataset transfer)')
    p.add_argument('--val_split', default=0.2, type=float)
    p.add_argument('--seed', default=42, type=int)
    return p.parse_args()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    iou_m, dice_m, hd_m, f1_m = (AverageMeter() for _ in range(4))
    for img, mask, _ in loader:
        img, mask = img.to(device), mask.to(device)
        out = get_main_output(model(img))
        iou, dice, hd95_, f1 = iou_score(out, mask)
        iou_m.update(iou, img.size(0)); dice_m.update(dice, img.size(0))
        hd_m.update(hd95_, img.size(0)); f1_m.update(f1, img.size(0))
    return iou_m.avg, dice_m.avg, hd_m.avg, f1_m.avg


def main():
    args = parse_eval_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    img_dir = os.path.join(args.data_dir, args.dataset, 'images')
    mask_dir = os.path.join(args.data_dir, args.dataset, 'masks')
    ids = list_image_ids(img_dir, args.img_ext)

    if not args.eval_all:
        from sklearn.model_selection import train_test_split
        _, ids = train_test_split(ids, test_size=args.val_split, random_state=args.seed)

    dataset = MedicalSegDataset(ids, img_dir, mask_dir, args.img_ext, args.mask_ext,
                                args.input_w, args.input_h, args.num_classes, augment=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers)

    model = KGC_UKAN(num_classes=args.num_classes, input_channels=args.input_channels,
                     deep_supervision=False, img_size=args.input_w, embed_dims=args.embed_dims,
                     use_edge=args.use_edge, use_pulask=args.use_pulask,
                     aspp_rates=tuple(args.aspp_rates)).to(device)

    ckpt = args.checkpoint or os.path.join('models', args.name, 'model_best.pth')
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state, strict=False)
    print(f'Loaded checkpoint: {ckpt}')

    iou, dice, hd95_, f1 = evaluate(model, loader, device)
    print(f'[{args.dataset}] IoU={iou*100:.2f}  Dice={dice*100:.2f}  '
          f'HD95={hd95_:.2f}  F1={f1*100:.2f}  (N={len(ids)})')


if __name__ == '__main__':
    main()
