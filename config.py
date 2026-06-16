"""Configuration and argument parsing for KGC-UKAN training/validation."""
import argparse
import os
import yaml


def str2bool(v):
    if isinstance(v, bool):
        return v
    return v.lower() in ('true', '1', 'yes', 'y', 't')


def parse_args():
    parser = argparse.ArgumentParser(description='KGC-UKAN training')

    # experiment
    parser.add_argument('--name', default='kgc_ukan', help='experiment name')
    parser.add_argument('--seed', default=42, type=int)

    # data
    parser.add_argument('--dataset', default='busi', help='dataset name (folder under --data_dir)')
    parser.add_argument('--data_dir', default='inputs', help='root data directory')
    parser.add_argument('--img_ext', default='.png')
    parser.add_argument('--mask_ext', default='.png')
    parser.add_argument('--input_w', default=512, type=int)
    parser.add_argument('--input_h', default=512, type=int)
    parser.add_argument('--num_classes', default=1, type=int)
    parser.add_argument('--input_channels', default=3, type=int)
    parser.add_argument('--val_split', default=0.2, type=float)

    # model
    parser.add_argument('--embed_dims', default=[128, 160, 256], type=int, nargs='+')
    parser.add_argument('--use_edge', default=True, type=str2bool, help='enable SAKE edge branch')
    parser.add_argument('--use_pulask', default=True, type=str2bool, help='enable PU-LASk skips')
    parser.add_argument('--deep_supervision', default=True, type=str2bool)
    parser.add_argument('--no_kan', default=False, type=str2bool, help='replace KANLinear with nn.Linear')
    parser.add_argument('--aspp_rates', default=[6, 12, 18], type=int, nargs='+',
                        help='use (2,4,6) for 256x256, (6,12,18) for 512x512')
    parser.add_argument('--drop_path_rate', default=0.0, type=float)

    # training
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--min_lr', default=1e-5, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--num_workers', default=4, type=int)

    # loss weights
    parser.add_argument('--boundary_weight', default=0.2, type=float)
    parser.add_argument('--uncertainty_loss_weight', default=0.1, type=float)
    parser.add_argument('--kan_reg_weight', default=0.0, type=float,
                        help='weight on KAN spline regularization (set >0 to enable)')
    parser.add_argument('--use_boundary', default=True, type=str2bool)
    parser.add_argument('--use_uncertainty_loss', default=True, type=str2bool)

    args = parser.parse_args()
    return args


def save_config(args, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'config.yml'), 'w') as f:
        yaml.dump(vars(args), f, default_flow_style=False)
