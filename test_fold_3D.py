import os
import argparse
import time
import random
import logging
import numpy as np

import torch
import torch.optim
from torch.utils.data import DataLoader
from configs.config import get_model_config
from data.BraTS import BraTS2019, BraTS2020
from models.model_3D.swin_unet_3D import SwinUnet_3D_All
from models.model_3D.vitb16_3D import ViTUperNet_3D_All
from utils.predict_fold_3D import validate_softmax
from utils.tools import log_args


parser = argparse.ArgumentParser()

# DataSet Information
parser.add_argument('--root', default='.../your_datasets_file', type=str)
parser.add_argument('--dataset_name', default='BraTS2019', type=str) # BraTS2019, BraTS2020
parser.add_argument('--valid_dir', default='Train', type=str)
parser.add_argument('--valid_file', default='valid_0.txt', type=str)

# Basic Information
parser.add_argument('--experiment', default='your_experiment_name_dir', type=str) # the name of ckpt dir for testing
parser.add_argument('--test_file', default='model_epoch_last.pth', type=str) # the name of ckpt for testing
parser.add_argument('--output_dir', default='output', type=str) # predict result root dir
parser.add_argument('--visual', default='visualization', type=str) # predict result visulizations
parser.add_argument('--use_TTA', default=True, type=bool)
parser.add_argument('--post_process', default=True, type=bool)
parser.add_argument('--seed', default=1024, type=int)

# Finetuning Information
parser.add_argument('--preckpt', default="supervised", type=str)  # supervised, clip, mae, mocov3, sam
parser.add_argument('--backbone_type', default='', type=str)
parser.add_argument('--transfer_type', default="med_adapter", type=str)

parser.add_argument('--num_workers', default=4, type=int)

args = parser.parse_args()
model_config = get_model_config(args)


def main():
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    log_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'log', args.experiment)
    log_file = log_dir + '.txt'
    log_args(log_file)
    logging.info('--------------------------------------This is Test log----------------------------------')

    # biuld model
    if 'SwinUnet' in args.backbone_type:
        model = SwinUnet_3D_All(args.transfer_type, args)
    elif 'ViT' in args.backbone_type:
        model = ViTUperNet_3D_All(args.transfer_type, args)
    else:
        raise ValueError("backbone_type '{}' is not supported".format(args.backbone_type))

    model = torch.nn.DataParallel(model).cuda() 

    load_file = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                             'checkpoint', args.experiment, args.test_file)

    if os.path.exists(load_file):
        checkpoint = torch.load(load_file)
        model.load_state_dict(checkpoint['state_dict'])
        logging.info('Successfully load checkpoint {}'.format(os.path.join(args.experiment, args.test_file)))
    else:
        logging.info('There is no resume file to load!')

    valid_list = os.path.join(args.root, args.valid_dir, args.valid_file)
    valid_root = os.path.join(args.root, args.valid_dir)

    if args.dataset_name=="BraTS2019":
        valid_set = BraTS2019(valid_list, valid_root, mode='valid')
    elif args.dataset_name=="BraTS2020":
        valid_set = BraTS2020(valid_list, valid_root, mode='valid')
    else:
        raise ValueError("dataset_name '{}' is not supported".format(args.dataset_name))

    logging.info('Samples for valid = {}'.format(len(valid_set)))

    valid_loader = DataLoader(valid_set, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # the path to save visualization
    visual = os.path.join(os.path.abspath(os.path.dirname(__file__)), args.output_dir,
                          args.visual, args.experiment) 
    if not os.path.exists(visual):
        os.makedirs(visual)

    start_time = time.time()

    with torch.no_grad():
        validate_softmax(
                        image_dir=valid_root,
                        valid_loader=valid_loader,
                        model=model,
                        visual=visual,
                        names=valid_set.names,
                        lines=valid_set.lines,
                        use_TTA=args.use_TTA, 
                        postprocess=args.post_process,
                        experiment_checkpoint=args.experiment
                        )

    end_time = time.time()
    full_test_time = (end_time-start_time)/60
    average_time = full_test_time/len(valid_set)
    logging.info('{:.2f} minutes!'.format(average_time))


if __name__ == '__main__':
    assert torch.cuda.is_available(), "Currently, we only support CUDA version"
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    main()


