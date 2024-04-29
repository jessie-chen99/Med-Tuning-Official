import os
import argparse
import random
import logging
import time
import math
import setproctitle
import torch
import torch.optim
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from data.BraTS import BraTS2019, BraTS2020
from models.model_3D.swin_unet_3D import SwinUnet_3D_All
from models.model_3D.vitb16_3D import ViTUperNet_3D_All
from utils import criterions
from utils.tools import all_reduce_tensor, freeze_params, params_count, log_args
from configs.config import get_model_config

local_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
parser = argparse.ArgumentParser()

# training data path
parser.add_argument('--train_dir', default='Train', type=str)
parser.add_argument('--valid_dir', default='Valid', type=str)
parser.add_argument('--train_file', default='train.txt', type=str)

# Basic Information
parser.add_argument('--experiment', default='your_experiment_name_dir', type=str)
parser.add_argument('--date', default=local_time.split(' ')[0], type=str)

# Finetuning Information
parser.add_argument('--preckpt', default="supervised", type=str)  # supervised, clip, mae, mocov3, sam
parser.add_argument('--backbone_type', default='ViTB16', type=str) # ViTB16
parser.add_argument('--transfer_type', default="med_adapter", type=str)  # head, scratch, full_finetuning_3D, med_adapter

# DataSet Information
parser.add_argument('--dataset_name', default='BraTS2019', type=str) # BraTS2019, BraTS2020
parser.add_argument('--root', default='.../your_datasets_file', type=str) # your_datasets_file, eg. ../data/BraTS2019
parser.add_argument('--mode', default='train', type=str) # train, valid

# Training Information
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--weight_decay', default=1e-5, type=float)
parser.add_argument('--criterion', default='softmax_dice', type=str) # use softmax_dice as default loss
parser.add_argument('--batch_size', default=1, type=int) # set batchsize for per gpu
parser.add_argument('--end_epoch', default=250, type=int) # set total training epochs
parser.add_argument('--warmup_epoch', default=25, type=int) # set warmup epochs
parser.add_argument('--save_freq', default=300, type=int) # save ckpt in each 'save_freq'

parser.add_argument('--load', default=False, type=bool) # train from previous ckpt(True) or set a new train(False)
parser.add_argument('--resume', default='', type=str) # the path of previous ckpt
parser.add_argument('--start_epoch', default=0, type=int) # the end sepoch of previous ckpt

parser.add_argument('--seed', default=1024, type=int) # fix this during your experiment
parser.add_argument('--num_workers', default=8, type=int)
parser.add_argument('--local_rank', default=0, type=int, help='node rank for distributed training')

args = parser.parse_args()
model_config = get_model_config(args)

def warm_up_learning_rate(init_lr, epoch, warm_epoch, max_epoch, optimizer):
    for param_group in optimizer.param_groups:
        if epoch < warm_epoch:
            param_group['lr'] = init_lr*(1-math.cos(math.pi/2*(epoch+1)/(warm_epoch)))
        else:
            param_group['lr'] = init_lr*(math.cos(math.pi*(epoch-warm_epoch)/max_epoch)+1)/2


def main_worker():
    if args.local_rank == 0:
        log_root = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'log')
        if not os.path.exists(log_root):
            os.makedirs(log_root)
        log_file = os.path.join(log_root, args.experiment+args.date)+'.txt'
        log_args(log_file)
        logging.info('--------------------------------------This is all args----------------------------------')
        for arg in vars(args):
            logging.info('{}={}'.format(arg, getattr(args, arg)))
        logging.info('--------------------------------------This is all model configs----------------------------------')
        logging.info(model_config)
        logging.info('---------------------------------------Start training-----------------------------------')

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed) 

    torch.distributed.init_process_group(backend='nccl')
    torch.cuda.set_device(args.local_rank)

    # biuld model
    if 'SwinUnet' in args.backbone_type:
        model = SwinUnet_3D_All(args.transfer_type, args)
    elif 'ViT' in args.backbone_type:
        model = ViTUperNet_3D_All(args.transfer_type, args)
    else:
        raise ValueError("backbone_type '{}' is not supported".format(args.backbone_type))

    if args.transfer_type == "scratch":
        logging.info("Training from scratch, do not need to load pre-train ckpt")
    else:
        logging.info("Start Load pre-train ckpt!")
        model.load_from(args.preckpt)
    
    # freeze emcoder params & calculate model params
    freeze_params(args.transfer_type, model)
    logging.info("Successfully freezed the Encoder params!")
    params_count(args.transfer_type, model)

    model.cuda(args.local_rank)

    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = nn.parallel.DistributedDataParallel(model,
                                                device_ids=[args.local_rank],
                                                output_device=[args.local_rank]
                                                # find_unused_parameters=True
                                                )

    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, amsgrad=True)
    criterion = getattr(criterions, args.criterion)

    if args.local_rank == 0:
        checkpoint_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'checkpoint', args.experiment+args.date)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

    if os.path.isfile(args.resume) and args.load:
        logging.info('loading checkpoint {}'.format(args.resume))
        checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage)

        model.load_state_dict(checkpoint['state_dict'])

        logging.info('Successfully loading checkpoint {} and training from epoch: {}'
                     .format(args.resume, args.start_epoch))
    else:
        logging.info('new-training!!!')

    # load data
    train_list = os.path.join(args.root, args.train_dir, args.train_file)
    train_root = os.path.join(args.root, args.train_dir)
    
    if args.dataset_name=="BraTS2019":
        train_set = BraTS2019(train_list, train_root, args.mode)
    elif args.dataset_name=="BraTS2020":
        train_set = BraTS2020(train_list, train_root, args.mode)
    else:
        raise ValueError("dataset_name '{}' is not supported".format(args.dataset_name))
   
    logging.info('Samples for train = {}'.format(len(train_set)))

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, shuffle=True)

    num_gpu = torch.cuda.device_count()

    train_loader = DataLoader(dataset=train_set, sampler=train_sampler, batch_size=args.batch_size,# // num_gpu,
                              drop_last=True, num_workers=args.num_workers, pin_memory=True)

    start_time = time.time()

    torch.set_grad_enabled(True)

    for epoch in range(args.start_epoch, args.end_epoch):
        train_sampler.set_epoch(epoch)  
        setproctitle.setproctitle('{}: {}/{}'.format('ft:', epoch+1, args.end_epoch))
        start_epoch = time.time()

        for i, data in enumerate(train_loader):
            warm_up_learning_rate(args.lr, epoch, args.warmup_epoch, args.end_epoch, optimizer)

            x, target = data
            x = x.cuda(args.local_rank, non_blocking=True)
            target = target.cuda(args.local_rank, non_blocking=True)

            output = model(x)

            loss, loss1, loss2, loss3 = criterion(output, target)

            reduce_loss = all_reduce_tensor(loss, world_size=num_gpu).data.cpu().numpy()
            reduce_loss1 = all_reduce_tensor(loss1, world_size=num_gpu).data.cpu().numpy()
            reduce_loss2 = all_reduce_tensor(loss2, world_size=num_gpu).data.cpu().numpy()
            reduce_loss3 = all_reduce_tensor(loss3, world_size=num_gpu).data.cpu().numpy()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if args.local_rank == 0:
            logging.info('Epoch: {} loss: {:.5f} || 1:{:.4f} | 2:{:.4f} | 3:{:.4f} ||'
                            .format(epoch, reduce_loss, reduce_loss1, reduce_loss2, reduce_loss3))

        end_epoch = time.time()
        if args.local_rank == 0:
            if (epoch + 1) % int(args.save_freq) == 0:
                file_name = os.path.join(checkpoint_dir, 'model_epoch_{}.pth'.format(epoch))
                torch.save({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optim_dict': optimizer.state_dict(),
                },
                    file_name)

        if args.local_rank == 0:
            epoch_time_minute = (end_epoch-start_epoch)/60
            remaining_time_hour = (args.end_epoch-epoch-1)*epoch_time_minute/60
            logging.info('Current epoch time consumption: {:.2f} minutes!'.format(epoch_time_minute))
            logging.info('Estimated remaining training time: {:.2f} hours!'.format(remaining_time_hour))

    if args.local_rank == 0:
        final_name = os.path.join(checkpoint_dir, 'model_epoch_last.pth')
        torch.save({
            'epoch': args.end_epoch,
            'state_dict': model.state_dict(),
            'optim_dict': optimizer.state_dict(),
        },
            final_name)
    end_time = time.time()
    total_time = (end_time-start_time)/3600
    logging.info('The total training time is {:.2f} hours'.format(total_time))

    logging.info('----------------------------------The training process finished!-----------------------------------')


if __name__ == '__main__':
    assert torch.cuda.is_available(), "Currently, we only support CUDA version"
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    main_worker()
