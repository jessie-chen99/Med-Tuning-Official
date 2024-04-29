import os
import cv2
import time
import logging
import setproctitle
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from medpy import metric
from utils.slide_test_3D import slide_window
from configs.config import get_cfg_defaults

config = get_cfg_defaults()

T_ori, H_ori, W_ori, input_img_size = config.TEST.input_D, config.TEST.input_H, config.TEST.input_W, config.TRAIN.input_size

def dice_score(o, t,eps=1e-8):
    num = 2*(o*t).sum() + eps
    den = o.sum() + t.sum() + eps
    return num/den

def softmax_output_dice(output, target):
    ret = []
    # WT
    o = output > 0
    t = target > 0 
    ret += dice_score(o, t),
    # TC
    o = (output == 1) | (output == 3)
    t = (target == 1) | (target == 4)
    ret += dice_score(o, t),
    # ET
    o = (output == 3)
    t = (target == 4)
    ret += dice_score(o, t),
    return ret

def one_hot(ori, classes):
    d, h, w = ori.shape[0], ori.shape[1], ori.shape[2]
    new_gd = np.zeros((classes, d, h, w), dtype=ori.dtype)
    new_gd[0] = np.where(ori==0, 1, 0)
    new_gd[1] = np.where(ori==1, 1, 0)
    new_gd[2] = np.where(ori==2, 1, 0)
    if 3 in ori:
        new_gd[3] = np.where(ori==3, 1, 0)
    if 4 in ori:
        new_gd[3] = np.where(ori==4, 1, 0)
    return new_gd

def hausdorff_distance_95(output, target):
    # test for emptiness
    if 0 == np.count_nonzero(output): 
        return float("NaN")
    elif 0 == np.count_nonzero(target): 
        return float("NaN")
    else:
        return metric.hd95(output, target)

def softmax_output_HD(output, target):
    ret = []
    output = one_hot(output, 4)
    target = one_hot(target, 4)
    # WT
    o = output[1,...]+output[2,...]+output[3,...]
    t = target[1,...]+target[2,...]+target[3,...]
    ret += hausdorff_distance_95(o, t),
    # TC
    o = output[1,...]+output[3,...]
    t = target[1,...]+target[3,...]
    ret += hausdorff_distance_95(o, t),
    # ET
    o = output[3,...]
    t = target[3,...]
    ret += hausdorff_distance_95(o, t),
    return ret

def mean_of_HD(hf_list):
    length = 0
    v_sum = 0.0
    for v in hf_list:
        if not np.isnan(v):
            v_sum += v
            length += 1
    
    if length == 0:
        v_sum = 0
    else :
        v_sum = v_sum / length     
    
    return v_sum



def validate_softmax(
        image_dir,
        valid_loader,
        model,
        names=None,  # The names of the patients orderly!
        lines=None,
        use_TTA=False,  # Test time augmentation
        visual='',  # the path to save visualization
        postprocess=False,  # Default False, when use postprocess, the score of dice_ET would be changed.
        experiment_checkpoint=None,
        ):

    model.eval()
    logging.info('Experiment_checkpoint: {}'.format(experiment_checkpoint))
    WT = []
    TC = []
    ET = []
    HF_WT = []
    HF_TC = []
    HF_ET = []

    for i, data in enumerate(valid_loader):
        setproctitle.setproctitle('test:{}/{}'.format(i+1, len(valid_loader)))
        logging.info('-------------------------------------------------------------------------------')
        
        target_cpu = data[1][0,...].numpy() # label
        data = [t.cuda(non_blocking=True) for t in data]
        x, _ = data

        if not use_TTA:
            torch.cuda.synchronize()  # add the code synchronize() to correctly count the runtime.
            start_time = time.time()
            logit, x4_pre, x4_post = model(x)[0], model(x)[-2], model(x)[-1]  # x4 (1, input_img_size, 30, 30, 20)
            torch.cuda.synchronize()
            end_time = time.time()
            elapsed_time = end_time - start_time
            logging.info('Single sample test time consumption {:.2f} minutes!'.format(elapsed_time/60))
            output = F.softmax(logit, dim=1)
        else:
            # Slide-Window TTA Single-Scale
            torch.cuda.synchronize()  # add the code synchronize() to correctly count the runtime.
            start_time = time.time()
        
            logit = F.softmax(slide_window(x, crop_size=input_img_size, model=model), 1)                                   # no flip
            logit += F.softmax(slide_window(x.flip(dims=(2,)), crop_size=input_img_size, model=model).flip(dims=(2,)), 1)  # flip D
            logit += F.softmax(slide_window(x.flip(dims=(3,)), crop_size=input_img_size, model=model).flip(dims=(3,)), 1)  # flip H_ori
            logit += F.softmax(slide_window(x.flip(dims=(4,)), crop_size=input_img_size, model=model).flip(dims=(4,)), 1)  # flip W_ori
            logit += F.softmax(slide_window(x.flip(dims=(2, 3)), crop_size=input_img_size, model=model).flip(dims=(2, 3)), 1)  # flip DH
            logit += F.softmax(slide_window(x.flip(dims=(2, 4)), crop_size=input_img_size, model=model).flip(dims=(2, 4)), 1)  # flip DW
            logit += F.softmax(slide_window(x.flip(dims=(3, 4)), crop_size=input_img_size, model=model).flip(dims=(3, 4)), 1)  # flip HW
            logit += F.softmax(slide_window(x.flip(dims=(2, 3, 4)), crop_size=input_img_size, model=model).flip(dims=(2, 3, 4)), 1)  # flip DHW
            output = logit / 8.0

            torch.cuda.synchronize()
            end_time = time.time()
            elapsed_time = end_time - start_time
            logging.info('TTA test time consumption {:.2f} minutes!'.format(elapsed_time/60))

        output = output[0, ...].cpu().detach().numpy()
        output = output.argmax(0)

        if postprocess == True:
            ET_voxels = (target_cpu == 4).sum()
            logging.info('ET_voxel: {}'.format(ET_voxels))

            ET_voxels_pred = (output == 3).sum()
            logging.info('ET_voxel_pred: {}'.format(ET_voxels_pred))
            if ET_voxels_pred < 500:
                output[np.where(output == 3)] = 1

        name = str(i)
        if names:
            name = names[i]
        if lines:
            line = lines[i]

        logging.info('Subject {}/{}, '.format(i+1, len(valid_loader)))

        if visual!='': # vis
            # red: (255,0,0) green:(0,255,0) blue:(0,0,255) 1 for NCR & NET, 2 for ED, 4 for ET, and 0 for everything else.
            gap_width = 2  # boundary width = 2
            Snapshot_img = np.zeros(shape=(T_ori, H_ori, W_ori * 2 + gap_width, 3), dtype=np.uint8)
            Snapshot_img[:, :, W_ori:W_ori + gap_width] = 255  # white boundary
            empty_fig = np.zeros(shape=(T_ori, H_ori, W_ori), dtype=np.uint8)
            empty_fig[np.where(output == 1)] = 255
            Snapshot_img[:, :, :W_ori, 0] = empty_fig
            empty_fig = np.zeros(shape=(T_ori, H_ori, W_ori), dtype=np.uint8)
            empty_fig[np.where(target_cpu == 1)] = 255
            Snapshot_img[:, :, W_ori + gap_width:, 0] = empty_fig

            empty_fig = np.zeros(shape=(T_ori, H_ori, W_ori), dtype=np.uint8)
            empty_fig[np.where(output == 2)] = 255
            Snapshot_img[:, :, :W_ori, 1] = empty_fig
            empty_fig = np.zeros(shape=(T_ori, H_ori, W_ori), dtype=np.uint8)
            empty_fig[np.where(target_cpu == 2)] = 255
            Snapshot_img[:, :, W_ori + gap_width:, 1] = empty_fig

            empty_fig = np.zeros(shape=(T_ori, H_ori, W_ori), dtype=np.uint8)
            empty_fig[np.where(output == 3)] = 255
            Snapshot_img[:, :, :W_ori, 2] = empty_fig
            empty_fig = np.zeros(shape=(T_ori, H_ori, W_ori), dtype=np.uint8)
            empty_fig[np.where(target_cpu == 4)] = 255
            Snapshot_img[:, :, W_ori + gap_width:, 2] = empty_fig

            flair_file = image_dir + '/' + line + '/' + line.split('/')[-1] + '_t1ce.nii.gz'
            logging.info('vis_image_file: {}'.format(flair_file))

            img_flair = sitk.ReadImage(flair_file)
            img_flair = sitk.GetArrayFromImage(img_flair).astype(np.float32)
            img_flair = np.rot90(img_flair, axes=(1,2))
            img_flair = np.flip(img_flair, axis=1)
            logging.info('flair: {}'.format(img_flair.shape))
            max_flair = img_flair.max()
            img_flair = img_flair * 256 / max_flair
            image = img_flair

            for frame in range(T_ori):
                if not os.path.exists(os.path.join(visual, name)):
                    os.makedirs(os.path.join(visual, name))

                image_slice_single = image[frame, :, :]
                image_slice = np.expand_dims(image_slice_single, axis=2)
                image_slice = image_slice.repeat(3, axis=2)

                Snapshot_img[frame, :, :W_ori, :] = Snapshot_img[frame, :, :W_ori, :]*0.8 + image_slice*0.5
                Snapshot_img[frame, :, W_ori + gap_width:, :] = Snapshot_img[frame, :, W_ori + gap_width:, :]*0.8 + image_slice*0.5

                cv2.imwrite(os.path.join(visual, name, str(frame) + '.png'), Snapshot_img[frame, :, :, :])


        scores = softmax_output_dice(output, target_cpu)
        HF_scores = softmax_output_HD(output, target_cpu)
        logging.info('scores: {}'.format(scores))
        logging.info('HF_scores: {}'.format(HF_scores))
        WT.append(scores[0])
        TC.append(scores[1])
        ET.append(scores[2])
        HF_WT.append(HF_scores[0])
        HF_TC.append(HF_scores[1])
        HF_ET.append(HF_scores[2])
    logging.info('ET: {}, WT: {}, TC: {}'.format(sum(ET)/len(ET)*100, sum(WT)/len(WT)*100, sum(TC)/len(TC)*100))    
    logging.info('HF_ET: {}, HF_WT: {}, HF_TC: {}'.format(mean_of_HD(HF_ET), mean_of_HD(HF_WT), mean_of_HD(HF_TC)))    
    dice_avr=(sum(ET)/len(ET)+sum(WT)/len(WT)+sum(TC)/len(TC))*100/3
    hf_avr=(mean_of_HD(HF_WT)+mean_of_HD(HF_WT)+mean_of_HD(HF_TC))/3
    logging.info('fold: Dice_AVR: {}, HF_AVR: {}'.format(dice_avr, hf_avr))    
