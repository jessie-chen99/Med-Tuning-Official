import os
import time
import logging
import setproctitle
import torch
import torch.nn.functional as F
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from utils.slide_test_3D import slide_window
from configs.config import get_cfg_defaults

config = get_cfg_defaults()

T_ori, H_ori, W_ori, input_img_size = config.TEST.input_D, config.TEST.input_H, config.TEST.input_W, config.TRAIN.input_size

def one_hot(ori, classes):
    batch, h, w, d = ori.size()
    new_gd = torch.zeros((batch, classes, h, w, d), dtype=ori.dtype).cuda()
    for j in range(classes):
        index_list = (ori == j).nonzero()
        for i in range(len(index_list)):
            batch, height, width, depth = index_list[i]
            new_gd[batch, j, height, width, depth] = 1
    return new_gd.float()


def validate_softmax(
        dataset_name,
        valid_loader,
        model,
        savepath='',  # when in validation set, you must specify the path to save the 'nii' segmentation results here
        names=None,  # The names of the patients orderly!
        verbose=False,
        use_TTA=False,  # Test time augmentation, False as default!
        save_format=None,  # ['nii','npy'], use 'nii' as default. Its purpose is for submission.
        postprocess=False,  # Default False, when use postprocess, the score of dice_ET would be changed.
        experiment_checkpoint=None,
        ):

    model.eval()
    logging.info('Experiment_checkpoint: {}'.format(experiment_checkpoint))
    runtimes = []

    for i, data in enumerate(valid_loader):
        setproctitle.setproctitle('offi-test:{}/{}'.format(i+1, len(valid_loader)))
        logging.info('-------------------------------------------------------------------------------')
        

        x = data 
        x.cuda()

        if use_TTA:
            torch.cuda.synchronize()  # add the code synchronize() to correctly count the runtime.
            start_time = time.time()

            # Slide-Window TTA Single-Scale
            logit = F.softmax(slide_window(x, crop_size=input_img_size, model=model), 1) # no flip
            logit += F.softmax(slide_window(x.flip(dims=(2,)), crop_size=input_img_size, model=model).flip(dims=(2,)), 1)  # flip D
            logit += F.softmax(slide_window(x.flip(dims=(3,)), crop_size=input_img_size, model=model).flip(dims=(3,)), 1)  # flip H
            logit += F.softmax(slide_window(x.flip(dims=(4,)), crop_size=input_img_size, model=model).flip(dims=(4,)), 1)  # flip W
            logit += F.softmax(slide_window(x.flip(dims=(2, 3)), crop_size=input_img_size, model=model).flip(dims=(2, 3)), 1)  # flip DH
            logit += F.softmax(slide_window(x.flip(dims=(2, 4)), crop_size=input_img_size, model=model).flip(dims=(2, 4)), 1)  # flip DW
            logit += F.softmax(slide_window(x.flip(dims=(3, 4)), crop_size=input_img_size, model=model).flip(dims=(3, 4)), 1)  # flip HW
            logit += F.softmax(slide_window(x.flip(dims=(2, 3, 4)), crop_size=input_img_size, model=model).flip(dims=(2, 3, 4)), 1)  # flip DHW
            output = logit / 8.0

            torch.cuda.synchronize()
            elapsed_time = time.time() - start_time
            logging.info('TTA test time consumption {:.2f} minutes!'.format(elapsed_time / 60))
            runtimes.append(elapsed_time)

        output = output[0,...].permute(0,2,3,1).contiguous().cpu().detach().numpy()
        output = output.argmax(0)

        if postprocess == True:
            ET_voxels_pred = (output == 3).sum()
            if ET_voxels_pred < 500:
                output[np.where(output == 3)] = 1

        name = str(i)
        if names:
            name = names[i]
        logging.info('Subject {}/{}, '.format(i+1, len(valid_loader)))

        if savepath:
            # .npy for further model ensemble
            # .nii for directly model submission
            assert save_format in ['npy', 'nii']
            if save_format == 'npy':
                np.save(os.path.join(savepath, name + '_preds'), output)
            if save_format == 'nii':
                # raise NotImplementedError
                oname = os.path.join(savepath, name + '.nii.gz')
                seg_img = np.zeros(shape=(H_ori, W_ori, T_ori), dtype=np.uint8)

                seg_img[np.where(output == 1)] = 1
                seg_img[np.where(output == 2)] = 2
                seg_img[np.where(output == 3)] = 4
                if verbose:
                    logging.info('1:', np.sum(seg_img == 1), ' | 2:', np.sum(seg_img == 2), ' | 4:', np.sum(seg_img == 4))
                    logging.info('WT:', np.sum((seg_img == 1) | (seg_img == 2) | (seg_img == 4)), ' | TC:',
                          np.sum((seg_img == 1) | (seg_img == 4)), ' | ET:', np.sum(seg_img == 4))

                # save submission
                if dataset_name=='BraTS2019':
                    nib.save(nib.Nifti1Image(seg_img, None), oname)
                    logging.info('Successfully save BraTS2019 pred {}'.format(oname))
                elif dataset_name=='BraTS2020':
                    # Read an submission file generated by the open_brats for the necessary reference information
                    c0 = sitk.ReadImage('./data/BraTS20_Validation_002.nii.gz')
                    Direction = c0.GetDirection()
                    Origin = c0.GetOrigin()
                    Spacing = c0.GetSpacing()
                    seg_img = sitk.GetImageFromArray(seg_img.transpose(2, 1, 0).contiguous())
                    seg_img.SetOrigin(Origin)
                    seg_img.SetSpacing(Spacing)
                    seg_img.SetDirection(Direction)
                    sitk.WriteImage(seg_img, f"{oname}.nii.gz")
                    logging.info('Successfully save BraTS2020 pred {}'.format(oname))