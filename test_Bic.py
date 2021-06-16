

import torch
from torch.autograd import Variable

import PIL
from PIL import Image

import numpy as np
from os import listdir, mkdir
from os.path import isfile, join, isdir
from tqdm import tqdm
import glob
import skimage

from utils import _load_img_array, _rgb2ycbcr, PSNR, im2tensor




### Quality mesuare ###
# LPIPS
import LPIPS.models.dist_model as dm
model_LPIPS = dm.DistModel()
model_LPIPS.initialize(model='net-lin',net='alex',use_gpu=True)



### Generator ###
## RRDB
import RRDBNet_arch as arch
model_RRDB = arch.RRDBNet(3, 3, 64, 23, gc=32).cuda()


lm = torch.load('./models/RRDB_BlurBicubicGAN.pth')
model_RRDB.load_state_dict(lm.state_dict(), strict=True)



## Create output folder
if not isdir('./output'):
    mkdir('./output')
if not isdir('./output/Bic'):
    mkdir('./output/Bic')


## Test
with torch.no_grad():
    model_RRDB.eval()

    # Test for Gaussian8
    for testset in ['Set5', 'Set14', 'BSDS100', 'Urban100', 'Manga109']:

        if not isdir('./output/Bic/{}'.format(testset)):
            mkdir('./output/Bic/{}'.format(testset))

        psnrs = []
        ssims = []
        lpips = []

        files_gt = glob.glob('./Bicubic/gt/{}/*.png'.format(testset))
        files_gt.sort()

        files_lr = glob.glob('./Bicubic/lr_x4/{}/*.png'.format(testset))
        files_lr.sort()

        for i in tqdm(range(len(files_gt))):
            # Load images
            img_gt = np.array(PIL.Image.open(files_gt[i])) # HxWxC

            img = _load_img_array(files_lr[i])
            img = np.transpose(img, [2, 0, 1]) # CxHxW
            batch_L = img[np.newaxis, ...]

            # to tensor
            batch_L = Variable(torch.from_numpy(np.copy(batch_L)), volatile=True).cuda()

            # Process
            batch_out = model_RRDB(batch_L)


            # to CPU
            batch_out = (batch_out).cpu().data.numpy()
            batch_out = np.clip(batch_out[0], 0. , 1.)      # CxHxW
            batch_out = np.transpose(batch_out, [1, 2, 0])  # HxWxC
            batch_out = np.around(batch_out*255).astype(np.uint8)
            
            # size matching
            if img_gt.shape[0] < batch_out.shape[0]:
                batch_out = batch_out[:img_gt.shape[0]]
            elif img_gt.shape[0] > batch_out.shape[0]:
                batch_out = np.pad(batch_out, ((0,img_gt.shape[0] - batch_out.shape[0]),(0,0),(0,0)))

            if img_gt.shape[1] < batch_out.shape[1]:
                batch_out = batch_out[:, :img_gt.shape[1]]
            elif img_gt.shape[1] > batch_out.shape[1]:
                batch_out = np.pad(batch_out, ((0,0), (0,img_gt.shape[1] - batch_out.shape[1]),(0,0)))

            # handling single channel images
            if len(img_gt.shape) < 3:
                img_gt = np.tile(img_gt[:,:,np.newaxis], [1,1,3])
                img_gt_y = np.copy(img_gt)[:,:,0]
                batch_out_y = np.copy(batch_out)[:,:,0]
            # RGB to Y
            else:
                img_gt_y = _rgb2ycbcr(np.copy(img_gt))[:,:,0]
                batch_out_y = _rgb2ycbcr(np.copy(batch_out))[:,:,0]


            # Save to file
            out_fn = files_gt[i].split("/")[-1][:-4]
            Image.fromarray(batch_out).save('./output/Bic/{}/{}.png'.format(testset, out_fn))


            # Evaluation
            CROP_S = 4

            # PSNR
            psnrs.append( PSNR(img_gt_y, batch_out_y, CROP_S) )

            # SSIM
            ssims.append( skimage.metrics.structural_similarity(img_gt_y, batch_out_y, data_range=255, multichannel=False, gaussian_weights=True, sigma=1.5, use_sample_covariance=False) )

            # LPIPS
            img_gt_t = im2tensor(img_gt)
            batch_out_t = im2tensor(batch_out)
            if CROP_S > 0:
                batch_out_t = batch_out_t[:,:,CROP_S:-CROP_S,CROP_S:-CROP_S]
                img_gt_t = img_gt_t[:,:,CROP_S:-CROP_S,CROP_S:-CROP_S]
            lpips.append( model_LPIPS.forward(batch_out_t, img_gt_t) )


        print('{} AVG PSNR/SSIM/LPIPS: {}/{}/{}'.format(testset, np.mean(np.asarray(psnrs)), np.mean(np.asarray(ssims)), np.mean(np.asarray(lpips))))
