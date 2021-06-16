

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


lm = torch.load('./models/RRDB_BlurRK.pth')
model_RRDB.load_state_dict(lm.state_dict(), strict=True)



## Create output folder
if not isdir('./output'):
    mkdir('./output')
if not isdir('./output/RK'):
    mkdir('./output/RK')


## Test
with torch.no_grad():
    model_RRDB.eval()

    psnrs = []
    ssims = []
    lpips = []

    for i in tqdm(range(1, 101)):
        
        # Load images
        img = _load_img_array('./DIV2KRK/lr_x4/im_{}.png'.format(i))
        img = np.transpose(img, [2, 0, 1]) # CxHxW
        batch_L = img[np.newaxis, ...]

        img_gt = np.array(PIL.Image.open('./DIV2KRK/gt/img_{}_gt.png'.format(i)))[:,:,:3] # HxWxC

        # to tensor
        batch_L = Variable(torch.from_numpy(np.copy(batch_L)), volatile=True).cuda()
                
        # Process
        batch_out = model_RRDB(batch_L)


        # to CPU
        batch_out = (batch_out).cpu().data.numpy()
        batch_out = np.clip(batch_out[0], 0. , 1.)      # CxHxW
        batch_out = np.transpose(batch_out, [1, 2, 0])  # HxWxC
        batch_out = np.around(batch_out*255).astype(np.uint8)
           
        img_gt_y = _rgb2ycbcr(np.copy(img_gt))[:,:,0]
        batch_out_y = _rgb2ycbcr(np.copy(batch_out))[:,:,0]


        # Save to file
        out_fn = 'im_{:03d}.png'.format(i)
        Image.fromarray(batch_out).save('./output/RK/{}.png'.format(out_fn))


        # Evaluation
        CROP_S = 4

        # PSNR
        psnrs.append( PSNR(img_gt_y, batch_out_y, CROP_S) )

        # SSIM
        ssims.append( skimage.metrics.structural_similarity(img_gt_y, batch_out_y, data_range=255, multichannel=False, gaussian_weights=True, sigma=1.5, use_sample_covariance=False) )

        # LPIPS
        img_gt = im2tensor(img_gt)
        batch_out = im2tensor(batch_out)
        if CROP_S > 0:
            batch_out = batch_out[:,:,CROP_S:-CROP_S,CROP_S:-CROP_S]
            img_gt = img_gt[:,:,CROP_S:-CROP_S,CROP_S:-CROP_S]
        lpips.append( model_LPIPS.forward(batch_out, img_gt) )


print('AVG PSNR/SSIM/LPIPS: {}/{}/{}'.format(np.mean(np.asarray(psnrs)), np.mean(np.asarray(ssims)), np.mean(np.asarray(lpips))))
