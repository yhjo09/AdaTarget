

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable, Function
from torch.utils.data import DataLoader

import PIL
from PIL import Image

import numpy as np
import time
from os import listdir, mkdir
from os.path import isfile, join, isdir
import math
from tqdm import tqdm
import glob


import sys
from utils import DirectoryIterator_DIV2K_RK, GeneratorEnqueuer, _load_img_array, _rgb2ycbcr, PSNR, im2tensor



### USER PARAMS ###
EXP_NAME = "AdaTarget"
VERSION = "RK"

DIV2K_DIR = './DIV2K/'                          # DIV2K dir
PRETRAINED_LOCNET = './models/LocNet_TR.pth'    # Pretrained Localization network

NB_BATCH = 16       # mini-batch
NB_CROP_FRAME = 4   # crop N patches from every image
PATCH_SIZE = 35     # Training LR patch size

START_ITER = 0              # Set 0 for from scratch, else will load saved params and trains further
NB_ITER_FINETUNE = 100000    # The number of iterations to start finetuning
NB_ITER = 120000            # Total number of training iterations

I_DISPLAY = 100     # display info every N iteration
I_VALIDATION = 1000  # validate every N iteration
I_SAVE = 1000       # save models every N iteration

LR_G = 1e-4                     # Learning rate for the generator
LR_G_FINETUNE = 1e-5            # Learning rate for the generator, when finetuning


from tensorboardX import SummaryWriter
writer = SummaryWriter(log_dir='./pt_log/{}/{}'.format(EXP_NAME, str(VERSION)))



### Quality mesuare ###
# LPIPS
import LPIPS.models.dist_model as dm
model_LPIPS = dm.DistModel()
model_LPIPS.initialize(model='net-lin',net='alex',use_gpu=True)



### Generator ###
## RRDB
import RRDBNet_arch as arch
model_RRDB = arch.RRDBNet(3, 3, 64, 23, gc=32).cuda()
lm = torch.load('./models/RRDB_PSNR_x4.pth')
model_RRDB.load_state_dict(lm, strict=True)


## Localization network of ATG 
class LocNet(torch.nn.Module):
    def __init__(self):
        super(LocNet, self).__init__()

        ch = 9**2 *3 + 7**2 *3
        self.layer1 = nn.Linear(ch, ch*2)
        self.bn1 = nn.BatchNorm1d(ch*2)
        self.layer2 = nn.Linear(ch*2, ch*2)
        self.bn2 = nn.BatchNorm1d(ch*2)
        self.layer3 = nn.Linear(ch*2, ch)
        self.bn3 = nn.BatchNorm1d(ch)
        self.layer4 = nn.Linear(ch, 6)

        # Init weights
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.lower().find('conv') != -1:
                # print(classname)
                nn.init.kaiming_normal(m.weight)
                nn.init.constant(m.bias, 0)
            elif classname.find('bn') != -1:
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, x):
        x = self.layer1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.layer2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.layer3(x)
        x = self.bn3(x)
        x = F.relu(x)

        x = self.layer4(x)
        return x

model_Loc = LocNet().cuda()
lm = torch.load(PRETRAINED_LOCNET)
model_Loc.load_state_dict(lm.state_dict(), strict=True)




# Iteration
print('===> Training start')
l_accum = [0.,0.]
dT = 0.
rT = 0.



## Prepare directories
if not isdir('{}'.format(EXP_NAME)):
    mkdir('{}'.format(EXP_NAME))
if not isdir('{}/checkpoint'.format(EXP_NAME)):
    mkdir('{}/checkpoint'.format(EXP_NAME))
if not isdir('{}/result'.format(EXP_NAME)):
    mkdir('{}/result'.format(EXP_NAME))
if not isdir('{}/checkpoint/{}'.format(EXP_NAME, str(VERSION))):
    mkdir('{}/checkpoint/{}'.format(EXP_NAME, str(VERSION)))
if not isdir('{}/result/{}'.format(EXP_NAME, str(VERSION))):
    mkdir('{}/result/{}'.format(EXP_NAME, str(VERSION)))



# Training dataset
Iter_H = GeneratorEnqueuer(DirectoryIterator_DIV2K_RK(  #GeneratorEnqueuer
                            DIV2K_DIR,    # ciplab-seven
                            crop_size = PATCH_SIZE, # LR size
                            crop_per_image = NB_CROP_FRAME,
                            out_batch_size = NB_BATCH,
                            shuffle=True))
Iter_H.start(max_q_size=16, workers=2)


## Load saved params
if START_ITER > 0:
    lm = torch.load('{}/checkpoint/{}/model_RRDB_i{:06d}.pth'.format(EXP_NAME, str(VERSION), START_ITER))
    model_RRDB.load_state_dict(lm.state_dict(), strict=True)
    

# Save function
def SaveCheckpoint(i):
    torch.save(model_RRDB, '{}/checkpoint/{}/model_RRDB_i{:06d}.pth'.format(EXP_NAME, str(VERSION), i))
    torch.save(model_Loc, '{}/checkpoint/{}/model_Loc_i{:06d}.pth'.format(EXP_NAME, str(VERSION), i))
    torch.save(opt_G, '{}/checkpoint/{}/opt_G_i{:06d}.pth'.format(EXP_NAME, str(VERSION), i))
    print("Checkpoint saved")





accum_samples = 0




# TRAINING
for i in tqdm(range(START_ITER+1, NB_ITER+1)):

    if i == START_ITER+1:
        params_G = list(filter(lambda p: p.requires_grad, model_RRDB.parameters()))
        opt_G = optim.Adam(params_G, lr=LR_G)


    if i == NB_ITER_FINETUNE+1:
        params_G = list(filter(lambda p: p.requires_grad, model_RRDB.parameters()))
        params_G += list(filter(lambda p: p.requires_grad, model_Loc.parameters()))
        opt_G = optim.Adam(params_G, lr=LR_G_FINETUNE)




    model_RRDB.train()
    model_Loc.train()

    # Data preparing
    # EDVR (vimeo): 7 frames, Matlab downsampling
    st = time.time()
    batch_L, batch_H = Iter_H.dequeue()  # BxCxTxHxW
    batch_H = Variable(torch.from_numpy(batch_H)).cuda()  # Matlab downsampled
    batch_L = Variable(torch.from_numpy(batch_L)).cuda()  # Matlab downsampled
    dT += time.time() - st


    st = time.time()

    opt_G.zero_grad()

    # model output
    batch_S = model_RRDB(batch_L)


    ## ATG ##
    # estimate affine transform params
    k_size = 7
    s_size = 9
    ds = s_size - k_size

    _, _, H, W = batch_H.size()
    unfold_H = F.unfold(F.pad(batch_H, [ds//2,ds//2,ds//2,ds//2], mode='reflect'), s_size, dilation=1, padding=0, stride=k_size)   # ([B, 363, N])
    B, _, N = unfold_H.size()
    unfold_H = unfold_H.permute(0, 2, 1).reshape(B*N, -1)

    unfold_S = F.unfold(batch_S, k_size, dilation=1, padding=0, stride=k_size)   # B, C*k_size*k_size, L
    B, _, N = unfold_S.size()
    unfold_S = unfold_S.permute(0, 2, 1).reshape(B*N, -1)

    param_Loc = model_Loc(torch.cat([unfold_S, unfold_H], 1))    # N', 6
    param_Loc = param_Loc.unsqueeze(2).view(-1, 2, 3)   # scale, theta, tx, ty

    # Generate new SR w.r.t. the current GT, instead of generating new GT.
    # See the right upper part of the page 5 of the paper.
    grid = F.affine_grid(param_Loc, torch.Size((B*N, 3, k_size, k_size)), align_corners=None)
    transformed_S = F.grid_sample(unfold_S.reshape(-1, 3, k_size, k_size), grid, padding_mode='border', align_corners=None)
    
    transformed_S = transformed_S.reshape(B, N, -1).permute(0, 2, 1)
    transformed_S = F.fold(transformed_S, [H, W], k_size, stride=k_size)
    

    # Loss
    loss_Pixel = torch.mean((transformed_S - batch_H)**2) 

    # Update
    loss_G = loss_Pixel
    loss_G.backward()
    torch.nn.utils.clip_grad_norm_(params_G, 0.1)
    opt_G.step()
    rT += time.time() - st

    # For monitoring
    l_accum[0] += loss_Pixel.item()
    accum_samples += NB_BATCH



    ## Show information
    if i % I_DISPLAY == 0:
        writer.add_scalar('loss_Pixel', l_accum[0]/I_DISPLAY, i)

        print("{} {}| Iter:{:6d}, Sample:{:6d}, Pixel:{:.2e}, dT:{:.4f}, rT:{:.4f}".format(
            EXP_NAME, VERSION, i, accum_samples, l_accum[0]/I_DISPLAY, dT/I_DISPLAY, rT/I_DISPLAY))
        l_accum = [0.,0.]
        dT = 0.
        rT = 0.


    ## Save models
    if i % I_SAVE == 0:
        SaveCheckpoint(i)


    ## Validate
    if i % I_VALIDATION == 0:
        with torch.no_grad():
            model_RRDB.eval()

            valset_name = 'valid_rk'

            files_lr = glob.glob(DIV2K_DIR + '/DIV2K_valid_LR_RK/*.png')
            files_lr.sort()
            files_lr = files_lr[:5]     # using a small number of images for simplicity

            files_gt = glob.glob(DIV2K_DIR + '/DIV2K_valid_HR/*.png')
            files_gt.sort()
            files_gt = files_gt[:5]

            psnrs = []
            lpips = []

            if not isdir('{}/result/{}/{}'.format(EXP_NAME, str(VERSION), valset_name)):
                mkdir('{}/result/{}/{}'.format(EXP_NAME, str(VERSION), valset_name))

            for ti, fn in enumerate(files_gt):
                # open image
                img = _load_img_array(files_lr[ti])
                img = np.asarray(img).astype(np.float32) # HxWxC
                img = np.transpose(img, [2, 0, 1]) # CxHxW
                batch_L = img[np.newaxis, ...]

                img_H = _load_img_array(fn)
                img_H = np.asarray(img_H).astype(np.float32) # HxWxC

                #
                batch_L = Variable(torch.from_numpy(np.copy(batch_L)), volatile=True).cuda()
                        
                batch_output = model_RRDB(batch_L)

                #
                batch_output = (batch_output).cpu().data.numpy()
                batch_output = np.clip(batch_output[0], 0. , 1.) # CxHxW
                batch_output = np.transpose(batch_output, [1, 2, 0])

                # Save to file
                img_gt = np.around(img_H*255).astype(np.uint8)
                img_out = np.around(batch_output*255).astype(np.uint8)

                Image.fromarray(img_out).save('{}/result/{}/{}/{}.png'.format(EXP_NAME, str(VERSION), valset_name, fn.split('/')[-1]))

                # PSNR
                CROP_S = 4
                psnrs.append(PSNR(_rgb2ycbcr(img_gt)[:,:,0], _rgb2ycbcr(img_out)[:,:,0], CROP_S))

                # LPIPS
                img_gt = im2tensor(img_gt) # RGB image from [-1,1]
                img_out = im2tensor(img_out)
                if CROP_S > 0:
                    img_out = img_out[:,:,CROP_S:-CROP_S,CROP_S:-CROP_S]
                    img_gt = img_gt[:,:,CROP_S:-CROP_S,CROP_S:-CROP_S]
                dist = model_LPIPS.forward(img_out, img_gt)
                lpips.append(dist)


        mean_psnr = np.mean(np.asarray(psnrs))
        mean_lpips = np.mean(np.asarray(lpips))

        print('AVG PSNR/LPIPS: {}: {}/{}'.format(valset_name, mean_psnr, mean_lpips))

        writer.add_scalar(valset_name, mean_psnr, i)
        writer.add_scalar(valset_name+'_lpips', mean_lpips, i)

