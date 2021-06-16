

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
VERSION = "LocNet"

NB_ITER = 200000     # Total number of iterations

I_DISPLAY = 100 
I_VALIDATION = 100 
I_SAVE = 1000


from tensorboardX import SummaryWriter
writer = SummaryWriter(log_dir='./pt_log/{}/{}'.format(EXP_NAME, str(VERSION)))



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





# Iteration
print('===> Training start')
l_accum = [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]
l_accum_n = 0.
dT = 0.
rT = 0.





## Prepare directories
if not isdir('{}'.format(EXP_NAME)):
    mkdir('{}'.format(EXP_NAME))
if not isdir('{}/checkpoint'.format(EXP_NAME)):
    mkdir('{}/checkpoint'.format(EXP_NAME))
if not isdir('{}/result'.format(EXP_NAME)):
    mkdir('{}/result'.format(EXP_NAME))
if not isdir('{}/checkpoint/v{}'.format(EXP_NAME, str(VERSION))):
    mkdir('{}/checkpoint/v{}'.format(EXP_NAME, str(VERSION)))
if not isdir('{}/result/v{}'.format(EXP_NAME, str(VERSION))):
    mkdir('{}/result/v{}'.format(EXP_NAME, str(VERSION)))




params_G = list(filter(lambda p: p.requires_grad, model_Loc.parameters()))
opt_G = optim.Adam(params_G, lr=0.0001)


START_ITER = 0
if START_ITER > 0:
    lm = torch.load('{}/checkpoint/{}/model_LOC_i{:06d}.pth'.format(EXP_NAME, str(VERSION), START_ITER))
    model_Loc.load_state_dict(lm.state_dict(), strict=True)




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
    torch.save(model_Loc, '{}/checkpoint/{}/model_LOC_i{:06d}.pth'.format(EXP_NAME, str(VERSION), i))
    print("Checkpoint saved")





accum_samples = 0





# TRAINING
for i in tqdm(range(START_ITER+1, NB_ITER+1)):

    model_Loc.train()


    # Data preparing
    # EDVR (vimeo): 7 frames, Matlab downsampling
    st = time.time()
    batch_H = Iter_H.dequeue()  # BxCxTxHxW
    batch_H = Variable(torch.from_numpy(batch_H)).cuda()  # Matlab downsampled
    dT += time.time() - st



    st = time.time()

    opt_G.zero_grad()


    k_size = 7
    s_size = 9

    border = (s_size - k_size) // 2



    # drop flat regions
    unfold_Target_ = F.unfold(batch_H[:,:,0], s_size, dilation=1, padding=0, stride=1)   # ([B, 363, N])
    B, _, N = unfold_Target_.size()

    unfold_Target = unfold_Target_.reshape(B, 3, s_size, s_size, N).permute(0,4,1,2,3).reshape(-1, 3, s_size, s_size)
    lap_res = torch.mean(torch.abs(4*unfold_Target[:, :, 1:-1, 1:-1] -(unfold_Target[:, :, :-2, 1:-1] + unfold_Target[:, :, 2:, 1:-1] + unfold_Target[:, :, 1:-1, :-2] + unfold_Target[:, :, 1:-1, 2:])), axis=[1,2,3])



    B, _ = unfold_Target.size()

    # synthesize transform
    # base_scale = 9/7
    base_scale = 1
    # random_rot_deg = 0
    random_rot_deg = (np.random.rand()*2 -1) *10
    random_rot_rad = random_rot_deg /180*np.pi
    base_t = 2/9
    random_tx = np.random.rand()*2 -1
    random_ty = np.random.rand()*2 -1
    # random_tx = 1
    # random_ty = 1


    gt_mat = torch.zeros(B, 3, 3).cuda()
    gt_mat[:,0,0] = base_scale * np.cos(random_rot_rad)
    gt_mat[:,0,1] = -base_scale * np.sin(random_rot_rad)
    gt_mat[:,0,2] = base_t * random_tx
    gt_mat[:,1,0] = base_scale * np.sin(random_rot_rad)
    gt_mat[:,1,1] = base_scale * np.cos(random_rot_rad)
    gt_mat[:,1,2] = base_t * random_ty
    gt_mat[:,2,2] = 1


    # synthesize new patch
    patch_Target = unfold_Target.reshape(-1, 3, s_size, s_size) # N', 3, 11, 11

    grid = F.affine_grid(gt_mat[:,0:2,:], torch.Size((B, 3, s_size, s_size)), align_corners=None)
    transformed_Target = F.grid_sample(patch_Target, grid, padding_mode='border', align_corners=None)

       
    
    # gt inverse transform
    base_scale = 1
    random_rot_rad = -random_rot_rad
    base_t = -2/7


    gt_mat_inv = torch.zeros(B, 3, 3).cuda()
    gt_mat_inv[:,0,0] = base_scale * np.cos(random_rot_rad)
    gt_mat_inv[:,0,1] = -base_scale * np.sin(random_rot_rad)
    gt_mat_inv[:,0,2] = base_t * random_tx
    gt_mat_inv[:,1,0] = base_scale * np.sin(random_rot_rad)
    gt_mat_inv[:,1,1] = base_scale * np.cos(random_rot_rad)
    gt_mat_inv[:,1,2] = base_t * random_ty
    gt_mat_inv[:,2,2] = 1


    # grid = F.affine_grid(gt_mat_inv[:,0:2], torch.Size((B, 3, k_size, k_size)), align_corners=None)
    # reversed_Target = F.grid_sample(transformed_Target[:,:,1:-1,1:-1], grid, padding_mode='border', align_corners=None)

    # imgs = []
    # b = np.random.randint(0, B, 10)
    # grid = grid.cpu().data.numpy()

    # for j in range(10):
        
    #     img_target = (patch_Target[b[j]].permute(1,2,0).cpu().data.numpy() *255).astype(np.uint8)
    #     img_tr_target = (transformed_Target[b[j]].permute(1,2,0).cpu().data.numpy() *255).astype(np.uint8)
    #     img_res_target = (reversed_Target[b[j]].permute(1,2,0).cpu().data.numpy() *255).astype(np.uint8)
    
    #     canvas_ref = np.zeros_like(img_target)
    #     border = 1
    #     canvas_ref[border:border+k_size, border:border+k_size] = img_res_target

    #     imgs += [np.concatenate([canvas_ref, img_target, img_tr_target], 1)]

    #     # if j == 0:
    #     #     PIL.Image.fromarray(img_target).save("test_.png")
    # PIL.Image.fromarray(np.concatenate(imgs, 0)).save("test_.png")
    # print(random_rot_deg)
    # print(gt_mat[0])
    # print(torch.inverse(gt_mat_inv)[0])
    # print( np.mean((img_res_target[2:-2,2:-2]-img_target[3:-3,3:-3])**2) )
    # input("WAIT")



    # Identity
    param_Loc = model_Loc(torch.cat([transformed_Target[:,:,1:-1,1:-1].reshape(B,-1), unfold_Target], 1))    # N', 6
    param_Loc = param_Loc.unsqueeze(2).view(-1, 2, 3)   # scale, theta, tx, ty

    # transform loss
    loss_mat = torch.mean((param_Loc - gt_mat_inv[:,:2])**2) 



    # image recovery
    grid = F.affine_grid(param_Loc, torch.Size((B, 3, k_size, k_size)), align_corners=None)
    transformed_Ref = F.grid_sample(transformed_Target[:,:,1:-1,1:-1], grid, padding_mode='border', align_corners=None)




    # loss
    loss_pix = torch.mean((transformed_Ref - patch_Target[:,:,1:-1,1:-1])**2)  *0.01

    loss = loss_mat + loss_pix
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params_G, 0.1)
    opt_G.step()
    rT += time.time() - st

    accum_samples += B

    l_accum[0] += loss_mat.item()
    l_accum[1] += loss_pix.item()
    l_accum[2] += loss.item()



    if i % I_DISPLAY == 0:
        writer.add_scalar('loss_Transform', l_accum[0]/I_DISPLAY, i)
        writer.add_scalar('loss_Pixel', l_accum[1]/I_DISPLAY, i)
        writer.add_scalar('loss_G', l_accum[2]/I_DISPLAY, i)

        print("{} {}| Iter:{:6d}, Sample:{:6d}, Loss_mat:{:.2e}, Loss_pix:{:.2e}, Loss:{:.2e}, dT:{:.4f}, rT:{:.4f}".format(
            EXP_NAME, VERSION, i, accum_samples, l_accum[0]/I_DISPLAY, l_accum[1]/I_DISPLAY, l_accum[2]/I_DISPLAY, dT/I_DISPLAY, rT/I_DISPLAY))
        l_accum = [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]
        l_accum_n = 0.
        n_mix=0
        dT = 0.
        rT = 0.


        # print(grid[0])

        imgs = []
        b = np.random.randint(0, B, 10)
        grid = grid.cpu().data.numpy()

        for j in range(10):
            img_gt = (patch_Target[b[j]].permute(1,2,0).cpu().data.numpy() *255).astype(np.uint8)
            img_ref = (transformed_Target[b[j]].permute(1,2,0).cpu().data.numpy() *255).astype(np.uint8)
            img_est = (transformed_Ref[b[j]].permute(1,2,0).cpu().data.numpy() *255).astype(np.uint8)
            
            canvas_ref = np.zeros_like(img_gt)
            border = 1
            canvas_ref[border:border+k_size, border:border+k_size] = img_est

            imgs += [np.concatenate([canvas_ref, img_gt, img_ref], 1)]
        PIL.Image.fromarray(np.concatenate(imgs, 0)).save("test_.png")
        # print(param_Loc[B//2,0,2].item(), param_Loc[B//2,1,2].item())
        # print(param_Loc[B//2].cpu().data.numpy())
        # print(gt_mat[B//2].cpu().data.numpy())
        # print(grid[0,:,:,0])
        # print(grid[0,:,:,1])
        # print(np.mean(img_ref - img_est_masked[1:-1,1:-1,:])**2)
        # input("WAIT")

    if i % I_SAVE == 0:
        SaveCheckpoint()
