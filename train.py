import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import dataloader
import net
import pro_net
import psanet
import multi_net
import numpy as np
from torchvision import transforms
# import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
def init(net):
    classname = net.__class__.__name__
    if classname.find('Conv') != -1:
        net.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        net.weight.data.normal_(1.0, 0.02)
        net.bias.data.fill_(0)

def calc_ssim(clean_image_np,img_orig_np):
# 遍历批次中的每一张图像
    ssim_scores = np.zeros((clean_image_np.shape[0],))  # 存储每张图像的 SSIM 分数
    for i in range(clean_image_np.shape[0]):
        # imgs1[i] 和 imgs2[i] 都是 (3, 640, 480) 形状
        # 调整为 (640, 480, 3) 以满足 skimage 的 SSIM 函数要求
        img1 = np.transpose(clean_image_np[i], (1, 2, 0))
        img2 = np.transpose(img_orig_np[i], (1, 2, 0))
        # 计算 SSIM
        # 为了正确计算，我们假设像素值范围在 0 到 1 之间
        # 如果您的图像是 0-255 范围的 uint8 类型，应该使用 data_range=255
        ssim_index = ssim(img1, img2, win_size=7,multichannel=True, channel_axis=-1,data_range=img1.max() - img1.min() )
        ssim_scores[i] = ssim_index

    return ssim_scores.mean()



def train(args):
    unfog_net = net.unfog_net()
    unfog_net.apply(init)
    if args.model=='default':
         unfog_net = net.unfog_net()
    elif args.model=='improved':
        unfog_net = pro_net.unfog_net()
    elif args.model=='psa':
        unfog_net = psanet.unfog_net()
    elif args.moel=='multi_net':
        unfog_net = multi_net.dehaze_net()
    else:
        pass
    train_dataset = dataloader.unfogging_loader(args.orig_images_path,
                                             args.foggy_images_path)
    # quit()
    val_dataset = dataloader.unfogging_loader(args.orig_images_path,
                                             args.foggy_images_path, mode="val")	
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.val_batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(unfog_net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    unfog_net.train()
    best_lost = np.inf
    data_loss=[]
    data_val_loss=[]
    data_pnsr=[]
    data_ssim=[]
    for epoch in range(args.num_epochs):
        loss_list = []
        if True:
            for iteration, (img_orig, img_fog) in enumerate(train_loader):
                img_orig = img_orig
                img_fog = img_fog
                # print(img_fog.shape)
                # return 
                clean_image = unfog_net(img_fog)
                img_orig_np = img_orig.detach().cpu().numpy()
                clean_image_np = clean_image.detach().cpu().numpy()
                loss = criterion(clean_image, img_orig)
                loss_list.append(loss)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm(unfog_net.parameters(),args.grad_clip_norm)
                optimizer.step()
                if ((iteration+1) % args.display_iter) == 0:
                    print("Loss at iteration", iteration+1, ":", loss.item())		
    
        psnr_list = []
        ssim_list = []
        val_loss_list = []
        # Validation Stage
        for iter_val, (img_orig, img_fog) in enumerate(val_loader):
            img_orig = img_orig
            img_fog = img_fog
            clean_image = unfog_net(img_fog)
            img_orig_np = img_orig.detach().cpu().numpy()
            clean_image_np = clean_image.detach().cpu().numpy()
            psnr_value = psnr(img_orig_np, clean_image_np, data_range=img_orig_np.max() - clean_image_np.min())
            ssim_value = calc_ssim(clean_image_np,img_orig_np)
            # ssim_value = ssim(img_orig_np, clean_image_np, data_range=img_orig_np.max() - clean_image_np.min(), multichannel=True)
            val_loss = criterion(clean_image, img_orig)
            val_loss_list.append(val_loss)
            psnr_list.append(psnr_value)
            ssim_list.append(ssim_value)
            torchvision.utils.save_image(torch.cat((img_fog, clean_image, img_orig),0), args.sample_output_folder+str(iter_val+1)+".jpg")
        
        final_psnr = np.array([x.item() for x in psnr_list]).sum() / len(psnr_list)
        final_loss = np.array([x.item() for x in loss_list]).sum() / len(loss_list)
        final_val_loss = np.array([x.item() for x in val_loss_list]).sum() / len(val_loss_list)
        final_ssim = np.array([x.item() for x in ssim_list]).sum() / len(ssim_list)
        if final_loss<best_lost:
            best_lost = final_loss
            torch.save(unfog_net.state_dict(), args.snapshots_folder + "net1.pth") 
        print('Epoch:{},train_avg_loss:{:.4f},eval_avg_loss:{:.4f},eval_avg_psnr:{:.4f},eval_avg_ssim:{:.4f}'.format(epoch+1,final_loss,final_val_loss,final_psnr,final_ssim))
        data_loss.append(final_loss)
        data_val_loss.append(final_val_loss)
        data_pnsr.append(final_psnr)
        data_ssim.append(final_ssim)
    print(data_loss,data_val_loss,data_pnsr,data_ssim)
    
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Input Parameters
    parser.add_argument('--orig_images_path', type=str, default="final/Image-Unfogging/data/images/")
    parser.add_argument('--foggy_images_path', type=str, default="final/Image-Unfogging/data/data/")
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--grad_clip_norm', type=float, default=0.1)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--val_batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--display_iter', type=int, default=1)
    parser.add_argument('--snapshot_iter', type=int, default=200)
    parser.add_argument('--snapshots_folder', type=str, default="snapshots/")
    parser.add_argument('--sample_output_folder', type=str, default="samples/")
    parser.add_argument('--degree', type=str, default="7_2",choices=['1_2','2_2','3_2','4_2','6_2','7_2'])
    parser.add_argument('--model', type=str, default="multi_net",choices=['default','pono','psa','improved','multi_net'])
    args = parser.parse_args()

    if not os.path.exists(args.snapshots_folder):
        os.mkdir(args.snapshots_folder)
    if not os.path.exists(args.sample_output_folder):
        os.mkdir(args.sample_output_folder)
# valid degree # 1-2,2-2,3-2,4-2,6-2,7-2
    train(args)
    







    
