import os
import cv2
from utils.metrics import calculate_psnr, calculate_ssim
import torch
import lpips
# loss_fn_alex = lpips.LPIPS(net='alex')
loss_fn_vgg = lpips.LPIPS(net='vgg')

base_path = '/mnt/sda/lxy/NightDrop/' 

nightend_path = 'NightRainDrop/gt/'
model_name   = 'restormer/'

gt_path = base_path + model_name + nightend_path
results_path = gt_path.replace('gt', 'sparse')
print(results_path)

imgsName = sorted(os.listdir(results_path))
imgslist = []
for i in range(len(imgsName)):
    path_1 = os.path.join(results_path, imgsName[i])
    dir_1 = sorted(os.listdir(path_1))
    for j in range(len(dir_1)):
        imgslist.append(path_1 + '/' +dir_1[j])
# gtsName = sorted(os.listdir(gt_path))
print('-len(imgslist)-',len(imgslist))
gtslist = []
for i in range(len(imgslist)):
    gts = imgslist[i].replace(results_path,gt_path)
    # gts = gts.replace('_output.','_gt.')
    gtslist.append(gts)
print(gtslist[0])
print(gtslist[-1])
print(imgslist[0])
print(imgslist[-1])
cumulative_psnr, cumulative_ssim,cumulative_lpips = 0, 0, 0

for i in range(len(imgslist)):
    # print(imgslist[i])
    res = cv2.imread(imgslist[i], cv2.IMREAD_COLOR)
    gt  = cv2.imread(gtslist[i], cv2.IMREAD_COLOR)
    cur_psnr = calculate_psnr(res, gt, test_y_channel=True)
    cur_ssim = calculate_ssim(res, gt, test_y_channel=True)

    torchres = torch.from_numpy(res.transpose((2, 0, 1))).float().unsqueeze(0)
    torchgt  = torch.from_numpy(gt.transpose((2, 0, 1))).float().unsqueeze(0)
    # torchres = torchres/255.0 *2 - 1
    # torchgt = torchgt/255.0 *2 - 1
    # print('-torchres-',torchres.shape,'-torchgt-',torchgt.shape)
    cur_lpips = loss_fn_vgg(torchres, torchgt)
    # print('PSNR is %.4f and SSIM is %.4f' % (cur_psnr, cur_ssim))
    cumulative_psnr += cur_psnr
    cumulative_ssim += cur_ssim
    cumulative_lpips +=cur_lpips.cpu().data.numpy()[0][0][0][0]
    if i%100==0:
        print('Testing set,'+str(i)+' PSNR is %.4f, SSIM is %.4f and lpips is %.4f' % (cumulative_psnr / (i+1), cumulative_ssim / (i+1), cumulative_lpips / (i+1)))
print(time_of_day)
print('%s, PSNR is %.4f, SSIM is %.4f and lpips is %.4f' % (model_name, cumulative_psnr / len(imgslist), cumulative_ssim / len(imgslist), cumulative_lpips / len(imgslist)))

