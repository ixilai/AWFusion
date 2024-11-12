from model.AWFusion_module import AWF
import os
import torch.nn as nn
import torch
import cv2
import argparse
from tqdm import tqdm
import glob
import torch.nn.functional as F
from skimage import img_as_ubyte
#
from PIL import Image
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

#
def hyper_args():
    parser = argparse.ArgumentParser(description='RobF Net train process')
    #
    parser.add_argument('--ir_path', default='./test/ir', type=str)
    parser.add_argument('--vi_path', default='./test/vi', type=str)
    parser.add_argument('--save_path', default='./results', type=str)
    parser.add_argument("--deweather_ckpt", default='./ckpt/Allweather_Fuse.pth', help="path to pretrained model")
    return parser.parse_args()
#
args = hyper_args()

#
ir_path = args.ir_path
vi_path = args.vi_path
saving_path = args.save_path
deweather_ckpt = args.deweather_ckpt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
deweather_model = AWF().to(device)
deweather_model = nn.DataParallel(deweather_model)
checkpoint = torch.load(deweather_ckpt, map_location=device)  #
deweather_model.load_state_dict(checkpoint)  # student
deweather_model.eval()

def rgb_to_ycrcb(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)

def ycrcb_to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_YCrCb2RGB)

def replace_y_channel(image_a, image_b):
    # 将图像A和图像B从RGB转换为YCrCb
    ycrcb_a = rgb_to_ycrcb(image_a)
    ycrcb_b = rgb_to_ycrcb(image_b)

    # 替换图像B的Y通道为图像A的Y通道
    y_a = ycrcb_a[:, :, 0]  # A图的Y通道
    ycrcb_b[:, :, 0] = y_a  # 替换B图的Y通道为A图的Y通道

    # 将修改后的YCrCb图像B转换回RGB
    result_b_rgb = ycrcb_to_rgb(ycrcb_b)

    return result_b_rgb


with torch.no_grad():
    ir_path1 = glob.glob(ir_path + '/*')
    vi_path1 = glob.glob(vi_path + '/*')
    for path1, path2 in zip(tqdm(vi_path1), ir_path1):
        img_multiple_of = 8
        save_path = path1.replace(str(vi_path), str(saving_path))

        #
        img_vi = cv2.imread(path1, cv2.IMREAD_COLOR)
        img_vi_read = cv2.cvtColor(img_vi, cv2.COLOR_BGR2RGB)
        if img_vi_read is not None and len(img_vi_read.shape) == 3:
            img_vi = torch.from_numpy(img_vi_read).float().div(255.).permute(2, 0, 1).unsqueeze(0).to(device)
        else:
            raise ValueError(f"Error processing img_vi from path: {path1}")

        #
        img_ir = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)
        if img_ir is not None and len(img_ir.shape) == 2:
            img_ir = torch.from_numpy(img_ir).float().div(255.).unsqueeze(0).unsqueeze(0).to(device)
        else:
            raise ValueError(f"Error processing img_ir from path: {path2}")

        #
        _, _, height, width = img_vi.shape

        #
        H = (height + img_multiple_of - 1) // img_multiple_of * img_multiple_of
        W = (width + img_multiple_of - 1) // img_multiple_of * img_multiple_of
        padh = H - height if height % img_multiple_of != 0 else 0
        padw = W - width if width % img_multiple_of != 0 else 0

        #
        img_vi = F.pad(img_vi, (0, padw, 0, padh), 'reflect')
        img_ir = F.pad(img_ir, (0, padw, 0, padh), 'reflect')

        #
        rgb_Fuse, out, I_Rtx, I_Rtx2, _ = deweather_model(img_vi, img_ir)
        restored = torch.clamp(out, 0, 1)

        #
        restored = restored[:, :, :height, :width]
        restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
        restored = img_as_ubyte(restored[0])

        #
        save_path = save_path.replace(str(vi_path), str(saving_path))
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        result_image = replace_y_channel(img_vi_read, restored)
        cv2.imwrite(save_path, cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
