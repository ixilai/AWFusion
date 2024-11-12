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
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

#
def hyper_args():
    parser = argparse.ArgumentParser(description='RobF Net train process')
    #
    parser.add_argument('--ir_path', default='../datasets/Harvard_dataset/test/ir', type=str)
    parser.add_argument('--vi_path', default='../datasets/Harvard_dataset/test/vi', type=str)
    parser.add_argument('--save_path', default='./results2/Harvard', type=str)
    parser.add_argument("--deweather_ckpt", default='./ckpt_student/Allweather_Fuse.pth', help="path to pretrained model (deweather)")
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

with torch.no_grad():
    ir_path1 = glob.glob(ir_path + '/*')
    vi_path1 = glob.glob(vi_path + '/*')
    for path1, path2 in zip(tqdm(vi_path1), ir_path1):
        img_multiple_of = 8
        save_path = path1.replace(str(vi_path), str(saving_path))

        #
        img_vi = cv2.imread(path1, cv2.IMREAD_COLOR)
        img_vi = cv2.cvtColor(img_vi, cv2.COLOR_BGR2RGB)
        if img_vi is not None and len(img_vi.shape) == 3:
            img_vi = torch.from_numpy(img_vi).float().div(255.).permute(2, 0, 1).unsqueeze(0).to(device)
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
        rgb_Fuse, out_1, I_Rtx, I_Rtx2, _ = deweather_model(img_vi, img_ir)
        restored_1 = torch.clamp(out_1, 0, 1)
        rgb_Fuse_2, out_2, I_Rtx_2, I_Rtx2_2, _ = deweather_model(restored_1, img_ir)
        restored = torch.clamp(out_2, 0, 1)

        #
        restored = restored[:, :, :height, :width]
        restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
        restored = img_as_ubyte(restored[0])

        #
        save_path = save_path.replace(str(vi_path), str(saving_path))
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, cv2.cvtColor(restored, cv2.COLOR_RGB2BGR))
