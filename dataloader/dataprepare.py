import kornia.utils
import torch.utils.data
from torch.utils.data import DataLoader
import cv2
from PIL import Image
import torchvision.transforms as transforms
import os
import glob
import numpy as np


def _imread(path):
    im_cv = Image.open(path).convert('L')
    # im_cv = cv2.imread(str(path), flags)
    im_cv = im_cv.resize((600,400), Image.ANTIALIAS)
    assert im_cv is not None, f"Image {str(path)} is invalid."
    # im_ts = kornia.utils.image_to_tensor(im_cv / 255.).type(torch.FloatTensor)
    tran = transforms.ToTensor()
    im_ts = tran(im_cv) / 255.
    return im_ts


class GetDataset_type2(torch.utils.data.Dataset):
    def __init__(self, split, ir_path=None, vi_path=None):
        super(GetDataset_type2, self).__init__()
        assert split in ['train', 'val', 'test'], 'split must be "train"|"val"|"test"'

        if split == 'train':
            data_dir_ir = ir_path
            data_dir_vis = vi_path
            self.filepath_vis, self.filenames_vis = prepare_data_path(data_dir_vis)
            self.filepath_ir, self.filenames_ir = prepare_data_path(data_dir_ir)


            self.split = split
            self.length = min(len(self.filenames_vis), len(self.filenames_ir))

        elif split == 'val':
            data_dir_vis = vi_path
            data_dir_ir = ir_path
            self.filepath_vis, self.filenames_vis = prepare_data_path(data_dir_vis)
            self.filepath_ir, self.filenames_ir = prepare_data_path(data_dir_ir)
            self.split = split
            self.length = min(len(self.filenames_vis), len(self.filenames_ir))

    def __getitem__(self, index):
        if self.split=='train':
            # print('-----------')
            vis_path = self.filepath_vis[index]
            ir_path = self.filepath_ir[index]

            image_vis = cv2.cvtColor(vis_path, cv2.COLOR_BGR2RGB)
            # image_vis = cv2.imread(vis_path, cv2.IMREAD_GRAYSCALE)
            image_inf = cv2.cvtColor(ir_path, cv2.COLOR_BGR2RGB)


            image_vis = np.array(image_vis)
            image_vis = (
                np.asarray(Image.fromarray(image_vis), dtype=np.float32).transpose(
                    (2, 0, 1)
                )
                / 255.0
            )
            image_ir = np.asarray(Image.fromarray(image_inf), dtype=np.float32) / 255.0
            image_ir = np.expand_dims(image_ir, axis=0)

            return (
                torch.tensor(image_ir),
                torch.tensor(image_vis),
            )
        elif self.split=='val':
            vis_path = self.filepath_vis[index]
            ir_path = self.filepath_ir[index]
            image_vis = np.array(Image.open(vis_path))
            image_inf = cv2.imread(ir_path, 0)
            image_vis = (
                np.asarray(Image.fromarray(image_vis), dtype=np.float32).transpose(
                    (2, 0, 1)
                )
                / 255.0
            )
            image_ir = np.asarray(Image.fromarray(image_inf), dtype=np.float32) / 255.0
            image_ir = np.expand_dims(image_ir, axis=0)
            name = self.filenames_vis[index]
            return (
                torch.tensor(image_vis),
                torch.tensor(image_ir),
                name,
            )


def prepare_data_path(dataset_path):
    filenames = os.listdir(dataset_path)
    data_dir = dataset_path
    data = glob.glob(os.path.join(data_dir, "*.bmp"))
    data.extend(glob.glob(os.path.join(data_dir, "*.tif")))
    data.extend(glob.glob((os.path.join(data_dir, "*.jpg"))))
    data.extend(glob.glob((os.path.join(data_dir, "*.png"))))
    data.sort()
    filenames.sort()
    return data, filenames


if __name__ == "__main__":
    train_dataset = GetDataset_type2('train')
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=1,
    )
    i = 0
    for vi, ir in train_loader:
        i += 1

        ir = ir.permute(0, 2, 3, 1)
        vi = vi.permute(0, 2, 3, 1)
        ir = torch.squeeze(ir, 0)
        vi = torch.squeeze(vi, 0)

        ir = ir.numpy()
        vi = vi.numpy()
        ir = (ir * 255).astype(np.uint8)
        vi = (vi * 255).astype(np.uint8)

        # ir = Image.fromarray(np.uint8(ir)).convert('RGB')
        # vi = Image.fromarray(np.uint8(vi)).convert('RGB')
        cv2.imwrite('/home/w_y/code/test/result/1/' + str(i) + '.jpg', ir)
        cv2.imwrite('/home/w_y/code/test/result/2/' + str(i) + '.jpg', vi)