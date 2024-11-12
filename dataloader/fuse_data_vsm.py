import torch.utils.data
from torch.utils.data import DataLoader
import cv2
from PIL import Image
import torchvision.transforms as transforms
import os
import glob
import numpy as np
from dataloader import transforms as T


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
    def __init__(self, split, ir_path=None, vi_path=None, gt_path=None):
        super(GetDataset_type2, self).__init__()

        if split == 'train':
            data_dir_ir = ir_path
            data_dir_vis = vi_path
            data_dir_gt = gt_path
            self.filepath_vis, self.filenames_vis = prepare_data_path(data_dir_vis)
            self.filepath_ir, self.filenames_ir = prepare_data_path(data_dir_ir)
            self.filepath_gt, self.filenames_gt = prepare_data_path(data_dir_gt)

            self.split = split
            self.length = min(len(self.filenames_vis), len(self.filenames_ir), len(self.filenames_gt))

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
            gt_path = self.filepath_gt[index]

            image_vis = Image.open(vis_path)
            # image_vis = cv2.imread(vis_path, cv2.IMREAD_GRAYSCALE)
            image_inf = cv2.imread(ir_path, 0)
            image_gt = Image.open(gt_path)


            image_vis = np.array(image_vis)
            image_vis = (
                np.asarray(Image.fromarray(image_vis), dtype=np.float32).transpose(
                    (2, 0, 1)
                )
                / 255.0
            )

            image_gt = np.array(image_gt)
            image_gt = (
                np.asarray(Image.fromarray(image_gt), dtype=np.float32).transpose(
                    (2, 0, 1)
                )
                / 255.0
            )

            image_ir = np.asarray(Image.fromarray(image_inf), dtype=np.float32) / 255.0
            image_ir = np.expand_dims(image_ir, axis=0)
            return (
                torch.tensor(image_ir),
                torch.tensor(image_vis),
                torch.tensor(image_gt)
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

    def __len__(self):
        # print(self.length)
        return self.length


class GetDataset_type3(torch.utils.data.Dataset):
    def __init__(self, split, ir_path=None, vi_path=None, gt_path=None, method_path=None, gt_ir_path=None):
        super(GetDataset_type3, self).__init__()

        if split == 'train':
            data_dir_ir = ir_path
            data_dir_vis = vi_path
            data_dir_gt = gt_path
            data_dir_method = method_path
            data_dir_gt_ir = gt_ir_path
            self.filepath_vis, self.filenames_vis = prepare_data_path(data_dir_vis)
            self.filepath_ir, self.filenames_ir = prepare_data_path(data_dir_ir)
            self.filepath_gt, self.filenames_gt = prepare_data_path(data_dir_gt)
            self.filepath_method, self.filenames_method = prepare_data_path(data_dir_method)
            self.filepath_gt_ir, self.filenames_gt_ir = prepare_data_path(data_dir_gt_ir)

            self.split = split
            self.length = min(len(self.filenames_vis), len(self.filenames_ir), len(self.filenames_gt),
                              len(self.filenames_method), len(self.filenames_gt_ir))

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
            gt_path = self.filepath_gt[index]
            method_path = self.filepath_method[index]
            gt_ir_path = self.filepath_gt_ir[index]

            image_vis = Image.open(vis_path)
            # image_vis = cv2.imread(vis_path, cv2.IMREAD_GRAYSCALE)
            image_inf = cv2.imread(ir_path, 0)
            image_gt = Image.open(gt_path)
            image_method = Image.open(method_path)
            image_gt_ir = cv2.imread(gt_ir_path, 0)


            image_vis = np.array(image_vis)
            image_vis = (
                np.asarray(Image.fromarray(image_vis), dtype=np.float32).transpose(
                    (2, 0, 1)
                )
                / 255.0
            )

            image_gt = np.array(image_gt)
            image_gt = (
                np.asarray(Image.fromarray(image_gt), dtype=np.float32).transpose(
                    (2, 0, 1)
                )
                / 255.0
            )

            image_method = np.array(image_method)
            image_method = (
                np.asarray(Image.fromarray(image_method), dtype=np.float32).transpose(
                    (2, 0, 1)
                )
                / 255.0
            )

            image_ir = np.asarray(Image.fromarray(image_inf), dtype=np.float32) / 255.0
            image_ir = np.expand_dims(image_ir, axis=0)

            image_gt_ir = np.asarray(Image.fromarray(image_gt_ir), dtype=np.float32) / 255.0
            image_gt_ir = np.expand_dims(image_gt_ir, axis=0)

            return (
                torch.tensor(image_ir),
                torch.tensor(image_vis),
                torch.tensor(image_gt),
                torch.tensor(image_method),
                torch.tensor(image_gt_ir)
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

    def __len__(self):
        # print(self.length)
        return self.length


class GetDataset_type4(torch.utils.data.Dataset):
    def __init__(self, split, size,
                 haze=None, haze_GT=None, haze_ir=None, haze_ir_GT=None,
                 rain=None, rain_GT=None, rain_ir=None, rain_ir_GT=None,
                 snow=None, snow_GT=None, snow_ir=None, snow_ir_GT=None):
        super(GetDataset_type4, self).__init__()

        if split == 'train':
            self.filepath_haze, self.filenames_haze = prepare_data_path(haze)
            self.filepath_haze_GT, self.filenames_haze_GT = prepare_data_path(haze_GT)
            self.filepath_haze_ir, self.filenames_haze_ir = prepare_data_path(haze_ir)
            self.filepath_haze_ir_GT, self.filenames_haze_ir_GT = prepare_data_path(haze_ir_GT)

            self.filepath_rain, self.filenames_rain = prepare_data_path(rain)
            self.filepath_rain_GT, self.filenames_rain_GT = prepare_data_path(rain_GT)
            self.filepath_rain_ir, self.filenames_rain_ir = prepare_data_path(rain_ir)
            self.filepath_rain_ir_GT, self.filenames_rain_ir_GT = prepare_data_path(rain_ir_GT)

            self.filepath_snow, self.filenames_snow = prepare_data_path(snow)
            self.filepath_snow_GT, self.filenames_snow_GT = prepare_data_path(snow_GT)
            self.filepath_snow_ir, self.filenames_snow_ir = prepare_data_path(snow_ir)
            self.filepath_snow_ir_GT, self.filenames_snow_ir_GT = prepare_data_path(snow_ir_GT)

            self.split = split
            self.length = len(self.filepath_haze)
            self.transform = T.Compose([T.RandomCrop(size),
                                    T.RandomHorizontalFlip(0.5),
                                    T.RandomVerticalFlip(0.5),
                                    T.ToTensor()])


    def __getitem__(self, index):
        if self.split=='train':
            haze_path = self.filepath_haze[index]
            haze_GT_path = self.filepath_haze_GT[index]
            haze_ir_path = self.filepath_haze_ir[index]
            haze_ir_GT_path = self.filepath_haze_ir_GT[index]

            rain_path = self.filepath_rain[index]
            rain_GT_path = self.filepath_rain_GT[index]
            rain_ir_path = self.filepath_rain_ir[index]
            rain_ir_GT_path = self.filepath_rain_ir_GT[index]

            snow_path = self.filepath_snow[index]
            snow_GT_path = self.filepath_snow_GT[index]
            snow_ir_path = self.filepath_snow_ir[index]
            snow_ir_GT_path = self.filepath_snow_ir_GT[index]

            image_haze = Image.open(haze_path).convert(mode='RGB')
            image_haze_GT = Image.open(haze_GT_path).convert(mode='RGB')
            image_haze_ir = Image.open(haze_ir_path).convert(mode='L')
            image_haze_ir_GT = Image.open(haze_ir_GT_path).convert(mode='L')

            image_rain = Image.open(rain_path).convert(mode='RGB')
            image_rain_GT = Image.open(rain_GT_path).convert(mode='RGB')
            image_rain_ir = Image.open(rain_ir_path).convert(mode='L')
            image_rain_ir_GT = Image.open(rain_ir_GT_path).convert(mode='L')

            image_snow = Image.open(snow_path).convert(mode='RGB')
            image_snow_GT = Image.open(snow_GT_path).convert(mode='RGB')
            image_snow_ir = Image.open(snow_ir_path).convert(mode='L')
            image_snow_ir_GT = Image.open(snow_ir_GT_path).convert(mode='L')

            image_haze, image_haze_GT, image_haze_ir, image_haze_ir_GT = self.transform(image_haze, image_haze_GT, image_haze_ir, image_haze_ir_GT)
            image_rain, image_rain_GT, image_rain_ir, image_rain_ir_GT = self.transform(image_rain, image_rain_GT, image_rain_ir, image_rain_ir_GT)
            image_snow, image_snow_GT, image_snow_ir, image_snow_ir_GT = self.transform(image_snow, image_snow_GT, image_snow_ir, image_snow_ir_GT)

            return [
                (image_haze, image_haze_GT, image_haze_ir, image_haze_ir_GT),
                (image_rain, image_rain_GT, image_rain_ir, image_rain_ir_GT),
                (image_snow, image_snow_GT, image_snow_ir, image_snow_ir_GT),
            ]

    def __len__(self):
        return self.length

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