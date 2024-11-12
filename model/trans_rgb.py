import os
import cv2
import torch
import numpy as np


def read_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images

def YCbCr2RGB(ycbcr):
    # 定义YCrCb到RGB转换矩阵
    mat = torch.tensor([[1.164,  0.000,  1.596],
                        [1.164, -0.392, -0.813],
                        [1.164,  2.017,  0.000]], device=ycbcr.device)

    # 调整维度以匹配矩阵乘法
    ycbcr = ycbcr.permute(0, 2, 3, 1)  # [N, H, W, C]
    ycbcr[:, :, :, 0] = ycbcr[:, :, :, 0] - 16.0 / 255.0
    ycbcr[:, :, :, 1:] = ycbcr[:, :, :, 1:] - 128.0 / 255.0

    # 进行矩阵乘法并转换为RGB
    rgb = torch.matmul(ycbcr, mat.T)
    rgb = rgb.permute(0, 3, 1, 2)  # [N, C, H, W]

    # 将值限制在[0, 1]范围内
    rgb = torch.clamp(rgb, 0, 1)
    return rgb


def replace_y_channel(rgb_image, gray_image):
    # Convert RGB image to YCbCr
    ycbcr_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2YCrCb)

    # Extract Y channel from YCbCr image
    y_channel = ycbcr_image[:, :, 0]

    # Replace Y channel with the gray_image
    ycbcr_image[:, :, 0] = gray_image

    # Convert YCbCr image back to RGB
    replaced_rgb_image = cv2.cvtColor(ycbcr_image, cv2.COLOR_YCrCb2BGR)

    return replaced_rgb_image


def main():
    folder_A = '/public/home/w__y/datasets/Harvard_dataset/test/vi'           # 原彩色图
    folder_B = '/public/home/w__y/code/AWFusion/results2/Harvard'      # 要转RGB图的灰度图
    output_folder = '/public/home/w__y/code/AWFusion/results2/Harvard_color'
    os.makedirs(output_folder, exist_ok=True)
    # Read images from folders A and B
    images_A = read_images_from_folder(folder_A)
    images_B = read_images_from_folder(folder_B)

    # Assuming images in folders A and B have corresponding names and are in the same order
    i = 0
    for filename in os.listdir(folder_B):
    # for i in range(len(images_A)):
        img_A = images_A[i]
        img_B = images_B[i]
        i += 1

        # Convert grayscale image B to single channel (assuming it's already grayscale)
        if len(img_B.shape) > 2 and img_B.shape[2] > 1:
            img_B = cv2.cvtColor(img_B, cv2.COLOR_BGR2GRAY)

        # Resize img_B to match img_A if sizes are different (optional step)
        if img_A.shape[:2] != img_B.shape[:2]:
            img_B = cv2.resize(img_B, (img_A.shape[1], img_A.shape[0]))

        # Replace Y channel in img_A with img_B's grayscale channel
        replaced_image = replace_y_channel(img_A, img_B)

        # Save the modified image
        # output_filename = os.path.basename(os.path.splitext(os.path.join(folder_A, images_A[i]))[0]) + '_modified.jpg'
        cv2.imwrite(os.path.join(output_folder, filename), replaced_image)

        print(f"Processed and saved: {os.path.join(output_folder, filename)}")


if __name__ == "__main__":
    main()
