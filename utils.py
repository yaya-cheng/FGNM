# coding: utf-8
import os
import numpy as np
from PIL import Image
import tensorflow as tf
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch
import cv2
import matplotlib.pyplot as plt


def get_val_loder(data_path, batch_size):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]
                                     )

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(data_path, transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size = batch_size, shuffle=True,
        num_workers = 8, pin_memory=True)
    return val_loader

def load_images(input_dir, csv_file, index, batch_shape):
    """Images for inception classifier are normalized to be in [-1, 1] interval"""
    images = np.zeros(batch_shape)
    filenames = []
    truelabel = []
    idx = 0
    for i in range(index, min(index + batch_shape[0], 1000)):
        img_obj = csv_file.loc[i]
        ImageID = img_obj['ImageId'] + '.png'
        img_path = os.path.join(input_dir, ImageID)
        images[idx, ...] = np.array(Image.open(img_path)).astype(np.float) / 255.0
        filenames.append(ImageID)
        truelabel.append(img_obj['TrueLabel'])
        idx += 1

    images = images * 2.0 - 1.0
    return images, filenames, truelabel

def save_images(images, filenames, output_dir):
    """Saves images to the output directory."""
    for i, filename in enumerate(filenames):
        # Images for inception classifier are normalized to be in [-1, 1] interval,
        # so rescale them back to [0, 1].
        with tf.gfile.Open(os.path.join(output_dir, filename), 'w') as f:
            image = (images[i, :, :, :] + 1.0) * 0.5
            # image = images[i, :, :, :]
            img = Image.fromarray((image * 255).astype('uint8')).convert('RGB')
            img.save(os.path.join(output_dir, filename), quality=95)

def images_to_FD(input_tensor):
    """Process the image to meet the input requirements of FD"""
    ret = tf.image.resize_images(input_tensor, [224, 224], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    ret = tf.reverse(ret, axis=[-1])  # RGB to BGR
    ret = tf.transpose(ret, [0, 3, 1, 2])
    return ret


def filterVisualizationResults(directory):
    allFiles = os.listdir(directory)
    allFiles = [x.split(".")[0] for x in allFiles if "ORI" not in x]
    results = dict()
    for name in allFiles:
        ori, n, _, m, res = name.split('-')
        if n not in results:
            results[n] = list()
        results[n].append((m, ori != res))
    filtered = set()
    for n, labels in results.items():
        flag = True
        for m, mismatch in labels:
            if m == "sign":
                flag = mismatch
        for m, mismatch in labels:
            if m != "sign":
                if not flag:
                    if mismatch:
                        filtered.add(n)

    allFiles = os.listdir(directory)
    notRemoved = [x for x in allFiles if x.split("-")[1] in filtered]
    allFiles = os.listdir(directory)
    removed = [x for x in allFiles if x not in notRemoved]
    for r in removed:
        os.remove(os.path.join(directory, r))

def frequency_filter(input_pth, output_pth, filename, cat = "high_pass"):
    # img = cv2.imread(input_pth, 0)
    # 傅里叶变换
    # [H, W, C]
    img = cv2.imread(input_pth)
    img = img[:, :, 2]
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
                       
    rows, cols = img.shape
    crow, ccol = int(rows / 2), int(cols / 2)
    
    #屏蔽低频信号   
    if cat == "high_pass":
        mask = np.ones((rows, cols))
        mask[crow-30:crow+30, ccol-30:ccol+30] = 0
    #屏蔽高频信号   
    else:
        mask = np.zeros((rows, cols))
        mask[crow-30:crow+30, ccol-30:ccol+30] = 1
    fshift = fshift * mask
    # iimg = np.abs(np.fft.ifft2(fshift))

    #逆傅里叶变换
    ishift = np.fft.ifftshift(fshift)
    iimg = np.abs(np.fft.ifft2(ishift))
    # iimg = (iimg - np.amin(iimg)) / (np.amax(iimg)- np.amin(iimg))
    save_pth = os.path.join(output_pth, filename)
    cv2.imwrite(save_pth, iimg)
    print("ok")

if __name__ == "__main__":
    # filterVisualizationResults("visualization")
    input_pth = "/home/chengyaya/data/FGNM/dataset/images/0e0f1fd2ed183781.png"
    output_pth = "/home/chengyaya/data/FGNM/visulization"
    frequency_filter(input_pth, output_pth, filename = "0e0f1fd2ed183781.png", cat = "high_pass")
