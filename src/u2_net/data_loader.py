from __future__ import print_function, division

import numpy as np
import torch

from skimage import io, transform, color
from torch.utils.data import Dataset


class RescaleT(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        imidx, image, label = sample['imidx'], sample['image'], sample['label']
        img = transform.resize(
            image, (self.output_size, self.output_size), mode='constant'
        )
        lbl = transform.resize(
            label, (self.output_size, self.output_size), mode='constant', order=0, preserve_range=True
        )
        return dict(imidx=imidx, image=img, label=lbl)


class ToTensorLab(object):

    def __init__(self, flag=0):
        self.flag = flag

    # noinspection PyUnresolvedReferences
    def __call__(self, sample):

        imidx, image, label = sample['imidx'], sample['image'], sample['label']

        temp_label = np.zeros(label.shape)

        if np.max(label) < 1e-6:
            label = label
        else:
            label = label / np.max(label)

        # Change the color space
        if self.flag == 2:  # With RGB and Lab colors
            tmp_image = np.zeros((image.shape[0], image.shape[1], 6))
            tmp_image_t = np.zeros((image.shape[0], image.shape[1], 3))
            if image.shape[2] == 1:
                tmp_image_t[:, :, 0] = image[:, :, 0]
                tmp_image_t[:, :, 1] = image[:, :, 0]
                tmp_image_t[:, :, 2] = image[:, :, 0]
            else:
                tmp_image_t = image
            tmp_image_tl = color.rgb2lab(tmp_image_t)

            # Normalize image to range [0,1]
            tmp_image[:, :, 0] = (tmp_image_t[:, :, 0] - np.min(tmp_image_t[:, :, 0])) / (
                    np.max(tmp_image_t[:, :, 0]) - np.min(tmp_image_t[:, :, 0]))
            tmp_image[:, :, 1] = (tmp_image_t[:, :, 1] - np.min(tmp_image_t[:, :, 1])) / (
                    np.max(tmp_image_t[:, :, 1]) - np.min(tmp_image_t[:, :, 1]))
            tmp_image[:, :, 2] = (tmp_image_t[:, :, 2] - np.min(tmp_image_t[:, :, 2])) / (
                    np.max(tmp_image_t[:, :, 2]) - np.min(tmp_image_t[:, :, 2]))
            tmp_image[:, :, 3] = (tmp_image_tl[:, :, 0] - np.min(tmp_image_tl[:, :, 0])) / (
                    np.max(tmp_image_tl[:, :, 0]) - np.min(tmp_image_tl[:, :, 0]))
            tmp_image[:, :, 4] = (tmp_image_tl[:, :, 1] - np.min(tmp_image_tl[:, :, 1])) / (
                    np.max(tmp_image_tl[:, :, 1]) - np.min(tmp_image_tl[:, :, 1]))
            tmp_image[:, :, 5] = (tmp_image_tl[:, :, 2] - np.min(tmp_image_tl[:, :, 2])) / (
                    np.max(tmp_image_tl[:, :, 2]) - np.min(tmp_image_tl[:, :, 2]))

            tmp_image[:, :, 0] = (tmp_image[:, :, 0] - np.mean(tmp_image[:, :, 0])) / np.std(tmp_image[:, :, 0])
            tmp_image[:, :, 1] = (tmp_image[:, :, 1] - np.mean(tmp_image[:, :, 1])) / np.std(tmp_image[:, :, 1])
            tmp_image[:, :, 2] = (tmp_image[:, :, 2] - np.mean(tmp_image[:, :, 2])) / np.std(tmp_image[:, :, 2])
            tmp_image[:, :, 3] = (tmp_image[:, :, 3] - np.mean(tmp_image[:, :, 3])) / np.std(tmp_image[:, :, 3])
            tmp_image[:, :, 4] = (tmp_image[:, :, 4] - np.mean(tmp_image[:, :, 4])) / np.std(tmp_image[:, :, 4])
            tmp_image[:, :, 5] = (tmp_image[:, :, 5] - np.mean(tmp_image[:, :, 5])) / np.std(tmp_image[:, :, 5])

        elif self.flag == 1:  # With Lab color
            tmp_image = np.zeros((image.shape[0], image.shape[1], 3))

            if image.shape[2] == 1:
                tmp_image[:, :, 0] = image[:, :, 0]
                tmp_image[:, :, 1] = image[:, :, 0]
                tmp_image[:, :, 2] = image[:, :, 0]
            else:
                tmp_image = image

            tmp_image = color.rgb2lab(tmp_image)

            tmp_image[:, :, 0] = (tmp_image[:, :, 0] - np.min(tmp_image[:, :, 0])) / (
                    np.max(tmp_image[:, :, 0]) - np.min(tmp_image[:, :, 0]))
            tmp_image[:, :, 1] = (tmp_image[:, :, 1] - np.min(tmp_image[:, :, 1])) / (
                    np.max(tmp_image[:, :, 1]) - np.min(tmp_image[:, :, 1]))
            tmp_image[:, :, 2] = (tmp_image[:, :, 2] - np.min(tmp_image[:, :, 2])) / (
                    np.max(tmp_image[:, :, 2]) - np.min(tmp_image[:, :, 2]))

            tmp_image[:, :, 0] = (tmp_image[:, :, 0] - np.mean(tmp_image[:, :, 0])) / np.std(tmp_image[:, :, 0])
            tmp_image[:, :, 1] = (tmp_image[:, :, 1] - np.mean(tmp_image[:, :, 1])) / np.std(tmp_image[:, :, 1])
            tmp_image[:, :, 2] = (tmp_image[:, :, 2] - np.mean(tmp_image[:, :, 2])) / np.std(tmp_image[:, :, 2])

        else:  # With rgb color
            tmp_image = np.zeros((image.shape[0], image.shape[1], 3))
            image = image / np.max(image)
            if image.shape[2] == 1:
                tmp_image[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
                tmp_image[:, :, 1] = (image[:, :, 0] - 0.485) / 0.229
                tmp_image[:, :, 2] = (image[:, :, 0] - 0.485) / 0.229
            else:
                tmp_image[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
                tmp_image[:, :, 1] = (image[:, :, 1] - 0.456) / 0.224
                tmp_image[:, :, 2] = (image[:, :, 2] - 0.406) / 0.225

        temp_label[:, :, 0] = label[:, :, 0]

        # Change the r,g,b to b,r,g from [0,255] to [0,1]
        tmp_image = tmp_image.transpose((2, 0, 1))
        temp_label = label.transpose((2, 0, 1))

        return dict(
            imidx=torch.from_numpy(imidx),
            image=torch.from_numpy(tmp_image),
            label=torch.from_numpy(temp_label)
        )


class SalObjDataSet(Dataset):

    def __init__(self, img_name_list, lbl_name_list, trans=None):
        self.image_name_list = img_name_list
        self.label_name_list = lbl_name_list
        self.transform = trans

    def __len__(self):
        return len(self.image_name_list)

    # noinspection PyUnresolvedReferences
    def __getitem__(self, idx):
        image = io.imread(self.image_name_list[idx])
        imidx = np.array([idx])

        if 0 == len(self.label_name_list):
            label_3 = np.zeros(image.shape)
        else:
            label_3 = io.imread(self.label_name_list[idx])

        label = np.zeros(label_3.shape[0:2])
        if 3 == len(label_3.shape):
            label = label_3[:, :, 0]
        elif 2 == len(label_3.shape):
            label = label_3

        if 3 == len(image.shape) and 2 == len(label.shape):
            label = label[:, :, np.newaxis]
        elif 2 == len(image.shape) and 2 == len(label.shape):
            image = image[:, :, np.newaxis]
            label = label[:, :, np.newaxis]

        sample = dict(imidx=imidx, image=image, label=label)

        if self.transform:
            sample = self.transform(sample)

        return sample
