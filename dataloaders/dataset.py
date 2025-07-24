import os
import random
import itertools

import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from torchvision import transforms

import numpy as np
from glob import glob
import h5py
from scipy import ndimage
from skimage import transform as sk_trans
from scipy.ndimage import rotate, zoom
import pdb
import matplotlib.pyplot as plt



class BaseDataSets(Dataset):
    def __init__(self, base_dir=None, split='train', num=None, transform=None):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        if self.split == 'train':
            with open(self._base_dir + '/train_slices.list', 'r') as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace('\n', '') for item in self.sample_list]

        elif self.split == 'val':
            with open(self._base_dir + '/val.list', 'r') as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace('\n', '') for item in self.sample_list]
        if num is not None and self.split == "train":
            self.sample_list = self.sample_list[:num]
        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        if self.split == "train":
            h5f = h5py.File(self._base_dir + "/data/slices/{}.h5".format(case), 'r')
        else:
            h5f = h5py.File(self._base_dir + "/data/{}.h5".format(case), 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        sample = {'image': image, 'label': label}
        if self.split == "train":
            sample = self.transform(sample)
        # sample["idx"] = idx
        sample['case'] = case
        return sample

def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        # ind = random.randrange(0, img.shape[0])
        # image = img[ind, ...]
        # label = lab[ind, ...]
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))
        sample = {'image': image, 'label': label}
        return sample


class LAHeart(Dataset):
    """ LA Dataset """
    def __init__(self, base_dir=None, split='train', num=None, transform=None):
        self._base_dir = base_dir
        self.transform = transform
        self.sample_list = []

        train_path = self._base_dir+'/train.list'
        test_path = self._base_dir+'/test.list'

        if split=='train':
            with open(train_path, 'r') as f:
                self.image_list = f.readlines()
        elif split == 'test':
            with open(test_path, 'r') as f:
                self.image_list = f.readlines()

        self.image_list = [item.replace('\n','') for item in self.image_list]
        if num is not None:
            self.image_list = self.image_list[:num]
        print("total {} samples".format(len(self.image_list)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        h5f = h5py.File(self._base_dir + "/2018LA_Seg_Training Set/" + image_name + "/mri_norm2.h5", 'r')
        # h5f = h5py.File(self._base_dir+"/"+image_name+"/mri_norm2.h5", 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)

        return sample

class Pancreas(Dataset):
    """ Pancreas Dataset """
    def __init__(self, base_dir=None, split='train', num=None, transform=None):
        self._base_dir = base_dir
        self.transform = transform
        self.sample_list = []

        train_path = self._base_dir+'/train.list'
        test_path = self._base_dir+'/test.list'

        if split=='train':
            with open(train_path, 'r') as f:
                self.image_list = f.readlines()
        elif split == 'test':
            with open(test_path, 'r') as f:
                self.image_list = f.readlines()

        self.image_list = [item.replace('\n','') for item in self.image_list]
        if num is not None:
            self.image_list = self.image_list[:num]
        print("total {} samples".format(len(self.image_list)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        h5f = h5py.File(self._base_dir + "/Pancreas_h5/" + image_name + "_norm.h5", 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)

        return sample



class Resize(object):

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        (w, h, d) = image.shape
        label = label.astype(np.bool)
        image = sk_trans.resize(image, self.output_size, order = 1, mode = 'constant', cval = 0)
        label = sk_trans.resize(label, self.output_size, order = 0)
        assert(np.max(label) == 1 and np.min(label) == 0)
        assert(np.unique(label).shape[0] == 2)
        
        return {'image': image, 'label': label}
    
    
class CenterCrop(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        (w, h, d) = image.shape

        w1 = int(round((w - self.output_size[0]) / 2.))
        h1 = int(round((h - self.output_size[1]) / 2.))
        d1 = int(round((d - self.output_size[2]) / 2.))

        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]

        return {'image': image, 'label': label}


class RandomCrop(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size, with_sdf=False):
        self.output_size = output_size
        self.with_sdf = with_sdf

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if self.with_sdf:
            sdf = sample['sdf']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            if self.with_sdf:
                sdf = np.pad(sdf, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        (w, h, d) = image.shape

        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])
        d1 = np.random.randint(0, d - self.output_size[2])

        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        if self.with_sdf:
            sdf = sdf[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
            return {'image': image, 'label': label, 'sdf': sdf}
        else:
            return {'image': image, 'label': label}

        

class RandomCrop_Test(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size, with_sdf=False):
        self.output_size = output_size
        self.with_sdf = with_sdf
        self.num_iter = 1

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        for indice_d in range(image.shape[2]):
             # 可视化
            image_clipped = np.clip(image[:, :, indice_d], 0, 1)

            # 然后缩放到[0,255]
            image_visual = (image_clipped * 255).astype(np.uint8)

            # # 创建RGB图像（将灰度图转换为3通道）
            # image_rgb = np.stack([image_visual, image_visual, image_visual], axis=-1)
            # # 将标签区域设置为红色（保持原图亮度，但添加红色通道）
            # mask = label[:, :, indice_d] > 0

            # alpha = 0.3  # 透明度
            # image_rgb[mask, 0] = np.clip(image_rgb[mask, 0] * (1-alpha) + 255 * alpha, 0, 255)

            plt.figure(figsize=(10, 4))
            plt.subplot(1, 2, 1)
            # 子图1：标准灰度显示
            plt.imshow(image_visual, cmap='gray', vmin=0, vmax=255)
            plt.colorbar()
            plt.subplot(1, 2, 2)
            plt.imshow(label[:, :, indice_d], cmap='jet', vmin=0, vmax=1)
            plt.colorbar()
            # plt.imsave('images/iter{} label{}.png'.format(self.num_iter, label.sum()), cmap='gray', vmin=0, vmax=255)
            plt.savefig("images/iter{}_label{}.png".format(self.num_iter, label[:,:,indice_d].sum()), bbox_inches='tight', dpi=150)
            plt.close()
            self.num_iter+=1


        if self.with_sdf:
            sdf = sample['sdf']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            if self.with_sdf:
                sdf = np.pad(sdf, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        (w, h, d) = image.shape

        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])
        d1 = np.random.randint(0, d - self.output_size[2])

        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        if self.with_sdf:
            sdf = sdf[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
            return {'image': image, 'label': label, 'sdf': sdf}
        else:
            return {'image': image, 'label': label}


class RandomRotFlip(object):
    """
    Crop randomly flip the dataset in a sample
    Args:
    output_size (int): Desired output size
    """

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image, label = random_rot_flip(image, label)

        return {'image': image, 'label': label}

class RandomRot(object):
    """
    Crop randomly flip the dataset in a sample
    Args:
    output_size (int): Desired output size
    """

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image, label = random_rotate(image, label)

        return {'image': image, 'label': label}


class RandomNoise(object):
    def __init__(self, mu=0, sigma=0.1):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        noise = np.clip(self.sigma * np.random.randn(image.shape[0], image.shape[1], image.shape[2]), -2*self.sigma, 2*self.sigma)
        noise = noise + self.mu
        image = image + noise
        return {'image': image, 'label': label}


# class CreateOnehotLabel(object):
#     def __init__(self, num_classes):
#         self.num_classes = num_classes

#     def __call__(self, sample):
#         image, label = sample['image'], sample['label']
#         # print(label.shape)
#         onehot_label = np.zeros((self.num_classes, label.shape[0], label.shape[1]), dtype=np.float32)
#         for i in range(self.num_classes):
#             onehot_label[i, :, :] = (label == i).type(torch.float32)
#         return {'image': image, 'label': label, 'onehot_label': onehot_label}
class CreateOnehotLabel(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
    
    def __call__(self, sample):
        result = {}
        
        # 处理所有 image 相关字段（保持不变）
        for key in sample:
            if key.startswith('image'):
                result[key] = sample[key]
        
        # 处理所有 label 相关字段，生成对应的 onehot
        for key in sample:
            if key.startswith('label'):
                label = sample[key]
                onehot_key = f'onehot_{key}'  # label -> onehot_label, label_strong -> onehot_label_strong
                
                onehot_label = np.zeros((self.num_classes, label.shape[0], label.shape[1]), dtype=np.float32)
                for i in range(self.num_classes):
                    onehot_label[i, :, :] = (label == i).type(torch.float32)
                
                result[key] = label
                result[onehot_key] = onehot_label
        
        return result

class CreateOnehotLabel3D(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        onehot_label = np.zeros((self.num_classes, label.shape[0], label.shape[1], label.shape[2]), dtype=np.float32)
        for i in range(self.num_classes):
            onehot_label[i, :, :, :] = (label == i).astype(np.float32)
        return {'image': image, 'label': label,'onehot_label':onehot_label}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample['image']

        image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2]).astype(np.float32)
        if 'onehot_label' in sample:
            return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long(),
                    'onehot_label': torch.from_numpy(sample['onehot_label']).long()}
        else:
            return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long()}



class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                    grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


class ThreeStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch + primary_batch
            for (primary_batch, secondary_batch, primary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                    grouper(secondary_iter, self.secondary_batch_size),
                    grouper(primary_iter, self.primary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


class WeakStrongAugment(object):
    """returns weakly and strongly augmented images
    Args:
        object (tuple): output size of network
    """
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        if random.random() > 0.5:  
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        # weak augmentation is rotation / flip
        x, y = image.shape
        image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y),order=0)  
        label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)

        # strong augmentation is color jitter
        image_strong, label_strong = cutout_gray(image,label, p=0.5)
        image_strong = color_jitter(image_strong).type("torch.FloatTensor")
        # image_strong = blur(image, p=0.5)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        # image_strong = torch.from_numpy(image_strong.astype(np.float32)).unsqueeze(0)

        label = torch.from_numpy(label.astype(np.uint8))
        label_strong = torch.from_numpy(label_strong.astype(np.uint8))
        sample = {
            "image": image,
            "image_strong": image_strong,
            "label": label,
            "label_strong": label_strong
        }
        return sample

    def resize(self, image):
        x, y = image.shape
        return zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)


class WeakStrongAugment_LA(object):
    """returns weakly and strongly augmented 3D images
    Args:
        output_size (tuple): output size of network (w, h, d)
    """
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]

        # image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        # label = torch.from_numpy(label.astype(np.uint8))
        # if torch.cuda.is_available():
        #     image = image.cuda()
        #     label = label.cuda()
        
        # Random crop to fixed size (like RandomCrop)
        image, label = self.random_crop_3d(image, label)
        
        # Apply weak augmentation (rotation/flip) with probability
        # if random.random() > 0.5:  
        #     image, label = random_rot_flip_3d(image, label)
        if random.random() > 0.5:
            image, label = random_rotate_3d(image, label)

        # Strong augmentation (cutout + color jitter)
        # image_strong, label_strong = cutout_gray_3d(image, label, p=0.5)
        # image_strong = color_jitter_3d(image_strong)
        
        # Convert to tensors
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        # image_strong = torch.from_numpy(image_strong.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))
        # label_strong = torch.from_numpy(label_strong.astype(np.uint8))
        
        sample = {
            "image": image,
            # "image_strong": image_strong,
            "label": label,
            # "label_strong": label_strong
        }
        return sample

    def random_crop_3d(self, image, label):
        """3D random crop to fixed output size (like RandomCrop)"""
        w, h, d = image.shape
        
        # Pad if necessary (same logic as original RandomCrop)
        if w <= self.output_size[0] or h <= self.output_size[1] or d <= self.output_size[2]:
            pw = max((self.output_size[0] - w) // 2 + 3, 0)
            ph = max((self.output_size[1] - h) // 2 + 3, 0)
            pd = max((self.output_size[2] - d) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
        
        # Get new dimensions after padding
        w, h, d = image.shape
        
        # Random crop
        w1 = random.randint(0, w - self.output_size[0])
        h1 = random.randint(0, h - self.output_size[1])
        d1 = random.randint(0, d - self.output_size[2])
        
        image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        
        return image, label

# 需要实现的3D增强函数
def random_rot_flip_3d(image, label):
    """3D random rotation and flip"""
    # 随机选择旋转轴 (0, 1, 2对应x, y, z轴)
    axis = random.choice([0, 1, 2])
    k = random.randint(0, 3)  # 旋转次数
    image = np.rot90(image, k, axes=(axis, (axis+1)%3))
    label = np.rot90(label, k, axes=(axis, (axis+1)%3))
    
    # 随机翻转
    if random.random() > 0.5:
        flip_axis = random.choice([0, 1, 2])
        image = np.flip(image, axis=flip_axis)
        label = np.flip(label, axis=flip_axis)
    
    return image, label

def random_rotate_3d(image, label):
    """3D random rotation with small angles"""
    from scipy.ndimage import rotate
    angle = random.uniform(-15, 15)
    axes = random.choice([(0, 1), (0, 2), (1, 2)])
    
    image = rotate(image, angle, axes=axes, reshape=False, order=0)
    label = rotate(label, angle, axes=axes, reshape=False, order=0)
    
    return image, label

def cutout_gray_3d(image, label, p=0.5):
    """3D cutout augmentation"""
    if random.random() > p:
        return image, label
    
    image_out = image.copy()
    label_out = label.copy()
    
    # 随机选择cutout的3D区域
    h, w, d = image.shape
    cut_h = random.randint(h//8, h//4)
    cut_w = random.randint(w//8, w//4) 
    cut_d = random.randint(d//8, d//4)
    
    start_h = random.randint(0, h - cut_h)
    start_w = random.randint(0, w - cut_w)
    start_d = random.randint(0, d - cut_d)
    
    # 将选定区域设为0
    image_out[start_h:start_h+cut_h, start_w:start_w+cut_w, start_d:start_d+cut_d] = 0
    
    return image_out, label_out

def color_jitter_3d(image):
    """3D color jitter (brightness, contrast adjustment)"""
    # 亮度调整
    brightness_factor = random.uniform(0.8, 1.2)
    image = image * brightness_factor
    
    # 对比度调整
    contrast_factor = random.uniform(0.8, 1.2)
    mean = np.mean(image)
    image = (image - mean) * contrast_factor + mean
    
    # 确保值在合理范围内
    image = np.clip(image, 0, 1)
    
    return image


import numpy as np
from scipy.ndimage import rotate
from copy import deepcopy
import warnings

class RandomFlip:
    """随机翻转"""
    def __init__(self, random_state, axis_prob=0.5, **kwargs):
        assert random_state is not None, 'RandomState cannot be None'
        self.random_state = random_state
        self.axes = (0, 1, 2)
        self.axis_prob = axis_prob

    def __call__(self, m): 
        assert m.ndim in [3, 4], 'Supports only 3D (DxHxW) or 4D (CxDxHxW) images'

        for axis in self.axes:
            if self.random_state.uniform() > self.axis_prob:
                if m.ndim == 3:
                    m = np.flip(m, axis)
                else:
                    channels = [np.flip(m[c], axis) for c in range(m.shape[0])]
                    m = np.stack(channels, axis=0)
        return m

class RandomRotate:
    """随机旋转"""
    def __init__(self, random_state, angle_spectrum=30, axes=None, mode='reflect', order=0, **kwargs):
        if axes is None:
            axes = [(1, 0), (2, 1), (2, 0)]
        else:
            assert isinstance(axes, list) and len(axes) > 0

        self.random_state = random_state
        self.angle_spectrum = angle_spectrum
        self.axes = axes
        self.mode = mode
        self.order = order

    def __call__(self, m):
        axis = self.axes[self.random_state.randint(len(self.axes))]
        angle = self.random_state.randint(-self.angle_spectrum, self.angle_spectrum)

        if m.ndim == 3:
            m = rotate(m, angle, axes=axis, reshape=False, order=self.order, mode=self.mode, cval=-1)
        else:
            channels = [rotate(m[c], angle, axes=axis, reshape=False, order=self.order, mode=self.mode, cval=-1) for c
                        in range(m.shape[0])]
            m = np.stack(channels, axis=0)
        return m

class RandomContrast:
    """随机对比度调整"""
    def __init__(self, random_state, alpha=(0.5, 1.5), mean=0.0, execution_probability=0.1, **kwargs):
        self.random_state = random_state
        assert len(alpha) == 2
        self.alpha = alpha
        self.mean = mean
        self.execution_probability = execution_probability

    def __call__(self, m):
        if self.random_state.uniform() < self.execution_probability:
            alpha = self.random_state.uniform(self.alpha[0], self.alpha[1])
            result = self.mean + alpha * (m - self.mean)
            return np.clip(result, -1, 1)
        return m
        

class DualAugmentTransform:
    """
    双增强Transform类 - 直接集成到transforms.Compose中
    
    输入: {'image': tensor/numpy, 'label': tensor/numpy}
    输出: {'image': tensor, 'label': tensor, 'image_strong': tensor, 'label_strong': tensor}
    """
    
    def __init__(self, base_random_state=None, **kwargs):
        """
        Args:
            base_random_state: 基础随机状态
            **kwargs: 可以传入weak_config, strong_config来自定义配置
        """
        if base_random_state is None:
            base_random_state = np.random.RandomState(42)
        
        self.base_random_state = base_random_state
        
        # 弱增强配置
        self.weak_config = {
            'flip': {'axis_prob': 0.3},
            'rotate': {'angle_spectrum': 15, 'mode': 'reflect', 'order': 0},
            'contrast': {'alpha': (0.8, 1.2), 'mean': 0.0, 'execution_probability': 0.1}
        }
        
        # 强增强配置
        self.strong_config = {
            'flip': {'axis_prob': 0.5},
            'rotate': {'angle_spectrum': 45, 'mode': 'reflect', 'order': 0},
            'contrast': {'alpha': (0.5, 1.8), 'mean': 0.0, 'execution_probability': 0.3}
        }
        
        # 允许自定义配置
        if 'weak_config' in kwargs:
            self._update_config(self.weak_config, kwargs['weak_config'])
        if 'strong_config' in kwargs:
            self._update_config(self.strong_config, kwargs['strong_config'])
    
    def _update_config(self, base_config, update_config):
        """递归更新配置"""
        for key, value in update_config.items():
            if key in base_config and isinstance(base_config[key], dict) and isinstance(value, dict):
                self._update_config(base_config[key], value)
            else:
                base_config[key] = value
    
    def __call__(self, sample):
        """
        Transform调用函数
        
        Args:
            sample: 可以是以下格式之一:
                   - {'image': array, 'label': array}  
                   - (image_array, label_array)
                   - image_array (只有图像)
        
        Returns:
            dict: {
                'image': torch.Tensor,      # 弱增强或原始
                'label': torch.Tensor,      # 弱增强或原始
                'image_strong': torch.Tensor,  # 强增强
                'label_strong': torch.Tensor   # 强增强
            }
        """
        
        # 统一输入格式
        if isinstance(sample, dict):
            image = sample['image']
            label = sample.get('label', None)
        elif isinstance(sample, (tuple, list)) and len(sample) == 2:
            image, label = sample
        else:
            image = sample
            label = None
        
        # 转换为numpy格式进行增强
        # image_np, label_np: (112, 112, 80), (112, 112, 80)
        image_np = np.array(image)
        label_np = np.array(label)

        # 生成随机种子
        weak_seed = self.base_random_state.randint(0, 2**31)
        strong_seed = self.base_random_state.randint(0, 2**31)
        
        # 弱增强（或保持原样）
        weak_image, weak_label = self._apply_augmentation(
            image_np, label_np, self.weak_config, weak_seed
        )
        
        # 强增强
        strong_image, strong_label = self._apply_augmentation(
            image_np, label_np, self.strong_config, strong_seed
        )
        
        # 转换回tensor格式
        result = {
            'image': self._to_tensor(weak_image),
            'image_strong': self._to_tensor(strong_image)
        }
        
        if label is not None:
            result['label'] = self._to_tensor(weak_label, is_label=True)
            result['label_strong'] = self._to_tensor(strong_label, is_label=True)
        
        return result
    
    def _to_tensor(self, data, is_label=False):
        """转换为tensor格式"""
        if data is None:
            return None
        
        tensor = torch.from_numpy(data.copy())
        
        # 添加batch维度和channel维度: (H, W, D) -> (1, 1, H, W, D)
        
        
        if is_label:
            return tensor.long()
        else:
            tensor = tensor.unsqueeze(0)
            return tensor.float()
        
    def _create_augmenters(self, config, seed):
        """根据配置创建增强器"""
        flip_random = np.random.RandomState(seed)
        rotate_random = np.random.RandomState(seed + 1)
        contrast_random = np.random.RandomState(seed + 2)
        
        return {
            'flip': RandomFlip(flip_random, **config['flip']),
            'rotate': RandomRotate(rotate_random, **config['rotate']),
            'contrast': RandomContrast(contrast_random, **config['contrast'])
        }
    
    def _apply_augmentation(self, image, label, config, seed):
        """应用增强"""
        augmenters = self._create_augmenters(config, seed)
        
        # 复制输入数据
        aug_image = image.copy()
        aug_label = label.copy() if label is not None else None

        if label is not None:
            # 对几何变换使用相同的随机状态以保证图像和标签的一致性
            geom_seed = seed + 10
            
            # 翻转 - 图像和标签使用相同随机状态
            flip_random_img = np.random.RandomState(geom_seed)
            flip_random_label = np.random.RandomState(geom_seed)
            flip_img = RandomFlip(flip_random_img, **config['flip'])
            flip_label = RandomFlip(flip_random_label, **config['flip'])
            aug_image = flip_img(aug_image)
            aug_label = flip_label(aug_label)
            
            # 旋转 - 标签使用最近邻插值
            rotate_config_label = config['rotate'].copy()
            rotate_config_label['order'] = 0  # 标签使用最近邻插值
            
            rotate_random_img = np.random.RandomState(geom_seed + 1)
            rotate_random_label = np.random.RandomState(geom_seed + 1)
            rotate_img = RandomRotate(rotate_random_img, **config['rotate'])
            rotate_label = RandomRotate(rotate_random_label, **rotate_config_label)
            aug_image = rotate_img(aug_image)
            aug_label = rotate_label(aug_label)
            
        else:
            # 只有图像时正常应用变换
            aug_image = augmenters['flip'](aug_image)
            aug_image = augmenters['rotate'](aug_image)
        
        # 对比度调整只应用于图像
        aug_image = augmenters['contrast'](aug_image)
        
        return aug_image, aug_label


def cutout_gray(img, mask, p=0.5, size_min=0.02, size_max=0.4, ratio_1=0.3, ratio_2=1/0.3, value_min=0, value_max=1, pixel_level=True):
    if random.random() < p:
        img = np.array(img)
        mask = np.array(mask)

        img_h, img_w = img.shape

        while True:
            size = np.random.uniform(size_min, size_max) * img_h * img_w
            ratio = np.random.uniform(ratio_1, ratio_2)
            erase_w = int(np.sqrt(size / ratio))
            erase_h = int(np.sqrt(size * ratio))
            x = np.random.randint(0, img_w)
            y = np.random.randint(0, img_h)

            if x + erase_w <= img_w and y + erase_h <= img_h:
                break

        if pixel_level:
            value = np.random.randint(value_min, value_max + 1, (erase_h, erase_w))
        else:
            value = np.random.randint(value_min, value_max + 1)

        img[y:y + erase_h, x:x + erase_w] = value
        mask[y:y + erase_h, x:x + erase_w] = 0

    return img, mask


def color_jitter(image):
    if not torch.is_tensor(image):
        np_to_tensor = transforms.ToTensor()
        image = np_to_tensor(image)
    # s is the strength of color distortion.
    s = 1.0
    jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    return jitter(image)


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)
