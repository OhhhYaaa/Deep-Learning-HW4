import torch
from torch.utils.data import Dataset
import json
import os, glob
from PIL import Image
from utils import transform, transform_seg
import cv2
import torchvision.transforms.functional as FT

class PascalVOCDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, split, keep_difficult=False, percent = 1):
        """
        :param data_folder: folder where data files are stored
        :param split: split, one of 'TRAIN' or 'TEST'
        :param keep_difficult: keep or discard objects that are considered difficult to detect?
        """
        self.split = split.upper()

        assert self.split in {'TRAIN', 'TEST', 'VAL'}

        self.data_folder = data_folder
        self.keep_difficult = keep_difficult

        # Read data files
        with open(os.path.join(data_folder, self.split + '_images.json'), 'r') as j:
            self.images = json.load(j)
        with open(os.path.join(data_folder, self.split + '_objects.json'), 'r') as j:
            self.objects = json.load(j)
        # self.images = list(sorted(self.images))
        # self.objects = list(sorted(self.objects))

        assert len(self.images) == len(self.objects)

        # self.images = self.images[:int(len(self.images) * percent)]
        # self.objects = self.images[:int(len(self.objects) * percent)]


    def __getitem__(self, i):
        # Read image
        image = Image.open(self.images[i], mode='r')
        image = image.convert('RGB')

        # Read objects in this image (bounding boxes, labels, difficulties)
        objects = self.objects[i]
        boxes = torch.FloatTensor(objects['boxes'])  # (n_objects, 4)
        labels = torch.LongTensor(objects['labels'])  # (n_objects)
        difficulties = torch.ByteTensor(objects['difficulties'])  # (n_objects)

        # Discard difficult objects, if desired
        if not self.keep_difficult:
            boxes = boxes[1 - difficulties]
            labels = labels[1 - difficulties]
            difficulties = difficulties[1 - difficulties]

        # Apply transformations
        image, boxes, labels, difficulties = transform(image, boxes, labels, difficulties, split=self.split)

        return image, boxes, labels, difficulties

    def __len__(self):
        return len(self.images)

    def collate_fn(self, batch):
        """
        Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).

        This describes how to combine these tensors of different sizes. We use lists.

        Note: this need not be defined in this Class, can be standalone.

        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
        """

        images = list()
        boxes = list()
        labels = list()
        difficulties = list()

        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])
            difficulties.append(b[3])

        images = torch.stack (images, dim=0)

        return images, boxes, labels, difficulties  # tensor (N, 3, 300, 300), 3 lists of N tensors each


class ADE20KDataset(Dataset) :
    def __init__(self, path, split) :
        self.path = path
        self.split = split
        # self.transforms = transform
        # load all image files, sorting them to
        # ensure that they are aligned
        # self.imgs = list(sorted(os.listdir(os.path.join(path, "imgs"))))
        # self.masks = list(sorted(os.listdir(os.path.join(path, "masks"))))
        
        with open('/home/rita/111/111-2DL/HW4/dataset/ADE20K_DL_course/train.txt', 'r') as f :
            self.train_idx = [line.strip() for line in f.readlines()]
        with open('/home/rita/111/111-2DL/HW4/dataset/ADE20K_DL_course/val.txt', 'r') as f :
            self.val_idx = [line.strip() for line in f.readlines()]
        with open('/home/rita/111/111-2DL/HW4/dataset/ADE20K_DL_course/test.txt', 'r') as f :
            self.test_idx = [line.strip() for line in f.readlines()]
        
        if split == 'train' :
            i = self.train_idx
        elif split == 'val':
            i = self.val_idx
        else : 
            i = self.test_idx
        
        self.imgs = list(sorted(glob.glob('/home/rita/111/111-2DL/HW4/dataset/ADE20K_DL_course/imgs/*jpg')))
        self.masks = list(sorted(glob.glob('/home/rita/111/111-2DL/HW4/dataset/ADE20K_DL_course/masks/*png')))

        self.imgs = [elem for elem in self.imgs if any(item in elem for item in i)]
        self.masks = [elem for elem in self.masks if any(item in elem for item in i)]

        # print(len(self.imgs))
        # print(len(self.masks))
        
    def __getitem__(self, idx) :      
        img = Image.open(self.path + '/imgs/' + self.imgs[idx][-20:]).convert("RGB")
        mask = Image.open(self.path + '/masks/' + self.masks[idx][-24:]).convert("L")
        # img_path = os.path.join(self.path, "imgs/png", self.imgs[idx])
        # mask_path = os.path.join(self.path, "masks", self.masks[idx])
        # img = Image.open(img_path).convert("RGB")
        # img = torch.Tensor(list(img))
        img = FT.resize(img, (300, 300))
        img = FT.to_tensor(img)#.long()

        # mask = Image.open(mask_path).convert('L')
        mask = FT.resize(mask, (300, 300))
        mask = FT.to_tensor(mask)#.long()
        # img, mask = transform_seg(img, mask)
        mask = mask.squeeze()
        return img, mask
        
    def __len__(self):
        return len(self.imgs)

    def collate_fn(self, batch):
        """
        Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).

        This describes how to combine these tensors of different sizes. We use lists.

        Note: this need not be defined in this Class, can be standalone.

        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
        """

        images = list()
        # boxes = list()
        # labels = list()
        # difficulties = list()
        masks = list()

        for b in batch:
            images.append(b[0])
            # boxes.append(b[1])
            # labels.append(b[2])
            # difficulties.append(b[3])
            masks.append(b[1])

        images = torch.stack (images, dim=0)
        masks = torch.stack (masks, dim=0)

        return images, masks
        # return images, boxes, labels, difficulties