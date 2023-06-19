import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.transforms import ToTensor
from PIL import Image

import torch.nn as nn
import torch.optim as optim
# import torch.utils as utils
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models import resnet

from functools import partial
import cv2, glob, os, itertools
from tqdm import tqdm, trange
import multiprocessing
from multiprocessing import Pool

import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np

FIGURE_SIZE = 800

pretrained_model = resnet.resnet50(pretrained=True)


def GetAnnotBoxLoc(imgpath, AnotPath, size):#AnotPath VOC標註文件路徑
    img = cv2.imread(imgpath)# .resize(size, size)
    im_shape = torch.tensor(img.shape[-2:])

    tree = ET.ElementTree(file=AnotPath)  #打開文件，解析成一棵樹型結構
    root = tree.getroot()#獲取樹型結構的根
    ObjectSet=root.findall('object')#找到文件中所有含有object關鍵字的地方，這些地方含有標註目標
    ObjBndBoxSet={} #以目標類別爲關鍵字，目標框爲值組成的字典結構
    for Object in ObjectSet:
        ObjName=Object.find('name').text
        BndBox=Object.find('bndbox')
        x1 = int(BndBox.find('xmin').text) * size / im_shape[0]#-1 #-1是因爲程序是按0作爲起始位置的
        y1 = int(BndBox.find('ymin').text) * size / im_shape[1]#-1
        x2 = int(BndBox.find('xmax').text) * size / im_shape[0]#-1
        y2 = int(BndBox.find('ymax').text) * size / im_shape[1]#-1
        BndBoxLoc=[int(x1),int(y1),int(x2),int(y2)]
        if ObjName in ObjBndBoxSet:
        	ObjBndBoxSet[ObjName].append(BndBoxLoc)#如果字典結構中含有這個類別了，那麼這個目標框要追加到其值的末尾
        else:
        	ObjBndBoxSet[ObjName]=[BndBoxLoc]#如果字典結構中沒有這個類別，那麼這個目標框就直接賦值給其值吧
    img = cv2.resize(img, (size, size))
    # img = torch.Tensor(img)
    return img, ObjBndBoxSet


class VOC2007(torch.utils.data.Dataset) :
    def __init__(self, root, transform) :
        self.root = root
        self.transforms = transform
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "JPEGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "Annotations"))))
        
    def __getitem__(self, idx) :
        img_path = os.path.join(self.root, "JPEGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "Annotations", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        
        img, annotation = GetAnnotBoxLoc(img_path, mask_path, FIGURE_SIZE)
        img = torch.Tensor(img)
        img = torch.permute(img, (2, 0, 1))
        # annotation = GetAnnotBoxLoc(mask_path)
        
        target = {}
        boxes = list(itertools.chain(*list(annotation.values())))
        boxes = torch.Tensor(boxes)
        labels = torch.ones((len(boxes), ), dtype = torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((len(boxes), ), dtype=torch.int64)
        
        # masks
        # mask = Image.open(mask_path)
        # # convert the PIL Image into a numpy array
        # mask = np.array(mask)
        # # instances are encoded as different colors
        # obj_ids = np.unique(mask)
        # # first id is the background, so remove it
        # obj_ids = obj_ids[1:]
        # masks = mask == obj_ids[:, None, None]
        masks = torch.zeros((2, FIGURE_SIZE, FIGURE_SIZE))
        
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        
        
        return img, target
        
    def __len__(self):
        return len(self.imgs)


import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# load a model pre-trained on COCO
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

# replace the classifier with a new one, that has
# num_classes which is user-defined
num_classes = 2  # 1 class (person) + background
# get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features
# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)


import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

# load a pre-trained model for classification and return
# only the features
backbone = torchvision.models.mobilenet_v2(weights="DEFAULT").features
# FasterRCNN needs to know the number of
# output channels in a backbone. For mobilenet_v2, it's 1280
# so we need to add it here
# backbone.out_channels = 1280
backbone.out_channels = 21

# let's make the RPN generate 5 x 3 anchors per spatial
# location, with 5 different sizes and 3 different aspect
# ratios. We have a Tuple[Tuple[int]] because each feature
# map could potentially have different sizes and
# aspect ratios
anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                   aspect_ratios=((0.5, 1.0, 2.0),))

# let's define what are the feature maps that we will
# use to perform the region of interest cropping, as well as
# the size of the crop after rescaling.
# if your backbone returns a Tensor, featmap_names is expected to
# be [0]. More generally, the backbone should return an
# OrderedDict[Tensor], and in featmap_names you can choose which
# feature maps to use.
roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                output_size=7,
                                                sampling_ratio=2)

# put the pieces together inside a FasterRCNN model
model = FasterRCNN(backbone,
                   num_classes=2,
                   rpn_anchor_generator=anchor_generator,
                   box_roi_pool=roi_pooler)




import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model


import torchvision.transforms as T

def get_transform(train):
    transforms = []
    transforms.append(T.PILToTensor())
    transforms.append(T.ConvertImageDtype(torch.float))
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def collate_fn(batch):
    return tuple(zip(*batch))


# Testing forward() method (Optional)
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
# print(model)
dataset = VOC2007('./dataset/VOC2007/VOCdevkit_trainval/VOC2007', get_transform(train=True))
# print(dataset)
# data_loader = torch.utils.data.DataLoader(
#  dataset, batch_size=2, shuffle=True, num_workers=4,
#  collate_fn=utils.collate_fn)
data_loader = DataLoader(
 dataset, batch_size=2, shuffle=True, num_workers=4,
 collate_fn=collate_fn)

# For Training
images,targets = next(iter(data_loader))
# print(images, targets)
images = list(image for image in images)
targets = [{k: v for k, v in t.items()} for t in targets]
output = model(images,targets)   # Returns losses and detections
# For inference
model.eval()
# x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
def to_x(path) :
    x = cv2.imread(path)
    x = cv2.resize(x, (FIGURE_SIZE, FIGURE_SIZE))
    x = torch.Tensor(x).permute(2, 0, 1)
    return x



from detection import utils
from detection import engine
from detection.engine import train_one_epoch, evaluate

def main():
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has two classes only - background and person
    num_classes = 2
    # use our dataset and defined transformations
    # dataset = PennFudanDataset('PennFudanPed', get_transform(train=True))
    # dataset_test = PennFudanDataset('PennFudanPed', get_transform(train=False))
    
    dataset = VOC2007('./dataset/VOC2007/VOCdevkit_trainval/VOC2007', get_transform(train=True))
    dataset_test = VOC2007('./dataset/VOC2007/VOCdevkit/VOC2007_test', get_transform(train=False))


    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    # dataset = torch.utils.data.Subset(dataset, indices[:-50])
    # dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

    # define training and validation data loaders
    data_loader = DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=4,
        collate_fn=collate_fn)

    data_loader_test = DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=collate_fn)

    # get the model using our helper function
    model = get_model_instance_segmentation(num_classes)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    # let's train it for 10 epochs
    num_epochs = 10

    for epoch in trange(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=1)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)

    print("That's it!")
main()
print('Finish')





























