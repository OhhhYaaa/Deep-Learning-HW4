import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
from model import SSD300, MultiBoxLoss, final_model, final
from datasets import PascalVOCDataset, ADE20KDataset
from utils import *
from eval import evaluate
import matplotlib.pyplot as plt

# Data parameters
data_folder = './'  # folder with data files
keep_difficult = True  # use objects considered difficult to detect?

# Model parameters
# Not too many here since the SSD300 has a very specific structure
n_classes = len(label_map)  # number of different types of objects
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Learning parameters
checkpoint = None  # path to model checkpoint, None if none
batch_size = 1  # batch size
iterations = 120000  # number of iterations to train
workers = 4  # number of workers for loading data in the DataLoader
print_freq = 200  # print training status every __ batches
lr = 1e-3  # learning rate
decay_lr_at = [80000, 100000]  # decay learning rate after these many iterations
decay_lr_to = 0.1  # decay learning rate to this fraction of the existing learning rate
momentum = 0.9  # momentum
weight_decay = 5e-4  # weight decay
grad_clip = None  # clip if gradients are exploding, which may happen at larger batch sizes (sometimes at 32) - you will recognize it by a sorting error in the MuliBox loss calculation

cudnn.benchmark = True


def main():
    """
    Training.
    """
    global start_epoch, label_map, epoch, checkpoint, decay_lr_at

    # Initialize model or load checkpoint
    if checkpoint is None:
        start_epoch = 0
        # model = SSD300(n_classes=n_classes)
        model = final(20, 150)
        # model = final_model(20, 150)
        # Initialize the optimizer, with twice the default learning rate for biases, as in the original Caffe repo
        biases = list()
        not_biases = list()
        for param_name, param in model.named_parameters():
            if param.requires_grad:
                if param_name.endswith('.bias'):
                    biases.append(param)
                else:
                    not_biases.append(param)
        optimizer = torch.optim.SGD(params=[{'params': biases, 'lr': 2 * lr}, {'params': not_biases}],
                                    lr=lr, momentum=momentum, weight_decay=weight_decay)

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']

    # Move to default device
    model = model.to(device)
    
    criterion_ob = MultiBoxLoss(priors_cxcy=model.priors_cxcy).to(device)
    criterion_seg = nn.CrossEntropyLoss()

    # Custom dataloaders
    train_dataset_ob = PascalVOCDataset(data_folder,
                                     split='train',
                                     keep_difficult=keep_difficult)
    train_loader_ob = torch.utils.data.DataLoader(train_dataset_ob, batch_size=128, shuffle=True,
                                               collate_fn=train_dataset_ob.collate_fn, num_workers=workers,
                                               pin_memory=True)
    
    train_dataset_seg = ADE20KDataset('/home/rita/111/111-2DL/HW4/dataset/ADE20K_DL_course', 'train')
    train_loader_seg = torch.utils.data.DataLoader(train_dataset_seg, batch_size=35, shuffle=True,
                                               collate_fn=train_dataset_seg.collate_fn, num_workers=workers,
                                               pin_memory=True)  # note that we're passing the collate function here

    # Calculate total number of epochs to train and the epochs to decay learning rate at (i.e. convert iterations to epochs)
    # To convert iterations to epochs, divide iterations by the number of iterations per epoch
    # The paper trains for 120,000 iterations with a batch size of 32, decays after 80,000 and 100,000 iterations
    # epochs = iterations // (len(train_dataset) // 32)
    epochs = iterations // (len(train_dataset_ob) // 32) // 512
    
    decay_lr_at = [it // (len(train_dataset_ob) // 32) for it in decay_lr_at]
    print(epochs)
    # Epochs
    for epoch in range(start_epoch, epochs):

        # Decay learning rate at particular epochs
        if epoch in decay_lr_at:
            adjust_learning_rate(optimizer, decay_lr_to)

        # One epoch's training
        train(train_loader=train_loader_ob,
              model=model,
              criterion=criterion_ob,
              optimizer=optimizer,
              epoch=epoch, 
              task = 'ob'
              )

        train(train_loader=train_loader_seg,
              model=model,
              criterion=criterion_seg,
              optimizer=optimizer,
              epoch=epoch, 
              task = 'seg')
        
        
        # Save checkpoint
        save_checkpoint(epoch, model, optimizer)


def train(train_loader, model, criterion, optimizer, epoch, task):
    """
    One epoch's training.

    :param train_loader: DataLoader for training data
    :param model: model
    :param criterion: MultiBox loss
    :param optimizer: optimizer
    :param epoch: epoch number
    """
    model.train()  # training mode enables dropout

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss

    start = time.time()

    # Batches
    if task == 'ob' :
        for i, (images, boxes, labels, _) in enumerate(train_loader):
            data_time.update(time.time() - start)

            # Move to default device
            images = images.to(device)  # (batch_size (N), 3, 300, 300)
            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]

            # Forward prop.
            predicted_locs, predicted_scores, seg = model(images)  # (N, 8732, 4), (N, 8732, n_classes)

            # Loss
            loss = criterion(predicted_locs, predicted_scores, boxes, labels)  # scalar

            # Backward prop.
            optimizer.zero_grad()
            loss.backward()

            # Clip gradients, if necessary
            if grad_clip is not None:
                clip_gradient(optimizer, grad_clip)

            # Update model
            optimizer.step()

            losses.update(loss.item(), images.size(0))
            batch_time.update(time.time() - start)

            start = time.time()
            if i % print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                    'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, i, len(train_loader),
                                                                    batch_time=batch_time,
                                                                    data_time=data_time, loss=losses))
            
    else :
        for i, (images, masks) in enumerate(train_loader):

            data_time.update(time.time() - start)

            # Move to default device
            images = images.to(device)  # (batch_size (N), 3, 300, 300)
            masks = masks.to(device)

            # Forward prop.
            predicted_locs, predicted_scores, seg = model(images)  # (N, 8732, 4), (N, 8732, n_classes)

            # Loss
            pre = F.interpolate(seg, (300,300), mode='bilinear', align_corners=False)
            loss = criterion(pre, masks.long())  # scalar

            # Backward prop.
            optimizer.zero_grad()
            loss.backward()

            # Clip gradients, if necessary
            if grad_clip is not None:
                clip_gradient(optimizer, grad_clip)

            # Update model
            optimizer.step()

            losses.update(loss.item(), images.size(0))
            batch_time.update(time.time() - start)

            start = time.time()

            # Print status
            if i % print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                    'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, i, len(train_loader),
                                                                    batch_time=batch_time,
                                                                    data_time=data_time, loss=losses))
    del predicted_locs, predicted_scores, images# , boxes, labels  # free some memory since their histories may be stored



def train_together(model, dataloader_ob, dataloader_seg, val_loader_ob, val_loader_seg, loss_ob, loss_seg):
    """
    Training.
    """
    ls_loss_ob_train = []
    ls_loss_seg_train = []
    ls_loss_ob_val = []
    ls_loss_seg_val = []
    global start_epoch, label_map, epoch, checkpoint, decay_lr_at
    
    # Move to default device
    model = model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, weight_decay = 1e-5)
    epochs = iterations // (len(train_dataset_ob) // 32) // 128
    
    decay_lr_at = [it // (len(train_dataset_ob) // 32) for it in decay_lr_at]
    
    print(epochs)
    # Epochs
    for epoch in range(epochs):
    # for epoch in range(start_epoch, epochs):

        # Decay learning rate at particular epochs
        if epoch in decay_lr_at:
            adjust_learning_rate(optimizer, decay_lr_to)

        model.train()  # training mode enables dropout

        batch_time = AverageMeter()  # forward prop. + back prop. time
        data_time = AverageMeter()  # data loading time
        losses = AverageMeter()  # loss

        start = time.time()
        # One epoch's training
        train_loss_ob = []
        train_loss_seg = []
        l = min(len(dataloader_ob), len(dataloader_seg))

        for i, ((images, boxes, labels, _), (images_seg, masks_seg)) in enumerate(zip(dataloader_ob, dataloader_seg)) :
            # ob #########################
            data_time.update(time.time() - start)
            
            # Move to default device
            images = images.to(device)  # (batch_size (N), 3, 300, 300)
            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]
            
            # Forward prop.
            predicted_locs, predicted_scores, seg = model(images)  # (N, 8732, 4), (N, 8732, n_classes)

            # Loss
            loss = loss_ob(predicted_locs, predicted_scores, boxes, labels)  # scalar
            train_loss_ob.append(loss)
            # Backward prop.
            optimizer.zero_grad()
            loss.backward()

            # Clip gradients, if necessary
            if grad_clip is not None:
                clip_gradient(optimizer, grad_clip)

            # Update model
            optimizer.step()

            losses.update(loss.item(), images.size(0))
            batch_time.update(time.time() - start)

            start = time.time()

            # seg #########################
            data_time.update(time.time() - start)

            # Move to default device
            images_seg = images_seg.to(device)  # (batch_size (N), 3, 300, 300)
            masks_seg = masks_seg.to(device)

            # Forward prop.
            predicted_locs, predicted_scores, seg = model(images_seg)  # (N, 8732, 4), (N, 8732, n_classes)

            # Loss
            pre = F.interpolate(seg, (300,300), mode='bilinear', align_corners=False)
            loss2 = loss_seg(pre, masks_seg.long())  # scalar
            train_loss_seg.append(loss2)
            # Backward prop.
            optimizer.zero_grad()
            loss2.backward()

            # Clip gradients, if necessary
            if grad_clip is not None:
                clip_gradient(optimizer, grad_clip)

            # Update model
            optimizer.step()

            losses.update(loss.item(), images_seg.size(0))
            batch_time.update(time.time() - start)

            start = time.time()
        
            if i % print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                    'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Loss_seg {loss2}\t'.format(epoch, i, len(dataloader_ob),
                                                                    batch_time=batch_time,
                                                                    data_time=data_time, loss=losses, loss2 = loss2))
        
        train_ob = sum(train_loss_ob) / len(train_loss_ob)
        ls_loss_ob_train.append(train_ob)
        train_seg = sum(train_loss_seg) / len(train_loss_seg)
        ls_loss_seg_train.append(train_seg)
        
        val_loss_ob = []
        val_loss_seg = []
        for i, (images, boxes, labels, _) in enumerate(val_loader_ob):
            # ob #########################
            
            # Move to default device
            images = images.to(device)  # (batch_size (N), 3, 300, 300)
            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]
            
            # Forward prop.
            predicted_locs, predicted_scores, seg = model(images)  # (N, 8732, 4), (N, 8732, n_classes)

            # Loss
            loss_val = loss_ob(predicted_locs, predicted_scores, boxes, labels)  # scalar
            val_loss_ob.append(loss_val)

        val_ob = sum(val_loss_ob) / len(val_loss_ob)
        ls_loss_ob_val.append(val_ob)
        
        for i, (images, masks) in enumerate(val_loader_seg):
            # seg #########################


            # Move to default device
            images_seg = images_seg.to(device)  # (batch_size (N), 3, 300, 300)
            masks_seg = masks_seg.to(device)

            # Forward prop.
            predicted_locs, predicted_scores, seg = model(images_seg)  # (N, 8732, 4), (N, 8732, n_classes)

            # Loss
            pre = F.interpolate(seg, (300,300), mode='bilinear', align_corners=False)
            loss2 = loss_seg(pre, masks_seg.long())  # scalar
            val_loss_seg.append(loss2)
        
        val_seg = sum(val_loss_seg) / len(val_loss_seg)
        ls_loss_seg_val.append(val_seg)
        
        return ls_loss_ob_train, ls_loss_seg_train, ls_loss_ob_val, ls_loss_seg_val
        # Save checkpoint
        # save_checkpoint(epoch, model, optimizer)

def self_evaluate(model, ob_loader, seg_loader, loss_ob, loss_seg) :
    ls_loss_ob = []
    ls_loss_seg = []
    val_loss_ob = []
    val_loss_seg = []
    
    for i, (images, boxes, labels, _) in enumerate(ob_loader):
        # ob #########################
        
        # Move to default device
        images = images.to(device)  # (batch_size (N), 3, 300, 300)
        boxes = [b.to(device) for b in boxes]
        labels = [l.to(device) for l in labels]
        
        # Forward prop.
        predicted_locs, predicted_scores, seg = model(images)  # (N, 8732, 4), (N, 8732, n_classes)

        # Loss
        loss_val = loss_ob(predicted_locs, predicted_scores, boxes, labels)  # scalar
        val_loss_ob.append(loss_val)

    val_ob = sum(val_loss_ob) / len(val_loss_ob)
    ls_loss_ob.append(val_ob)
    
    for i, (images, masks) in enumerate(seg_loader):
        # seg #########################


        # Move to default device
        images_seg = images_seg.to(device)  # (batch_size (N), 3, 300, 300)
        masks_seg = masks_seg.to(device)

        # Forward prop.
        predicted_locs, predicted_scores, seg = model(images_seg)  # (N, 8732, 4), (N, 8732, n_classes)

        # Loss
        pre = F.interpolate(seg, (300,300), mode='bilinear', align_corners=False)
        loss2 = loss_seg(pre, masks_seg.long())  # scalar
        val_loss_seg.append(loss2)
        
    val_seg = sum(val_loss_seg) / len(val_loss_seg)
    ls_loss_seg.append(val_seg)
    print('ob loss :', ls_loss_ob, 'seg loss : ', ls_loss_seg)
    return ls_loss_ob, ls_loss_seg


if __name__ == '__main__':
    # main()
    model = final(21, 151)
    # model = final(20, 150)
    criterion_ob = MultiBoxLoss(priors_cxcy=model.priors_cxcy).to(device)
    criterion_seg = nn.CrossEntropyLoss()
    
    # train data
    train_dataset_ob = PascalVOCDataset(data_folder,
                                        split='train',
                                        keep_difficult=keep_difficult, 
                                        percent = 0.5)
    train_loader_ob = torch.utils.data.DataLoader(train_dataset_ob, batch_size=batch_size, shuffle=True,
                                                collate_fn=train_dataset_ob.collate_fn, num_workers=workers,
                                                pin_memory=True)
        
    train_dataset_seg = ADE20KDataset('/home/rita/111/111-2DL/HW4/dataset/ADE20K_DL_course', 'train')
    train_loader_seg = torch.utils.data.DataLoader(train_dataset_seg, batch_size=batch_size, shuffle=True,
                                                collate_fn=train_dataset_seg.collate_fn, num_workers=workers,
                                                pin_memory=True)
    
    # val data
    val_dataset_ob = PascalVOCDataset(data_folder,
                                        split='val',
                                        keep_difficult=keep_difficult)
    val_loader_ob = torch.utils.data.DataLoader(train_dataset_ob, batch_size=batch_size, shuffle=True,
                                                collate_fn=val_dataset_ob.collate_fn, num_workers=workers,
                                                pin_memory=True)
        
    val_dataset_seg = ADE20KDataset('/home/rita/111/111-2DL/HW4/dataset/ADE20K_DL_course', 'val')
    val_loader_seg = torch.utils.data.DataLoader(train_dataset_seg, batch_size=batch_size, shuffle=True,
                                                collate_fn=val_dataset_seg.collate_fn, num_workers=workers,
                                                pin_memory=True)
    
    ls_loss_ob_train, ls_loss_seg_train, ls_loss_ob_val, ls_loss_seg_val = train_together(model, train_loader_ob, train_loader_seg, val_loader_ob, val_loader_seg, criterion_ob, criterion_seg)
    
    # plot
    n = len(ls_loss_ob_train)
    plt.title('Train_Val_Loss')
    plt.plot(range(n), ls_loss_ob_train, label="Train object detection")
    plt.plot(range(n), ls_loss_seg_train, label="Train semantic segmentation", c = 'red')
    plt.plot(range(n), ls_loss_ob_val, label="Val object detection", c = 'green')
    plt.plot(range(n), ls_loss_seg_val, label="Val semantic segmentation", c = 'orange')
    plt.legend()
    plt.savefig('./figure/train_val_loss.png')
    plt.show()

    
    # Load test data
    test_dataset_ob = PascalVOCDataset(data_folder,
                                    split='test',
                                    keep_difficult=keep_difficult)
    test_loader_ob = torch.utils.data.DataLoader(test_dataset_ob, batch_size=batch_size, shuffle=False,
                                            collate_fn=test_dataset_ob.collate_fn, num_workers=workers, pin_memory=True)
    
    test_dataset_seg = ADE20KDataset('/home/rita/111/111-2DL/HW4/dataset/ADE20K_DL_course', 'test')
    test_loader_seg = torch.utils.data.DataLoader(train_dataset_seg, batch_size=batch_size, shuffle=True,
                                                collate_fn=test_dataset_seg.collate_fn, num_workers=workers,
                                                pin_memory=True)
    
    
    self_evaluate(model, test_loader_ob, test_loader_seg, criterion_ob, criterion_seg)
