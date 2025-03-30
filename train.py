import numpy as np
import torch
import torch.backends.cudnn as cudnn
import argparse
import os
import time
from sklearn.metrics import confusion_matrix
from dataset import SUM_Dataset
from torch import nn
from models import Model
from utils import AverageTracker
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--batch_size', type=int, default=4096,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=30,
                        help='number of training epochs')
    parser.add_argument('--num_classes', type=int, default=6,
                        help='number of classes')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--lr_decay_freq', type=int, default=10,
                        help='learning rate decay frequency')
    parser.add_argument('--lr_decay_rate', type=float, default=0.5,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.001,
                        help='weight decay')
    parser.add_argument('--optim_momentum', type=float, default=0.98,
                        help='optimizer momentum')
    parser.add_argument('--model_name', type=str, default='model.pt')

    config = parser.parse_args()
    return config
    
def set_config(weight=None):
    """
    Initialize the model and loss function based on the configuration.

    Args:
        weight (np.ndarray, optional): Class weights for the loss function.

    Returns:
        tuple: The model and the loss function.
    """
    model = Model()
    if weight is None:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(weight).float())

    model = model.to(device)
    criterion = criterion.to(device)
    return model, criterion

def set_loader(config, mode):
    """
    Set up a DataLoader for the specified dataset mode.

    Args:
        config (argparse.Namespace): Configuration options.
        mode (str): The mode e.g., 'train' or 'val'.

    Returns:
        torch.utils.data.DataLoader: A configured DataLoader.
    """
    dataset = SUM_Dataset(mode=mode)

    class_count = np.array([np.sum(dataset.labels==i) for i in range(config.num_classes)])
    print('{} class distribution: '.format(mode), class_count)

    if mode=='train':
        weight = 1/np.sqrt(class_count)
        samples_weight = np.array([weight[int(t)] for t in dataset.labels.flatten()])
        samples_weight = torch.from_numpy(samples_weight)
        sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=False, sampler=sampler, num_workers=config.num_workers, pin_memory=True)
    else:
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=True)
    return data_loader

def decreas_lr(args, optimizer, epoch):
    """
    Adjust the learning rate based on the epoch.

    Args:
        args (argparse.Namespace): Configuration options.
        optimizer (torch.optim.Optimizer): The optimizer being used.
        epoch (int): Current epoch number.
    """
    if epoch%args.lr_decay_freq-1==0:
        lr = args.lr * (args.lr_decay_rate ** (epoch // args.lr_decay_freq))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

def train(train_loader, model, criterion, optimizer, epoch, config):
    """
    Train the model for one epoch.

    Args:
        train_loader (torch.utils.data.DataLoader): DataLoader for training data.
        model (torch.nn.Module): The model to train.
        criterion (torch.nn.Module): The loss function.
        optimizer (torch.optim.Optimizer): The optimizer.
        epoch (int): Current epoch number.
        config (argparse.Namespace): Configuration options.
    """
    model.train()

    losses = AverageTracker()
    acc_track = AverageTracker()
    conf_mat = np.zeros((config.num_classes,config.num_classes))
    classes = list(range(config.num_classes))

    for i, (points, labels, area) in enumerate(train_loader):
        bsz = labels.shape[0]
        points, labels = points.float().to(device), labels.long().to(device)
        output = model(points)

        loss = criterion(output, labels)
        losses.add(loss.detach().item(), bsz)
        preds = torch.argmax(output, dim=1)
        conf_mat+=confusion_matrix(labels.cpu().numpy(), preds.cpu().numpy(), labels=classes, sample_weight=area.cpu().numpy())
        acc = torch.sum(preds == labels).item() / bsz
        acc_track.add(acc, bsz)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    miou=0
    miou_abs=0
    oa_area = 0
    for i in range(config.num_classes):
        iou = conf_mat[i,i] / (np.sum(conf_mat[i,:]) + np.sum(conf_mat[:,i]) - conf_mat[i,i])
        miou+=iou
        oa_area+=conf_mat[i,i]
        print('Class:', i, 'IoU: {:.2f}%'.format(100.*iou))

    print('Loss: {:.4f}'.format(losses.average), 'Accuracy: {:.2f}%'.format(100.*oa_area/np.sum(conf_mat)), 'mIoU: {:.2f}%'.format(100.*miou/config.num_classes))

def validate(val_loader, model, criterion, epoch, config):
    """
    Validate the model on a validation set.

    Args:
        val_loader (torch.utils.data.DataLoader): DataLoader for validation data.
        model (torch.nn.Module): The model.
        criterion (torch.nn.Module): The loss function.
        epoch (int): Current epoch number.
        config (argparse.Namespace): Configuration options.

    Returns:
        float: Mean Intersection over Union (mIoU) for the validation set.
    """
    model.eval()

    losses = AverageTracker()
    acc_track = AverageTracker()
    conf_mat = np.zeros((config.num_classes,config.num_classes))
    classes = list(range(config.num_classes))

    with torch.no_grad():
        for i, (points, labels, area) in enumerate(val_loader):
            bsz = labels.shape[0]
            points, labels = points.float().to(device), labels.long().to(device)
            output = model(points)
            loss = criterion(output, labels)
            losses.add(loss.item(), bsz)
            preds = torch.argmax(output, dim=1)
            conf_mat+=confusion_matrix(labels.cpu().numpy(), preds.cpu().numpy(), labels=classes, sample_weight=area.cpu().numpy())
            acc = torch.sum(preds == labels).item() / bsz
            acc_track.add(acc, bsz)

    miou=0
    mf1=0
    oa_area = 0
    for i in range(config.num_classes):
        iou = conf_mat[i,i] / (np.sum(conf_mat[i,:]) + np.sum(conf_mat[:,i]) - conf_mat[i,i])
        miou+=iou
        print('Class:', i, 'IoU: {:.2f}%'.format(100.*iou))
        oa_area+=conf_mat[i,i]
        f1 = 2*conf_mat[i,i] / (np.sum(conf_mat[i,:]) + np.sum(conf_mat[:,i]))
        mf1+=f1

    print('Loss: {:.4f}'.format(losses.average), 'Accuracy: {:.2f}%'.format(100.*oa_area/np.sum(conf_mat)) , 'mIoU: {:.2f}%'.format(100.*miou/config.num_classes), 'mF1 {:.2f}%'.format(100.*mf1/config.num_classes))

    return miou

def main():
    global device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    config = parse_option()
    model_path = BASE_DIR / 'saved_models'
    
    train_loader = set_loader(config, mode='train')
    val_loader = set_loader(config, mode='val')

    model, criterion = set_config(config)
    optimizer = torch.optim.SGD(model.parameters(),
                          lr=config.lr,
                          momentum=config.optim_momentum,
                          weight_decay=config.weight_decay)

    best_val=0.

    for epoch in range(1, config.epochs + 1):
        print('Epoche', epoch)
        decreas_lr(config, optimizer, epoch)

        start = time.time()
        train(train_loader, model, criterion, optimizer, epoch, config)
        print('Train time: {:.2f}s'.format(time.time()-start))

        if epoch>10:
            print('VAL Epoche', epoch)
            start = time.time()
            mIoU_val = validate(val_loader, model, criterion, epoch, config)
            time2 = time.time()
            print('Validation time: {:.2f}s'.format(time.time()-start))

            if best_val<mIoU_val:
                best_val = mIoU_val
                torch.save(model, model_path / config.model_name)

if __name__=='__main__':
    main()




