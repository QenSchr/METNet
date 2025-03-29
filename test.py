import numpy as np
import torch
from torch import nn
import argparse
from dataset import SUM_Dataset
from models import Model
import os
import time
from sklearn.metrics import confusion_matrix
from utils import AverageTracker
from pathlib import Path

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--batch_size', type=int, default=4096,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--num_classes', type=int, default=6,
                        help='number of classes')
    parser.add_argument('--model_name', type=str, default='model.pt')

    config = parser.parse_args()
    return config

def set_config(config, model_path, weight=None):
    """
    Load the model and configure the loss function.

    Args:
        config (argparse.Namespace): Configuration options.
        model_path (str): Path to the saved model file.
        weight (np.ndarray, optional): Class weights for the loss function.

    Returns:
        tuple: Loaded model and configured loss function.
    """
    model = torch.load(model_path / config.model_name)
	
    if weight is None:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(weight).float())

    model = model.to(device)
    criterion = criterion.to(device)
    return model, criterion

def set_loader(config, mode):
    """
    Set up the data loader for a given mode.

    Args:
        config (argparse.Namespace): Configuration options.
        mode (str): Mode of the dataset.

    Returns:
        Data loader for the specified mode.
    """
    dataset = SUM_Dataset(mode=mode)

    class_count = np.array([np.sum(dataset.labels==i) for i in range(config.num_classes)])
    print('{} class distribution: '.format(mode), class_count)

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=True)
    return data_loader

def test(val_loader, model, criterion, config):
    """
    Evaluate the model on the test set.

    Args:
        val_loader (torch.utils.data.DataLoader): Data loader for test data.
        model (torch.nn.Module): The trained model to evaluate.
        criterion (torch.nn.Module): Loss function.
        config (argparse.Namespace): Configuration options.
    """
    model.eval()

    losses = AverageTracker()
    top1 = AverageTracker()
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
            top1.add(acc, bsz)

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

def main():
    global device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    config = parse_option()
    BASE_DIR = Path(__file__).resolve().parent
    model_path = BASE_DIR / 'saved_models'
    
    test_loader = set_loader(config, mode='test')

    model, criterion = set_config(config, model_path)
    start = time.time()
    test(test_loader, model, criterion, config)
    print('Test time: {:.2f}s'.format(time.time()-start))

if __name__=='__main__':
    main()




