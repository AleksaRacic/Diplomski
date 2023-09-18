import numpy as np
import torch
import os
import Models_V2

from tqdm import trange
from zipfile import BadZipFile

from Models.wren import WildRelationNetworkPairs
from data_loader import PGM_dataset
from utils import get_transforms
from datetime import datetime

MODEL_ROOT_FOLDER = 'Results/WildRelationalNetwork_15_16_09'

START_EPOCH = 17
BATCH_SIZE = 32
NUM_EPOCHS = 24
IMG_SIZE = 160
WORKERS = 16 #Max on given server
LR = 0.0001

VAL_FREQUENCY = 10

train_dataset_path = 'Dataset/neutral/train'
validation_dataset_path = 'Dataset/neutral/val'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if torch.cuda.device_count() > 0:
    print("Available: ", torch.cuda.device_count(), "GPUs")

'''
    Change the model type here
    
'''
model = Models_V2.WildRelationalNetwork(LR, 0.9, 0.999, 1e-08).to(device)

tf = get_transforms()

train_set = PGM_dataset(train_dataset_path, tf)
val_set = PGM_dataset(validation_dataset_path, tf)



train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS, pin_memory=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS, pin_memory=True)

history = {'val_acc': []}

def validation_accuracy():
    model.eval()
    iter_val = iter(val_loader)
    metricsa = {'correct': [], 'count': []}
    for i in trange(len(iter_val)):
        image, target, meta_target = next(iter_val)
        image = torch.autograd.Variable(image, requires_grad=False).to(device)
        target = torch.autograd.Variable(target, requires_grad=False).to(device)
        meta_target = torch.autograd.Variable(meta_target, requires_grad=False).to(device)

        loss, correct, count = model.validate_(image, target, meta_target)

        metricsa['correct'].append(correct)
        metricsa['count'].append(target.size(0))

    acc_val = 100 * np.sum(metricsa['correct']) / np.sum(metricsa['count'])
    model.train()
    return acc_val

def train(epoch, save_path_model : str):

    metrics = {'loss': [], 'correct': [], 'count': []}
    train_loader_iter = iter(train_loader)
    val_period = len(train_loader_iter) // VAL_FREQUENCY

    for batch_idx in trange(len(train_loader_iter)):
        try:
            image, target, meta_target = next(train_loader_iter)
        except BadZipFile:
            continue

        image = torch.autograd.Variable(image, requires_grad=True).to(device)
        target = target.to(device)
        meta_target = meta_target.to(device)

        loss, correct, count = model.train_(image, target, meta_target)

        metrics['loss'].append(loss)
        metrics['correct'].append(correct)
        metrics['count'].append(count)

        if (batch_idx > 1 and batch_idx % val_period == 0) or batch_idx == len(train_loader_iter) - 1:
            print('Epoch: {:d}/{:d},  Loss: {:.8f}'.format(epoch, NUM_EPOCHS, np.mean(metrics['loss'])))

            acc_val = validation_accuracy()
            
            if batch_idx == len(train_loader_iter) - 1:
                history['val_acc'].append(acc_val)

            print(' Validation Accuracy: {:.8f} \n'.format(acc_val))

            acc_train= 100 * np.sum(metrics['correct']) / np.sum(metrics['count'])

            time_now = datetime.now().strftime('%D-%H:%M:%S')

            with open(train_log_name, 'a') as f:
                f.write('Epoch {:02d}: Batch_idx {:d}: Acc_val {:.8f}: Acc_train {:.8f}: Loss {:.8f}: Time {:s}\n'.format(
                    epoch, batch_idx,  acc_val, acc_train, np.mean(metrics['loss']), time_now))

            metrics = {'loss': [], 'correct': [], 'count': []}


    accuracy = 100 * np.sum(metrics['correct']) / np.sum(metrics['count'])

    print('Epoch: {:d}/{:d},  Loss: {:.8f}, Acc: {:.8f}'.format(epoch, NUM_EPOCHS, np.mean(metrics['loss']),
                                                              accuracy))

    model.save_model(save_path_model, epoch)
    if epoch > 5:
        if history['val_acc'][-1] <= history['val_acc'][-2] and history['val_acc'][-2] <= history['val_acc'][-3]:
            print('Early stopping')
            with open(train_log_name, 'a') as f:
                f.write('Early stopping\n')
            exit(0)
        

    return metrics

if __name__ == '__main__':
    results_folder = MODEL_ROOT_FOLDER
    train_log_name = os.path.join(results_folder, 'train_log.txt')
    model.load_model(MODEL_ROOT_FOLDER, START_EPOCH-1)
    for epoch in range(START_EPOCH, NUM_EPOCHS + 1):
        train(epoch, results_folder)
    