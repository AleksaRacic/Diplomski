import numpy as np
import torch
import os

from tqdm import trange
from zipfile import BadZipFile

from Models.wren import WildRelationNetworkPairs
from data_loader import PGM_dataset
from utils import get_transforms
from datetime import datetime

BATCH_SIZE = 32
NUM_EPOCHS = 2
IMG_SIZE = 160
WORKERS = 4
LR = 0.0001

VAL_FREQUENCY = 10

train_dataset_path = 'Dataset/neutral/train'
validation_dataset_path = 'Dataset/neutral/val'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if torch.cuda.device_count() > 0:
    print("Available: ", torch.cuda.device_count(), "GPUs")

model = WildRelationNetworkPairs().to(device)
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
loss_fn = torch.nn.CrossEntropyLoss().to(device)

tf = get_transforms()

train_set = PGM_dataset(train_dataset_path, tf)
val_set = PGM_dataset(validation_dataset_path, tf)



train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS, pin_memory=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS, pin_memory=True)

def validation_accuracy():
    model.eval()
    iter_val = iter(val_loader)
    metricsa = {'correct': [], 'count': []}
    for i in range(len(iter_val)):
        image, target = next(iter_val)
        image = torch.autograd.Variable(image, requires_grad=False).to(device)
        target = torch.autograd.Variable(target, requires_grad=False).to(device)

        with torch.no_grad():
            predict = model(image)

        pred = torch.max(predict[:, :], 1)[1]
        correct = pred.eq(target.data).cpu().sum().numpy()

        metricsa['correct'].append(correct)
        metricsa['count'].append(target.size(0))

    acc_val = 100 * np.sum(metricsa['correct']) / np.sum(metricsa['count'])
    model.train()
    return acc_val

def train(save_path_model : str):

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        metrics = {'loss': [], 'correct': [], 'count': []}

        train_log_name = os.path.join(save_path_model, 'train_log.txt')
        with open(train_log_name, 'a') as f:
            f.write('Batch size {:02d}: Num epochs {:02d}: LR {:.8f}: Val frequency {:02d}: Img Size {:03d}: Time {:s}\n'.format(
                BATCH_SIZE, NUM_EPOCHS,  LR, VAL_FREQUENCY, IMG_SIZE, datetime.now().strftime('%D-%H:%M:%S')))

        train_loader_iter = iter(train_loader)

        val_period = len(train_loader_iter) // VAL_FREQUENCY

        for batch_idx in trange(len(train_loader_iter)):
            try:
                image, target = next(train_loader_iter)
            except BadZipFile:
                continue

            image = torch.autograd.Variable(image, requires_grad=True).to(device)

            target = target.to(device)
            predict = model(image)

            loss = loss_fn(predict, target)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            metrics['loss'].append(loss.item())

            pred = torch.max(predict[:, :], 1)[1]
            correct = pred.eq(target.data).cpu().sum().numpy()

            metrics['correct'].append(correct)
            metrics['count'].append(target.size(0))

            if batch_idx > 1 and batch_idx % val_period == 0:
                print('Epoch: {:d}/{:d},  Loss: {:.8f}'.format(epoch, NUM_EPOCHS, np.mean(metrics['loss'])))

                acc_val = validation_accuracy()
                print(' Validation Accuracy: {:.8f} \n'.format(acc_val))

                acc_train= 100 * np.sum(metrics['correct']) / np.sum(metrics['count'])

                time_now = datetime.now().strftime('%D-%H:%M:%S')

                with open(train_log_name, 'a') as f:
                    f.write('Epoch {:02d}: Batch_idx {:d}: Acc_val {:.8f}: Acc_train {:.8f}: Loss {:.8f}: Time {:s}\n'.format(
                        epoch, batch_idx,  acc_val,acc_train, np.mean(metrics['loss']), time_now))

                metrics = {'loss': [], 'correct': [], 'count': []}


        accuracy = 100 * np.sum(metrics['correct']) / np.sum(metrics['count'])

        print('Epoch: {:d}/{:d},  Loss: {:.8f}, Acc: {:.8f}'.format(epoch, NUM_EPOCHS, np.mean(metrics['loss']),
                                                                    accuracy))
        if epoch > 0:
            save_name = os.path.join(save_path_model, 'model_{:02d}.pth'.format(epoch))
            torch.save(model.state_dict(), save_name)

    return metrics

if __name__ == '__main__':
    time_now = datetime.now().strftime('_%H_%d_%m')
    results_folder = 'Results/' + model.__class__.__name__ + time_now
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    train(results_folder)
    