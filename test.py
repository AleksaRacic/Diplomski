import json
import Models_V2

from datetime import datetime

from torchvision.transforms import Compose
from transforms import ResizeImage, ToTensor

from data_loader import PGM_dataset
from utils import get_transforms
import torch
import numpy as np
from tqdm import trange

BATCH_SIZE = 32
WORKERS = 16

LR = 0

MODEL_ROOT_FOLDER = 'Results/CNN_MLP_19_09_09'
MODEL_NAME = 'CNN_MLP_19_09_09CNN_MLP_epoch_16.pth'
MODEL_PATH = MODEL_ROOT_FOLDER+'/'+MODEL_NAME
TEST_SAVE_NAME = 'test_acc.txt'
TEST_METRICS_NAME = 'test_metrics.json'

TEST_DATAST_PATH = 'Dataset/neutral/test'
TEST_ACC_PATH = MODEL_ROOT_FOLDER+'/'+TEST_SAVE_NAME
TEST_METRICS_PATH = MODEL_ROOT_FOLDER+'/'+TEST_METRICS_NAME

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = model = Models_V2.CNN_MLP(LR, 0.9, 0.999, 1e-08).to(device)

tf = Compose([ResizeImage(80), ToTensor()])
test_set = PGM_dataset(TEST_DATAST_PATH, tf)

test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKERS, pin_memory=True)

def test():
    model.eval()
    metrics = {'correct': [], 'count': []}

    test_loader_iter = iter(test_loader)
    for idx in trange(len(test_loader_iter)):
        image, target, meta_target = next(test_loader_iter)

        image = torch.autograd.Variable(image, requires_grad=False).to(device)
        target = torch.autograd.Variable(target, requires_grad=False).to(device)
        meta_target = torch.autograd.Variable(meta_target, requires_grad=False).to(device)

        correct, count = model.test_(image, target, meta_target)

        metrics['correct'].append(correct)
        metrics['count'].append(count)

    return metrics

if __name__ == '__main__':
    model.load_state_dict(torch.load(MODEL_PATH))

    metrics_test = test()

    acc_test = 100 * np.sum(metrics_test['correct']) / np.sum(metrics_test['count'])

    time_now = datetime.now().strftime('%H:%M:%S')

    metrics_test['correct'] = [x.tolist() for x in metrics_test['correct']]

    with open(TEST_METRICS_PATH, 'w') as f:
        json.dump(metrics_test, f)

    with open(TEST_ACC_PATH, 'a') as f:
        f.write('Acc Test: {:.8f}, Time: {:s}\n'.format(
            acc_test, time_now))