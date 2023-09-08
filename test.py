from datetime import datetime
import json
from Models.wren import WildRelationNetworkPairs
from data_loader import PGM_dataset
from utils import get_transforms
import torch
import numpy as np
from tqdm import trange

BATCH_SIZE = 16
WORKERS = 4

MODEL_ROOT_FOLDER = 'Results/WildRelationNetworkPairs_23_06_09'
MODEL_NAME = 'model_10.pth'
MODEL_PATH = MODEL_ROOT_FOLDER+'/'+MODEL_NAME
TEST_SAVE_NAME = 'test_acc.txt'
TEST_METRICS_NAME = 'test_metrics.json'

TEST_DATAST_PATH = 'Dataset/neutral/test'
TEST_ACC_PATH = MODEL_ROOT_FOLDER+'/'+TEST_SAVE_NAME
TEST_METRICS_PATH = MODEL_ROOT_FOLDER+'/'+TEST_METRICS_NAME

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = WildRelationNetworkPairs().to(device)

tf = get_transforms()
test_set = PGM_dataset(TEST_DATAST_PATH, tf)

test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKERS, pin_memory=True)

def test():
    model.eval()
    metrics = {'correct': [], 'count': []}

    test_loader_iter = iter(test_loader)
    for idx in trange(len(test_loader_iter)):
        image, target = next(test_loader_iter)

        image = torch.autograd.Variable(image, requires_grad=False).to(device)
        target = torch.autograd.Variable(target, requires_grad=False).to(device)

        with torch.no_grad():
            predict = model(image)

        pred = torch.max(predict[:, :], 1)[1]
        correct = pred.eq(target.data).cpu().sum().numpy()

        metrics['correct'].append(correct)
        metrics['count'].append(target.size(0))

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