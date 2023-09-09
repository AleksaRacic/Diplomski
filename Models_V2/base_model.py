import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

    def load_model(self, path, epoch):
        self.state_dict = torch.load(path+'{}_epoch_{}.pth'.format(self.__class__.__name__, epoch))

    def save_model(self, path, epoch):
        torch.save(self.state_dict(), path+'{}_epoch_{}.pth'.format(self.__class__.__name__, epoch))
    
    def get_arhitecture(self):
        return str(self)

    def compute_loss(self, output, target, meta_target):
        raise NotImplementedError

    def train_(self, input, target, meta_target):
        self.optimizer.zero_grad()
        output = self(input)
        loss = self.compute_loss(output, target, meta_target)
        loss.backward()
        self.optimizer.step()
        pred = output[0].data.max(1)[1]
        correct = pred.eq(target.data).cpu().sum().numpy()
        return loss.item(), correct, target.size()[0]

    def validate_(self, input, target, meta_target):
        output = self(input)
        loss = self.compute_loss(output, target, meta_target)
        pred = output[0].data.max(1)[1]
        correct = pred.eq(target.data).cpu().sum().numpy()
        accuracy = correct * 100. / target.size()[0]
        return loss.item(), correct, target.size()[0]

    def test_(self, input, target, meta_target):
        output = self(input)
        pred = output[0].data.max(1)[1]
        correct = pred.eq(target.data).cpu().sum().numpy()
        return correct, target.size()[0]