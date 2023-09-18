import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

    def load_model(self, path, epoch):
        checkpoint = torch.load(path+'/{}_epoch_{}.pth'.format(self.__class__.__name__, epoch))
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def save_model(self, path, epoch):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }
        torch.save(self.state_dict(), path+'/{}_epoch_{}.pth'.format(self.__class__.__name__, epoch))
    
    def get_arhitecture(self):
        return str(self)

    def compute_loss(self, pred, target):
        loss = F.cross_entropy(pred, target)
        return loss

    def train_(self, input, target):
        self.optimizer.zero_grad()
        output = self(input)
        loss = self.compute_loss(output, target)
        loss.backward()
        self.optimizer.step()
        pred = output.data.max(1)[1]
        correct = pred.eq(target.data).cpu().sum().numpy()
        return loss.item(), correct, target.size()[0]

    def validate_(self, input, target):
        output = self(input)
        loss = self.compute_loss(output, target,)
        pred = output.data.max(1)[1]
        correct = pred.eq(target.data).cpu().sum().numpy()
        return loss.item(), correct, target.size()[0]

    def test_(self, input, target):
        output = self(input)
        pred = output.data.max(1)[1]
        correct = pred.eq(target.data).cpu().sum().numpy()
        return correct, target.size()[0]