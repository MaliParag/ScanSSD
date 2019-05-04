import torch

import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


class FocalLoss(nn.Module):

    def __init__(self, focusing_param=2, balance_param=0.25): #TODO try changing balance_param
        super(FocalLoss, self).__init__()

        self.focusing_param = focusing_param
        self.balance_param = balance_param

    def compute(self, output, target):

        logpt = - F.cross_entropy(output, target, reduction='sum')
        pt = torch.exp(logpt)

        focal_loss = -((1 - pt) ** self.focusing_param) * logpt

        balanced_focal_loss = self.balance_param * focal_loss

        return balanced_focal_loss


def test_focal_loss():
    loss = FocalLoss()

    input = Variable(torch.randn(3, 5), requires_grad=True)
    target = Variable(torch.LongTensor(3).random_(5))

    print(input)
    print(target)

    output = loss(input, target)
    print(output)
    output.backward()

if __name__=='__main__':
    test_focal_loss()
