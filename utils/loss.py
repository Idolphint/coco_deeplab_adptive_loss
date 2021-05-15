import torch
import torch.nn as nn
from config.coco_person_config import parser
import torch.nn.functional as F
from torch import Tensor, einsum
from utils import simplex, one_hot, dist_map_transform, class2one_hot

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=0, size_average=True, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.size_average = size_average

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs, targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()

# Variable is a simple wrapper of tensor ,with recording of creator and grad ..

cfg = parser.parse_args()


class MultiLosses(object):
    def __init__(self, weight=None, cuda=True, device=None):
        self.cuda = cuda
        self.device = torch.device(device)
        self.weight = weight
        self.loss = self.L1loss

    def bulid_loss(self, mode='ce'):
        """ choices: [0: L1, 1: L2, 2: exp, 3: ce, 4: focal, 5: dice, 6:smooth01]
        """
        if mode == 'L2' or mode == '0':
            self.loss = self.MSEloss
        elif mode == 'ce' or mode == '1':
            self.loss = self.crossentropy
        elif mode == 'focal' or mode == '2':
            self.loss = self.focal1
            #self.loss = self.focalloss
        elif mode == 'dice' or mode == '3':
            self.loss = self.MultiClassDiceloss
        elif mode == 'Fscore' or mode == '4':
            self.loss = self.softFMeasure
        elif mode == 'Bound' or mode == '5':
            self.loss = self.boundaryLoss
        elif mode == 'mix':
            self.loss = self.mixloss
        elif mode == 'bound_dice':
            self.loss = self.boundDiceLoss
        # elif mode == 'smooth01' or mode == '6':
        #    self.loss = self.smooth01loss
        else:
            raise NotImplementedError

    def toOneHot(self, label, class_num=10):
        ''' input [n, x,x], output [n, c, x, x]
        '''
        n, w, h = label.shape
        flush = torch.zeros(n, class_num, w, h).to(self.device)

        for i in range(n):
            for j in range(class_num):
                flush[i, j, :, :] = torch.eq(label[i, :, :], j)
        one_hot = flush.contiguous()

        return one_hot

    ##是否应该考虑把c领出来？
    def L1loss(self, logit, target):
        target = target['label']
        pred = logit.view(-1)
        one_hot_target = self.toOneHot(target, class_num=cfg.num_classes)  # n, c, w, hkeneng zhkanl
        groundT = one_hot_target.view(-1)
        # ? cuda????
        return torch.mean(torch.abs(pred - groundT))

    def MSEloss(self, logit, target):
        target = target['label']
        pred = logit.view(-1)
        one_hot_target = self.toOneHot(target, class_num=cfg.num_classes)
        groundT = one_hot_target.view(-1)
        return torch.mean(torch.pow(pred - groundT, 2))

    def exploss(self, logit, target):
        target = target['label']
        pred = logit.contiguous().view(-1)
        groundT = target.contiguous().view(-1)

        return torch.mean(torch.exp(-1 * groundT * pred))

    def multiClassExp(self, logit, target):
        target = target['label']
        n, c, w, h = logit.shape
        OH_target = self.toOneHot(target, class_num=cfg.num_classes)
        exp_total = 0
        for i in range(c):
            exp_item = self.exploss(logit[:, i, :, :], OH_target[:, i, :, :])
            exp_total += exp_item
        return exp_total / c

    def crossentropy(self, logit, target):
        target = target['label']
        # n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean', weight=self.weight)
        loss = criterion(logit, target)
        return loss

    def focalloss(self, logit, target):  # only for 2 class!!
        target = target['label']
        alpha = 1.0
        gamma = 2
        # n, c, h, w = logit.size()
        ce_loss = F.cross_entropy(
            logit, target, reduction='none', ignore_index=255)
        pt = torch.exp(-ce_loss)
        focal_loss = alpha * (1 - pt) ** gamma * ce_loss
        return focal_loss.mean()

    def focal1(self, logit, target):  # no use ce
        target = target['label']
        gamma = 2.0
        N, C, W, H = logit.shape
        P = F.softmax(logit, dim=1)

        class_mask = self.toOneHot(target, class_num=cfg.num_classes)

        probs = (P * class_mask).sum(1).view(-1, 1)  # sum.dim=1
        log_p = probs.log()

        batch_loss = -(torch.pow((1 - probs), gamma)) * log_p

        loss = batch_loss.mean() * 5  # all loss mean about pixel, so just *10
        return loss

    def diceloss(self, logit, target):
        ##是否需要将target转为 onehot
        logit = logit.contiguous().view(-1)
        target = target.contiguous().view(-1)

        smooth = 0.0001
        intersection = torch.sum(target * logit)

        dice = (2. * intersection + smooth) / (torch.sum(target * target) +
                                               torch.sum(logit * logit) + smooth)

        return (1. - dice)

    def MultiClassDiceloss(self, logit, target):
        target = target['label']
        n, c, w, h = logit.size()
        OHtarget = self.toOneHot(target, class_num=cfg.num_classes)
        dice_loss = 0
        for i in range(c):
            dice_loss += self.diceloss(logit[:, i, :, :], OHtarget[:, i, :, :])
        dice_loss /= c
        return dice_loss

    def normalData(self, data):
        maxp = torch.max(data)
        minp = torch.min(data)
        nor_data = (data - minp) / (maxp - minp)
        return nor_data

    def softlabel(self, label, epsilon=0.1):
        newlabel = label * (1 - epsilon) + epsilon / cfg.num_classes

        return newlabel

    def softFMeasure(self, logit, target, beta=1.0):
        target = target['label']
        # logit = logit.contiguous().view(-1)
        # target = target.contiguous().view(-1)
        logit_nor = self.normalData(logit)
        OHtarget = self.toOneHot(target, class_num=cfg.num_classes)
        newlabel = self.softlabel(OHtarget)

        zh_target = newlabel[:, 0, :, :]
        zh_logit = logit_nor[:, 0, :, :]
        re_target = newlabel[:, 1, :, :]
        re_logit = logit_nor[:, 1, :, :]

        TP = torch.sum(zh_logit * zh_target)
        FP = torch.sum(zh_logit * re_target)
        FN = torch.sum(re_logit * zh_target)

        FM_b = ((1 + beta * beta) * TP) / ((1 + beta * beta) * TP + beta * beta * FN + FP)
        loss = (1.0 - FM_b) * 50

        return loss

    def boundaryLoss(self, logits, target) -> Tensor:
        idc = [1]
        dist_maps = target['dist_map']
        probs = F.softmax(logits, dim=1)
        #probs = logits
        assert simplex(probs)
        assert not one_hot(dist_maps)

        pc = probs[:, idc, ...].type(torch.float32)
        dc = dist_maps[:, idc, ...].type(torch.float32)

        multipled = einsum("bkwh,bkwh->bkwh", pc, dc) #简记求和，（求和前a下标， 求和b下标 -> 求和后x下标)j即一一求和

        loss = multipled.mean()
        return loss

    def boundDiceLoss(self, logit, target):
        bound = self.boundaryLoss(logit,target)
        dice = self.MultiClassDiceloss(logit, target)
        loss = bound+dice

        return loss

    def mixloss(self, logit, target):
        l2 = self.MSEloss(logit, target)
        dice= self.MultiClassDiceloss(logit, target)
        ce = self.crossentropy(logit, target)
        focal = self.focalloss(logit, target)
        bound = self.boundaryLoss(logit, target)

        mix = l2 + dice + ce + focal + bound
        return mix
    def smooth01loss(self, logit, target):
        pass

    def forward(self):
        return self.loss

    def getBiggestGrad(self):
        # TODO
        pass


if __name__ == "__main__":
    criterion = MultiLosses(device='cuda:0')
    a = torch.rand(2, 2, 192, 192).to(criterion.device)
    b = torch.rand(2, 192, 192).to(criterion.device).long()
    criterion.bulid_loss(mode='ce')
    print(criterion.loss(a, b).item())
    for i in range(7):
        mode_str = str(i)
        criterion.bulid_loss(mode=mode_str)
        print(criterion.loss(a, b).item())



