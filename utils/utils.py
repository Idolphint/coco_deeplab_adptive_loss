import numpy as np
import os
from functools import partial
from torch import Tensor
from torchvision.transforms.functional import normalize
from torchvision import transforms
from collections import OrderedDict
import torch.nn as nn
import torch
from operator import itemgetter
from typing import cast, Iterable, Set, Tuple, Callable, Union
from scipy.ndimage import distance_transform_edt as eucl_distance

D = Union[np.ndarray, Tensor]


def state_transform(old_state):
    new_state_dict = OrderedDict()
    for key, value in old_state.items():
        while True:
            head = key[0:7]
            if head == "module.":
                key = key[7:]
            else:
                break
        new_state_dict[key] = value
    return new_state_dict


def denormalize(tensor, mean, std):
    mean = np.array(mean)
    std = np.array(std)

    _mean = -mean/std
    _std = 1/std
    return normalize(tensor, _mean, _std)


class Denormalize(object):
    def __init__(self, mean, std):
        mean = np.array(mean)
        std = np.array(std)
        self._mean = -mean/std
        self._std = 1/std

    def __call__(self, tensor):
        if isinstance(tensor, np.ndarray):
            return (tensor - self._mean.reshape(-1,1,1)) / self._std.reshape(-1,1,1)
        return normalize(tensor, self._mean, self._std)


def set_bn_momentum(model, momentum=0.1):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = momentum


def fix_bn(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()


def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def simplex(t: Tensor, axis=1) -> bool:
    _sum = cast(Tensor, t.sum(axis).type(torch.float32)) #在c维度上的和，应当加起来为1！
    _ones = torch.ones_like(_sum, dtype=torch.float32)
    return torch.allclose(_sum, _ones)


def one_hot(t: Tensor, axis=1) -> bool: #判断是否是one_hot的
    if not simplex(t, axis):
        print(t.shape)
    return simplex(t, axis) and sset(t, [0, 1])


def class2one_hot(seg: Tensor, K: int) -> Tensor:
    # Breaking change but otherwise can't deal with both 2d and 3d
    # if len(seg.shape) == 3:  # Only w, h, d, used by the dataloader
    #     return class2one_hot(seg.unsqueeze(dim=0), K)[0]

    assert sset(seg, list(range(K))), (uniq(seg), K)

    b, *img_shape = seg.shape  # type: Tuple[int, ...]
    device = seg.device
    res = torch.zeros((b, K, *img_shape), dtype=torch.int32, device=device).scatter_(1, seg[:, None, ...], 1)

    assert res.shape == (b, K, *img_shape)
    assert one_hot(res)
    return res


def sset(a: Tensor, sub: Iterable) -> bool:
    return uniq(a).issubset(sub)


def uniq(a: Tensor) -> Set:
    return set(torch.unique(a.cpu()).numpy())


def one_hot2dist(seg: np.ndarray, resolution: Tuple[float, float, float] = None,
                 dtype=None) -> np.ndarray:
    # 它接受的是c,w,h的输入，不可以输入n,c,w,h
    assert one_hot(torch.tensor(seg), axis=0)
    K: int = len(seg)

    res = np.zeros_like(seg, dtype=dtype)
    for k in range(K):
        posmask = seg[k].astype(np.bool)

        if posmask.any():
            negmask = ~posmask
            res[k] = eucl_distance(negmask, sampling=resolution) * negmask \
                - (eucl_distance(posmask, sampling=resolution) - 1) * posmask
        # The idea is to leave blank the negative classes
        # since this is one-hot encoded, another class will supervise that pixel

    return res


def gt_transform(resolution: Tuple[float, ...], K: int) -> Callable[[D], Tensor]:
        # 返回的是函数
        # 输入的本就是tensor状态的 label
        return transforms.Compose([
                lambda img: np.array(img)[...],
                lambda nd: torch.tensor(nd, dtype=torch.int64)[None, ...],  # Add one dimension to simulate batch
                partial(class2one_hot, K=K),
                itemgetter(0)  # Then pop the element to go back to img shape
        ])


def dist_map_transform(resolution: Tuple[float, ...], K: int) -> Callable[[D], Tensor]:
        return transforms.Compose([
                gt_transform(resolution, K),
                lambda t: t.detach().cpu().numpy(),
                partial(one_hot2dist, resolution=resolution),
                lambda nd: torch.tensor(nd, dtype=torch.float32)
        ])
