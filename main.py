from tqdm import tqdm
import network
import utils
import os
import random
import numpy as np

from metrics import StreamSegMetrics

import torch
from utils.visualizer import Visualizer
from trainer import trainer

from config.coco_person_config import parser

def main():
    opts = parser.parse_args()

    vis = Visualizer(port=opts.vis_port,
                     env=opts.vis_env) if opts.enable_vis else None
    if vis is not None:  # display options
        vis.vis_table("Options", vars(opts))

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    # Setup random seed
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    # Set up model
    model_map = {
        'deeplabv3_resnet50': network.deeplabv3_resnet50,
        'deeplabv3plus_resnet50': network.deeplabv3plus_resnet50,
        'deeplabv3_resnet101': network.deeplabv3_resnet101,
        'deeplabv3plus_resnet101': network.deeplabv3plus_resnet101,
        'deeplabv3_mobilenet': network.deeplabv3_mobilenet,
        'deeplabv3plus_mobilenet': network.deeplabv3plus_mobilenet
    }

    model = model_map[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
    if opts.separable_conv and 'plus' in opts.model:
        network.convert_to_separable_conv(model.classifier)
    utils.set_bn_momentum(model.backbone, momentum=0.01)

    # Set up metrics
    metrics = StreamSegMetrics(opts.num_classes)

    # Set up optimizer
    optimizer = torch.optim.SGD(params=[
        {'params': model.backbone.parameters(), 'lr': 0.1 * opts.lr},
        {'params': model.classifier.parameters(), 'lr': opts.lr},
    ], lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
    # optimizer = torch.optim.SGD(params=model.parameters(), lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
    # torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.lr_decay_step, gamma=opts.lr_decay_factor)
    if opts.lr_policy == 'poly':
        scheduler = utils.PolyLR(optimizer, opts.total_itrs, power=0.9)
    elif opts.lr_policy == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.step_size, gamma=0.1)
    else:
        scheduler = None
        print("please assign a scheduler!")



    utils.mkdir('checkpoints')
    mytrainer = trainer(model, optimizer, scheduler, device, cfg=opts)
    # ==========   Train Loop   ==========#
    #loss_list = ['bound_dice', 'v3_bound_dice']
    #loss_list = ['v5_bound_dice', 'v4_bound_dice']
    loss_list = ['focal']
    if opts.test_only:
        loss_i = 'v3'
        ckpt = os.path.join("checkpoints", loss_i, "latest_deeplabv3plus_mobilenet_coco_epoch01.pth")
        mytrainer.validate(ckpt, loss_i)
    else:
        for loss_i in loss_list:
            mytrainer.train(loss_i)


if __name__ == '__main__':
    main()
