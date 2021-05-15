import argparse

parser = argparse.ArgumentParser()

# Datset Options
parser.add_argument("--data_root", type=str, default='/home/yangtingyang/yty/ltt/data/coco/',
                    help="path to Dataset")
parser.add_argument("--dataset", type=str, default='coco',
                    choices=['voc', 'cityscapes', 'coco'], help='Name of dataset')
parser.add_argument("--num_classes", type=int, default=2,
                    help="num classes (default: None)")
parser.add_argument("--input-image-size", type=tuple, default=(512, 512), help="all image will be resize to this")

# Deeplab Options
parser.add_argument("--model", type=str, default='deeplabv3plus_mobilenet',
                    choices=['deeplabv3_resnet50', 'deeplabv3plus_resnet50',
                             'deeplabv3_resnet101', 'deeplabv3plus_resnet101',
                             'deeplabv3_mobilenet', 'deeplabv3plus_mobilenet'], help='model name')
parser.add_argument("--separable_conv", action='store_true', default=False,
                    help="apply separable conv to decoder and aspp")
parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

# Loss Options
parser.add_argument('--use_strategy', action='store_true')
parser.add_argument('--version', type=int, default=5)

# Train Options
parser.add_argument("--test_only", action='store_true', default=False)
parser.add_argument("--save_val_results", action='store_true', default=False,
                    help="save segmentation results to \"./results\"")
parser.add_argument("--eval_every_epoch", default=False, action='store_true')
parser.add_argument("--total_itrs", type=int, default=30e3,
                    help="epoch number (default: 30k)")
parser.add_argument("--total_epochs", type=int, default=30,
                    help="epoch number (default: 30)")
parser.add_argument("--lr", type=float, default=4e-5,
                    help="learning rate (default: 0.01)")
parser.add_argument("--lr_policy", type=str, default='step', choices=['poly', 'step'],
                    help="learning rate scheduler policy")
parser.add_argument("--step_size", type=int, default=10000)

#about train data
parser.add_argument("--crop_val", action='store_true', default=False,
                    help='crop validation (default: False)')
parser.add_argument("--batch_size", type=int, default=16,
                    help='batch size (default: 16)')
parser.add_argument("--val_batch_size", type=int, default=1,
                    help='batch size for validation (default: 1)')
parser.add_argument("--crop_size", type=int, default=513)


#path
parser.add_argument("--ckpt", default=None, type=str,
                    help="restore from checkpoint")
parser.add_argument("--continue_training", action='store_true', default=False)

#necessary part
parser.add_argument("--loss_type", type=str, default='cross_entropy',
                    choices=['cross_entropy', 'focal_loss'], help="loss type (default: False)")
parser.add_argument("--gpu_id", type=str, default='3',
                    help="GPU ID")
parser.add_argument("--weight_decay", type=float, default=1e-4,
                    help='weight decay (default: 1e-4)')
parser.add_argument("--random_seed", type=int, default=1,
                    help="random seed (default: 1)")
parser.add_argument("--print_interval", type=int, default=30,
                    help="print interval of loss (default: 10)")
parser.add_argument("--val_interval", type=int, default=100,
                    help="epoch interval for eval (default: 100)")
parser.add_argument("--download", action='store_true', default=False,
                    help="download datasets")

# PASCAL VOC Options
parser.add_argument("--year", type=str, default='2017',
                    choices=['2012_aug', '2012', '2011', '2009', '2008', '2017'], help='year of coco')

# Visdom options
parser.add_argument("--enable_vis", action='store_true', default=False,
                    help="use visdom for visualization")
parser.add_argument("--vis_port", type=str, default='9901',
                    help='port for visdom')
parser.add_argument("--vis_env", type=str, default='main',
                    help='env for visdom')
parser.add_argument("--vis_num_samples", type=int, default=8,
                    help='number of samples for visualization (default: 8)')

