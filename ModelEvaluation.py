import time
import torch
from torch.utils import data
import os
import scipy.io as scio
from config.coco_person_config import parser
from utils.utils import state_transform
from datasets.coco import cocoSegmentation
import network
import utils
import torchvision.transforms as transforms
import numpy as np

cfg = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = cfg.gpu_id
device = torch.device('cuda:0')

transf = transforms.ToTensor()
val_dst = cocoSegmentation(args=cfg, root=cfg.data_root, year='2017',
                                   image_set='val', transform=transf)
val_loader = data.DataLoader(
    val_dst, batch_size=cfg.val_batch_size, shuffle=True, num_workers=8, pin_memory=True)

checkpointdir = './checkpoints/'
DNWdir = '/home/yangtingyang/yty/ltt/DeepLabV3Plus-Pytorch-master/res/'

def GetNorData(data):
    datamin = np.min(data)
    datamax = np.max(data)
    data = (data - datamin) / (datamax - datamin)
    return data

def CalIouDice2(pred, labd, thes):
    pred[pred < thes] = 0
    pred[pred > thes] = 1
    inte = np.multiply(labd, pred)

    intesum = np.sum(inte)
    Labdsum = np.sum(labd)
    nordsum = np.sum(pred)
    iou = intesum / (Labdsum + nordsum - intesum)
    dice = 2 * intesum / (Labdsum + nordsum)
    return iou, dice

def GetMeanIouDice2(res, labd, thes):
    Pred0 = res[0, :, :]
    Pred1 = res[1, :, :]
    labd0 = 1 - labd.copy()
    labd1 = labd.copy()
    nordata0 = GetNorData(Pred0.copy())
    nordata1 = GetNorData(Pred1.copy())
    iou0, dice0 = CalIouDice2(nordata0.copy(), labd0.copy(), thes)
    iou1, dice1 = CalIouDice2(nordata1.copy(), labd1.copy(), thes)
    iou = (iou0 + iou1) / 2.0
    dice = (dice0 + dice1) / 2.0
    return [iou, dice, iou0, dice0, iou1, dice1]

def getDNWmat(model, ckptdir, lossname):
    listnum = []
    foldName = os.path.join(DNWdir, time.strftime('%Y%m%d'))
    os.makedirs(foldName, exist_ok=True)
    for fn in os.listdir(ckptdir):
        spfn = fn.split('.')
        if len(spfn) > 1:
            spfn1 = spfn[0]
            print()
            num = int( spfn1[-2:] )

            listnum.append( num )

    listnum.sort()
    print( listnum )
    Evares = [ ]

    for i in range( 0, len(listnum) ):
        # if( listnum[i] > 60000 ):
        #     break
        Evares.append( [ ] )
        checkpoint_path = os.path.join(ckptdir, 'latest_deeplabv3plus_mobilenet_coco_epoch%02d' % listnum[i] + '.pth')

        print(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        old_state = checkpoint["model_state"]
        new_state = state_transform(old_state)
        model.load_state_dict(new_state)

        #model = nn.DataParallel(model)
        model.to(device)
        model.eval()
        mk= 0
        for (img_array, mask, dist_map) in val_loader:
            # img_array = torch.unsqueeze(img_array, 0)
            imgs = img_array.to(device, dtype=torch.float32)
            mask = mask.to(device, dtype=torch.long)

            masks_pred = model(imgs) #n, c, w,h
            res = masks_pred[0,:,:,:] #the first one
        
            res = res.cpu().detach().numpy() #c, w,h
            mask = mask.cpu().detach().numpy()#w,h
            Evares[i].append( [] )
            eva = GetMeanIouDice2( res.copy(), mask.copy(), 0.5 ) #以0.5为阈值，计算iou和dice
            #eva = GetMultiThresholdEva0(res.copy(), mask.copy(), 0.5)
            Evares[i][mk]=eva
            if( mk % 100 == 0 ):
                print( checkpoint_path, mk,
                       'mIoU={:.3f}, mdice={:.3f}, IoU0={:.3f}, dice0={:.3f}, IoU1={:.3f}, dice1={:.3f}'.format(
                           eva[0], eva[1], eva[2], eva[3], eva[4], eva[5]) ,flush=True)
            mk+=1
        
    Evares = np.array(Evares)
    print("end!!!!")
    scio.savemat( os.path.join(foldName, 'DNW_'+lossname+'.mat') , mdict = { 'data' : Evares } )



#loss_name = ['v4', 'focal', 'v1', 'dice', 'ce', 'L2', 'mix', 'Bound', 'v5_4loss', "v5_5loss"]
#loss_name = ['dice', 'Bound']
loss_name = ['L2', 'v3']
model = network.deeplabv3plus_mobilenet(num_classes=cfg.num_classes, output_stride=cfg.output_stride)
#network.convert_to_separable_conv(model.classifier)
utils.set_bn_momentum(model.backbone, momentum=0.01)
for lossi in loss_name:
    print("====================test "+lossi+" ======================")
    ckptlittledir = os.path.join(checkpointdir, lossi)
    getDNWmat(model, ckptlittledir, lossi)

print("all loss has been evaluated!")

