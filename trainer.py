import numpy as np
import torch
import torch.nn as nn
from utils.loss import MultiLosses
from utils.utils import state_transform
from utils.visualizer import Visualizer
from tqdm import tqdm
import scipy.io as scio
from torch.utils import data
from datasets.utils import get_dataset
from metrics import StreamSegMetrics
import os
import cv2

class trainer():
    def __init__(self, model, optimizer, scheduler, device, cfg):
        self.scheduler = scheduler
        self.model = model
        self.cfg = cfg
        self.optimizer = optimizer
        self.device = device
        self.loss_function = MultiLosses(device=device)

        # Setup dataloader
        self.train_dst, self.val_dst = get_dataset(self.cfg)
        self.train_loader = data.DataLoader(
            self.train_dst, batch_size=self.cfg.batch_size, shuffle=True, num_workers=8, pin_memory=True)
        self.val_loader = data.DataLoader(
            self.val_dst, batch_size=self.cfg.val_batch_size, shuffle=True, num_workers=8, pin_memory=True)
        print("Dataset: %s, Train set: %d, Val set: %d" %
              (self.cfg.dataset, len(self.train_dst), len(self.val_dst)))

        # visom setup
        vis = Visualizer(port=self.cfg.vis_port,
                         env=self.cfg.vis_env) if self.cfg.enable_vis else None
        if vis is not None:  # display options
            vis.vis_table("Options", vars(self.cfg))
        self.vis = vis
        self.vis_sample_id = np.random.randint(0, len(self.val_loader), self.cfg.vis_num_samples,
                                               np.int32) if self.cfg.enable_vis else None  # sample idxs for visualization

        # metric
        self.metrics = StreamSegMetrics(self.cfg.num_classes)

    def save_ckpt(self, path, cur_itrs, best_score):
        """ save current model
        """
        torch.save({
            "cur_itrs": cur_itrs,
            "model_state": self.model.module.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "best_score": best_score,
        }, path)
        print("Model saved as %s" % path)

    def load_model(self, ckpt=None):
        cur_itrs = 0
        best_score = 0
        print(ckpt)
        if ckpt is not None and os.path.isfile(ckpt):
            # https://github.com/VainF/DeepLabV3Plus-Pytorch/issues/8#issuecomment-605601402, @PytaichukBohdan
            checkpoint = torch.load(ckpt, map_location=torch.device('cuda:0'))
            old_state = checkpoint["model_state"]
            new_state = state_transform(old_state)
            self.model.load_state_dict(new_state)
            self.model = nn.DataParallel(self.model)
            self.model.to(self.device)
            if self.cfg.continue_training:
                self.optimizer.load_state_dict(checkpoint["optimizer_state"])
                self.scheduler.load_state_dict(checkpoint["scheduler_state"])
                cur_itrs = checkpoint["cur_itrs"]
                best_score = checkpoint['best_score']
                print("Training state restored from %s" % ckpt)
            print("Model restored from %s" % ckpt)
            del checkpoint  # free memory
        else:
            print("[!] Retrain")
            self.model = nn.DataParallel(self.model)
            self.model.to(self.device)
        return cur_itrs, best_score


    def train(self, loss_name):
        cur_epochs = 0

        preLoss = []
        loss_record = []
        last_5loss = {}
        lossChange = 0
        # load checkpoints from saved path
        cur_itrs, best_score = self.load_model()

        print("using loss : %s" % loss_name)
        os.makedirs("checkpoints/%s" % loss_name, exist_ok=True)
        while cur_epochs < self.cfg.total_epochs:
            # =====  Train  =====
            self.model.train()
            cur_epochs += 1
            for (images, labels, dist_maps) in self.train_loader:
                cur_itrs += 1

                images = images.to(self.device, dtype=torch.float32)
                labels = labels.to(self.device, dtype=torch.long)
                dist_maps = dist_maps.to(self.device, dtype=torch.float32)
                my_labels = {'label': labels, 'dist_map': dist_maps}
                self.optimizer.zero_grad()
                outputs = self.model(images)

                lossi = []
                maxgrad = 0
                idx = 0
                if 'v' in loss_name:
                    for lit, ll in enumerate(['Bound', 'dice']):
                        if lit not in last_5loss:
                            last_5loss[lit] = np.zeros(5)
                        self.loss_function.bulid_loss(ll)
                        loss = self.loss_function.loss(outputs, my_labels)

                        if 'v2' in loss_name:
                            if len(preLoss) != 0:
                                lossChange = (loss - preLoss[lit]) / preLoss[lit]

                            if (len(preLoss) == 0 or
                                    (lossChange > 0 and lossChange > maxgrad) or
                                    (lossChange < 0 and maxgrad < 0 and lossChange < maxgrad)):  # 更新要选择的loss
                                maxgrad = lossChange
                                idx = ll
                        elif 'v1' in loss_name:
                            if loss > maxgrad:
                                maxgrad = loss
                                idx = ll
                        elif 'v3' in loss_name or 'v5' in loss_name:
                            if len(preLoss) != 0:
                                lossChange = (loss - preLoss[lit]) / preLoss[lit]
                                if 'v5' in loss_name:
                                    if lossChange < 0: #反向的改变应当降低权重
                                        lossChange /= 1.5
                            if (len(preLoss) == 0 or abs(lossChange) > maxgrad):
                                maxgrad = abs(lossChange)
                                idx = ll
                        elif 'v4' in loss_name:
                            preLoss = np.mean(last_5loss[lit])
                            lossChange = (loss - preLoss) / preLoss
                            if (preLoss == 0 or abs(lossChange) > maxgrad):
                                maxgrad = abs(lossChange)
                                idx = ll
                        lossi.append(loss.item())
                        last_5loss[lit][cur_itrs % 5] = loss.item()
                    preLoss = lossi
                    if cur_itrs % self.cfg.print_interval == 0:
                        loss_record.append([lossi, idx])
                    self.loss_function.bulid_loss(str(idx))
                else:
                    self.loss_function.bulid_loss(loss_name)

                loss = self.loss_function.loss(outputs, my_labels)
                loss.backward()
                self.optimizer.step()

                np_loss = loss.item()
                if self.vis is not None:
                    self.vis.vis_scalar('Loss', cur_itrs, np_loss)

                if 'v' in loss_name:
                    #print("\riter: {} : criteria: {:.2f}, using loss {} | L2: {:.2f}, ce: {:.2f}, focal: {:.2f}, dice: {:.2f}".format(
                    #    cur_itrs, maxgrad, idx, lossi[0], lossi[1], lossi[2], lossi[3]), end='', flush=True)
                    print("\riter: {} : criteria: {:.2f}, using loss {} | boundary: {:.2f}, dice: {:.2f}".format(
                        cur_itrs, maxgrad, idx, lossi[0], lossi[1]), end='', flush=True)
                else:
                    print('\riter: {} | loss: {:.7f}'.format(cur_itrs, np_loss), end='', flush=True)
            # save ckpt after every epoch
            self.save_ckpt('checkpoints/%s/latest_%s_%s_epoch%02d.pth' %
                           (loss_name, self.cfg.model, self.cfg.dataset, cur_epochs), cur_itrs, best_score)
            print("save ckpt every epoch! cur_epoch: %d, best_score_%f" %
                  (cur_epochs, best_score))
            if self.cfg.eval_every_epoch:
                print("validation...", flush=True)
                self.model.eval()
                val_score, ret_samples = self.validate()
                print(self.metrics.to_str(val_score))
                if val_score['Mean IoU'] > best_score:  # save best model
                    best_score = val_score['Mean IoU']
                    self.save_ckpt('checkpoints/%s/best_%s_%s.pth' %
                                   (loss_name, self.cfg.model, self.cfg.dataset), cur_itrs, best_score)
                    print("save ckpt best! cur_epoch: %d, best_score_%f" %
                          (cur_epochs, best_score))

                if self.vis is not None:  # visualize validation score and samples
                    self.vis.vis_scalar("[Val] Overall Acc", cur_itrs, val_score['Overall Acc'])
                    self.vis.vis_scalar("[Val] Mean IoU", cur_itrs, val_score['Mean IoU'])
                    self.vis.vis_table("[Val] Class IoU", val_score['Class IoU'])

                    for k, (img, target, lbl) in enumerate(ret_samples):
                        img = img.astype(np.uint8)
                        target_rgb = np.zeros_like(img)
                        lbl_rgb = np.zeros_like(img)
                        for i in range(3):
                            target_rgb[i][target > 0] = 255
                            lbl_rgb[i][lbl > 0] = 255

                        target_rgb = target_rgb.astype(np.uint8)
                        lbl_rgb = lbl_rgb.astype(np.uint8)
                        concat_img = np.concatenate((img, target_rgb, lbl_rgb), axis=2)  # concat along width
                        self.vis.vis_image('Sample %d' % k, concat_img)
                else:
                    print("iter_%d : overall acc: %.4f, miou: %.4f, class_iou: [0: %.4f, 1: %.4f]" % (cur_itrs,
                                                     val_score['Overall Acc'],
                                                     val_score['Mean IoU'],
                                                     val_score['Class IoU'][0],
                                                     val_score['Class IoU'][1]), flush=True)
                self.model.train()
            self.scheduler.step() # 每个轮次，学习率都有可能不一样

        loss_record = np.array(loss_record)
        scio.savemat("./lossRecord-{}.mat".format(loss_name), mdict={'data': loss_record})

        print(loss_record.shape, "save loss record", loss_name)

    def validate(self, ckpt=None, loss_name='focal'):
        """Do validation and return specified samples"""
        opts = self.cfg
        model = self.model
        loader = self.val_loader
        device = self.device
        metrics = self.metrics
        ret_samples_ids = self.vis_sample_id
        if ckpt == None:
            print("if want val exist ckpt, please assign a checkpoint path!")
        else:
            self.load_model(ckpt)
        metrics.reset()
        model.eval()
        ret_samples = []
        if opts.save_val_results:
            if not os.path.exists('results/{}' % loss_name):
                os.mkdir('results/{}' % loss_name)
            img_id = 0

        with torch.no_grad():
            for i, (images, labels, _) in tqdm(enumerate(loader)):

                images = images.to(device, dtype=torch.float32)
                labels = labels.to(device, dtype=torch.long)

                outputs = model(images)
                preds = outputs.detach().max(dim=1)[1].cpu().numpy()
                targets = labels.detach().cpu().numpy()

                metrics.update(targets, preds)
                if ret_samples_ids is not None and i in ret_samples_ids:  # get vis samples
                    ret_samples.append(
                        (images[0].detach().cpu().numpy(), targets[0], preds[0]))

                if opts.save_val_results:
                    for img_id in range(len(images)):
                        image = images[img_id].detach().cpu().numpy()
                        target = targets[img_id]
                        pred = preds[img_id]

                        image = (image).transpose(1, 2, 0).astype(np.uint8)
                        target = target.astype(np.uint8)
                        target[target > 0] = 255
                        pred = pred.astype(np.uint8)
                        mask = np.zeros_like(image)
                        mask[pred > 0] = 255

                        cv2.imwrite('results/%s/%d_%d_image.png' % (loss_name, i, img_id), image)
                        cv2.imwrite('results/%s/%d_%d_target.png' % (loss_name, i, img_id), target)
                        cv2.imwrite('results/%s/%d_%d_pred.png' % (loss_name, i, img_id), mask)

                        mask_img = cv2.addWeighted(image, 1, mask, 0.5, 0)
                        cv2.imwrite('results/%s/%d_%d_overlay.png' % (loss_name, i, img_id), mask_img)

            score = metrics.get_results()
        print(metrics.to_str(score))
        return score, ret_samples


if __name__ == "__main__":
    mytrainer = trainer(None, None, None, None, None)