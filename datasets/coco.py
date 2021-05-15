import torch.utils.data as data
import numpy as np
import cv2
from pycocotools.coco import COCO
import sys,os
sys.path.append('../')
from utils.utils import dist_map_transform


class cocoSegmentation(data.Dataset):
    def __init__(self,
                 args,
                 root,
                 year='2017',
                 image_set='train',
                 transform=None):

        dataType = image_set+year
        annFile = '{}/annotations/instances_{}.json'.format(root, dataType)
        self.img_path = os.path.join(root, "images", dataType)
        self.coco = COCO(annFile)
        self.catId_person = self.coco.getCatIds(catNms=['person'])
        self.imgIds = self.coco.getImgIds(catIds=self.catId_person)
        if image_set == 'train':
            # 训练数量减半
            half_imgIds = self.imgIds[0:20000]
            self.imgIds = half_imgIds
        self.transform = transform
        self.disttransform = dist_map_transform([1, 1], 2)
        self.args = args
        self.root = root

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img_info = self.coco.loadImgs(self.imgIds[index])[0]
        img = cv2.imread(os.path.join(self.img_path, img_info['file_name']), cv2.IMREAD_COLOR)
        mask = self.getBinaryMask(img_info, self.coco, self.catId_person, img.shape)
        img = self.my_resize(img, self.args.input_image_size)
        mask = self.my_resize(mask[:, :, np.newaxis], self.args.input_image_size)

        """img = Image.fromarray(np.uint8(img))
        mask = Image.fromarray(np.uint8(mask))"""
        if self.transform is not None:
            img, mask = self.transform(img), self.transform(mask)[0]
        dist_map = self.disttransform(mask)
        return img, mask, dist_map
 
    def __len__(self):
        return len(self.imgIds)

    def my_resize(self, input, out_size):
        ox, oy = out_size
        input_size = input.shape
        ix, iy, ic = input_size[0], input_size[1], input_size[2]
        #解决img 和mask 通道数不同的问题
        out_size = (ox, oy, ic)

        max_i = max(ix, iy)
        # 如果图片最大的比要求的大，就应该缩小，如果图片最大的比要求的小，就应当按照最大的放大
        scale = ox / max_i
        img_buffer = cv2.resize(src=input, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        if ic == 1:
            img_buffer = img_buffer[:,:,np.newaxis]

        output = np.zeros(out_size)

        begin_x = int(out_size[0] / 2 - (ix*scale / 2))
        begin_y = int(out_size[1] / 2 - (iy*scale / 2))
        if begin_x < 0:
            print(begin_x)
            begin_x = 0
        if begin_y < 0:
            print(begin_y)
            begin_y = 0
        if begin_x > out_size[0]:
            print(begin_x)
            begin_x = out_size[0]
        if begin_y > out_size[1]:
            print(begin_y)
            begin_y = out_size[1]

        output[begin_x:begin_x+img_buffer.shape[0], begin_y:begin_y+img_buffer.shape[1], : ] = img_buffer

        return output

    def getClassName(self, classID, cats):
        for i in range(len(cats)):
            if cats[i]['id'] == classID:
                return cats[i]['name']
        return None

    def getNormalMask(self, imageObj, classes, coco, catIds, input_image_size):
        annIds = coco.getAnnIds(imageObj['id'], catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)
        cats = coco.loadCats(catIds)
        train_mask = np.zeros(input_image_size)
        for a in range(len(anns)):
            className = self.getClassName(anns[a]['category_id'], cats)
            pixel_value = classes.index(className) + 1
            new_mask = cv2.resize(coco.annToMask(anns[a]) * pixel_value, input_image_size)
            train_mask = np.maximum(new_mask, train_mask)
        print(train_mask.shape)
        # Add extra dimension for parity with train_img size [X * X * 3]
        # train_mask = train_mask.reshape(input_image_size[0], input_image_size[1], 1)
        return train_mask

    def getBinaryMask(self, imageObj, coco, catIds, input_image_size):
        if len(input_image_size) ==3:
            input_image_size = input_image_size[:-1]
        annIds = coco.getAnnIds(imageObj['id'], catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)
        train_mask = np.zeros(input_image_size)
        for a in range(len(anns)):
            new_mask = cv2.resize(coco.annToMask(anns[a]), (input_image_size[1], input_image_size[0]))

            # Threshold because resizing may cause extraneous values
            new_mask[new_mask >= 0.5] = 1
            new_mask[new_mask < 0.5] = 0
            train_mask = np.maximum(new_mask, train_mask)


        # Add extra dimension for parity with train_img size [X * X * 3]
        # train_mask = train_mask.reshape(input_image_size[0], input_image_size[1], 1)
        return train_mask
