import os
import cv2
import numpy as np
import torch
import myseg.tv_transform as my_transforms
import myseg.cv2_transform as cv2_transforms
from PIL import Image


def read_images_dir(root_dir, folder, is_label=False):
    city_names = sorted(os.listdir(os.path.join(root_dir, folder)))
    img_dirs = []
    for city_name in city_names:
        img_names = os.listdir(os.path.join(root_dir, folder, city_name))
        for img_name in img_names:
            if is_label == False:
                img_dirs.append(os.path.join(root_dir, folder, city_name, img_name))
            if is_label == True:
                img_dirs.append(os.path.join(root_dir, folder, city_name, img_name))

    img_dirs = sorted(img_dirs)
    list_ = []
    list_2 = []
    last_c = None
    for ii,ll in enumerate(img_dirs):
        cn = ll.split('/')[-2]    
        if cn not in list_:
            list_.append(cn)
        if cn !=last_c:
            list_2.append(ii)
            last_c = cn
    print(list_)
    print(list_2)
    return img_dirs

class CamVid_Dataset(torch.utils.data.Dataset):
    train_folder = "images/train"
    train_lb_folder = "labels/train"
    val_folder = "images/val"
    val_lb_folder = "labels/val"


    def __init__(self, args,root_dir, split='train'):
        self.root_dir = root_dir
        self.split = split  # 'train', 'val', 'test'
        self.args = args

        if self.split == 'train':
            self.image_dirs, self.label_dirs = \
                read_images_dir(root_dir, self.train_folder), read_images_dir(root_dir, self.train_lb_folder, is_label=True)
        elif self.split == 'val':
            self.image_dirs, self.label_dirs = \
                read_images_dir(root_dir, self.val_folder), read_images_dir(root_dir, self.val_lb_folder, is_label=True)

        assert len(self.image_dirs) == len(self.label_dirs), 'errors'
        print('find ' + str(len(self.image_dirs)) + ' examples')



    def __getitem__(self, idx):


        # 读入图片
        if self.args.dataset=='voc':
            image = Image.open(self.image_dirs[idx])
            image = np.array(image)
            label = Image.open(self.label_dirs[idx])
            label = np.array(label)

        else:
            image = cv2.imread(self.image_dirs[idx], cv2.IMREAD_COLOR)[:, :, ::-1]
            label = cv2.imread(self.label_dirs[idx], cv2.IMREAD_GRAYSCALE)
        # print(image.shape, label.shape)  # (1024, 2048, 3) (1024, 2048)

        # 将label进行remap
        #label = self.lb_map[label]
        if self.args.dataset=='camvid':

            if self.split == 'val':  
                label = np.uint8(label)-1
        elif self.args.dataset=='ade20k':
            label = np.uint8(label)-1
        elif self.args.dataset=='voc':
            label[label==255]=0
            label =  np.uint8(label)-1
        scale_ = 512
        if  self.args.dataset=='voc' or self.args.dataset=='ade20k':
            scale_ = 480


        # transform : 同时处理image和label
        image_label = dict(im=image, lb=label)

        if self.split == 'train':
            image_label = cv2_transforms.TransformationTrain(scales=(0.5, 1.5), cropsize=(scale_, scale_))(image_label)

        if self.split == 'val':
            image_label = cv2_transforms.TransformationVal()(image_label)

        # ToTensor
        image_label = cv2_transforms.ToTensor(
            mean=(0.3257, 0.3690, 0.3223),  # city, rgb
            std=(0.2112, 0.2148, 0.2115),
        )(image_label)

        image, label = image_label['im'], image_label['lb']
        # print(image.shape, label.shape) # torch.Size([3, 512, 1024]) torch.Size([512, 1024])

        return (image, label)

    def __len__(self):
        return len(self.image_dirs)

def load_dataiter(root_dir, batch_size, use_DDP=False):
    num_workers = 16
    image_transform, label_transform = my_transforms.get_transform()
    train_set = CamVid_Dataset(root_dir, 'train')
    test_set = CamVid_Dataset(root_dir, 'val')

    if use_DDP:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)

        train_iter = torch.utils.data.DataLoader(
            train_set, batch_size, shuffle=False, drop_last=True, num_workers=num_workers, sampler=train_sampler)
        test_iter = torch.utils.data.DataLoader(
            test_set, batch_size, drop_last=True, num_workers=num_workers)
    else:
        train_iter = torch.utils.data.DataLoader(
            train_set, batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
        test_iter = torch.utils.data.DataLoader(
            test_set, batch_size, drop_last=True, num_workers=num_workers)

    return train_iter, test_iter


if __name__ == '__main__':
    train_iter, test_iter = load_dataiter(root_dir='../data/cityscapes', batch_size=16)
    print("train_iter.len", len(train_iter))
    print("test_iter.len", len(test_iter))
    for i, (images, labels) in enumerate(train_iter):
        if (i < 3):
            print(images.size(), labels.size())  # torch.Size([16, 3, 512, 1024]) torch.Size([16, 512, 1024])
            print(labels.dtype)
            print(labels[0, 245:255, 250:260])
