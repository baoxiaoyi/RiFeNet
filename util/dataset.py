import os
import os.path
import cv2
import numpy as np

from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import random
import time
import json

from tqdm import tqdm
from torchvision.transforms import ColorJitter
IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']



def is_image_file(filename):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(split=0, data_root=None, data_list=None, sub_list=None, filter_intersection = False):
    assert split in [0, 1, 2, 3, 10, 11, 999]
    if not os.path.isfile(data_list):
        raise (RuntimeError("Image list file do not exist: " + data_list + "\n"))

    split_data_list = data_list.split('.')[0] + '_split{}'.format(split) + '.pth'
    #split_data
    if os.path.isfile(split_data_list):
        image_label_list, sub_class_file_list = torch.load(split_data_list)
        return image_label_list, sub_class_file_list
    # Shaban uses these lines to remove small objects:
    # if util.change_coordinates(mask, 32.0, 0.0).sum() > 2:
    #    filtered_item.append(item)      
    # which means the mask will be downsampled to 1/32 of the original size and the valid area should be larger than 2, 
    # therefore the area in original size should be accordingly larger than 2 * 32 * 32    
    image_label_list = []  
    # list_read = open(data_list).readlines()
    list_read = json.load(open(data_list))
    print("Processing data...")
    sub_class_file_list = {}
    for sub_c in sub_list:
        sub_class_file_list[sub_c] = []
    for l_idx in tqdm(range(len(list_read))):
        line = list_read[l_idx]
        # line = line.strip()
        # line_split = line.split(' ')
        line_split = line
        image_name = os.path.join(data_root, line_split[0])
        label_name = os.path.join(data_root, line_split[1])
        item = (image_name, label_name)
        label = cv2.imread(label_name, cv2.IMREAD_GRAYSCALE)
        label_class = np.unique(label).tolist()

        if 0 in label_class:
            label_class.remove(0)
        if 255 in label_class:
            label_class.remove(255)

        new_label_class = []
        if filter_intersection:  # filter images containing objects of novel categories during meta-training
            if set(label_class).issubset(set(sub_list)):
                for c in label_class:
                    if c in sub_list:
                        tmp_label = np.zeros_like(label)
                        target_pix = np.where(label == c)
                        tmp_label[target_pix[0],target_pix[1]] = 1
                        if tmp_label.sum() >= 2 * 32 * 32:
                            new_label_class.append(c)
        else:
            for c in label_class:
                if c in sub_list:
                    tmp_label = np.zeros_like(label)
                    target_pix = np.where(label == c)
                    tmp_label[target_pix[0],target_pix[1]] = 1
                    if tmp_label.sum() >= 2 * 32 * 32:
                        new_label_class.append(c)

        label_class = new_label_class    

        if len(label_class) > 0:
            image_label_list.append(item)
            for c in label_class:
                if c in sub_list:
                    sub_class_file_list[c].append(item)
                    
    print("Checking image&label pair {} list done! ".format(split))
    print("Saving processed data...")
    torch.save((image_label_list, sub_class_file_list), split_data_list)
    print("Done")
    return image_label_list, sub_class_file_list



class SemData(Dataset):
    def __init__(self, split=3, shot=1, data_root=None, data_list=None, transform=None, strong_transform=False, mode='train', use_coco=False, use_split_coco=False,num_unlabel=2):
        assert mode in ['train', 'val', 'test']
        
        self.mode = mode
        self.split = split  
        self.shot = shot
        self.data_root = data_root   
        self.use_coco = use_coco
        self.num_unlabel = num_unlabel

        if not use_coco:
            self.class_list = list(range(1, 21)) #[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
            if self.split == 3: 
                self.sub_list = list(range(1, 16)) #[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
                self.sub_val_list = list(range(16, 21)) #[16,17,18,19,20]
            elif self.split == 2:
                self.sub_list = list(range(1, 11)) + list(range(16, 21)) #[1,2,3,4,5,6,7,8,9,10,16,17,18,19,20]
                self.sub_val_list = list(range(11, 16)) #[11,12,13,14,15]
            elif self.split == 1:
                self.sub_list = list(range(1, 6)) + list(range(11, 21)) #[1,2,3,4,5,11,12,13,14,15,16,17,18,19,20]
                self.sub_val_list = list(range(6, 11)) #[6,7,8,9,10]
            elif self.split == 0:
                self.sub_list = list(range(6, 21)) #[6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
                self.sub_val_list = list(range(1, 6)) #[1,2,3,4,5]

        else:
            if use_split_coco:
                print('INFO: using SPLIT COCO')
                self.class_list = list(range(1, 81))
                if self.split == 3:
                    self.sub_val_list = list(range(4, 81, 4))
                    self.sub_list = list(set(self.class_list) - set(self.sub_val_list))                    
                elif self.split == 2:
                    self.sub_val_list = list(range(3, 80, 4))
                    self.sub_list = list(set(self.class_list) - set(self.sub_val_list))    
                elif self.split == 1:
                    self.sub_val_list = list(range(2, 79, 4))
                    self.sub_list = list(set(self.class_list) - set(self.sub_val_list))    
                elif self.split == 0:
                    self.sub_val_list = list(range(1, 78, 4))
                    self.sub_list = list(set(self.class_list) - set(self.sub_val_list))    
            else:
                print('INFO: using COCO')
                self.class_list = list(range(1, 81))
                if self.split == 3:
                    self.sub_list = list(range(1, 61))
                    self.sub_val_list = list(range(61, 81))
                elif self.split == 2:
                    self.sub_list = list(range(1, 41)) + list(range(61, 81))
                    self.sub_val_list = list(range(41, 61))
                elif self.split == 1:
                    self.sub_list = list(range(1, 21)) + list(range(41, 81))
                    self.sub_val_list = list(range(21, 41))
                elif self.split == 0:
                    self.sub_list = list(range(21, 81)) 
                    self.sub_val_list = list(range(1, 21))    

        print('sub_list: ', self.sub_list)
        print('sub_val_list: ', self.sub_val_list)
        if self.mode == 'train':
            self.data_list, self.sub_class_file_list = make_dataset(split, data_root, data_list, self.sub_list,True)
            assert len(self.sub_class_file_list.keys()) == len(self.sub_list)
        elif self.mode == 'val':
            self.data_list, self.sub_class_file_list = make_dataset(split, data_root, data_list, self.sub_val_list,False)
            assert len(self.sub_class_file_list.keys()) == len(self.sub_val_list) 
        self.transform = transform
        self.strong_transform = None
        if strong_transform:
            self.strong_transform = ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.25)


    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        label_class = []
        image_path, label_path = self.data_list[index]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR) 
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
        image = np.float32(image)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)  

        padding_mask = np.zeros_like(label)

        if image.shape[0] != label.shape[0] or image.shape[1] != label.shape[1]:
            raise (RuntimeError("Query Image & label shape mismatch: " + image_path + " " + label_path + "\n"))          
        label_class = np.unique(label).tolist()
        if 0 in label_class:
            label_class.remove(0)
        if 255 in label_class:
            label_class.remove(255) 
        new_label_class = []       
        for c in label_class:
            if c in self.sub_val_list:
                if self.mode == 'val' or self.mode == 'test':
                    new_label_class.append(c)
            if c in self.sub_list:
                if self.mode == 'train':
                    new_label_class.append(c)
        label_class = new_label_class    
        assert len(label_class) > 0


        class_chosen = label_class[random.randint(1,len(label_class))-1]
        class_chosen = class_chosen
        target_pix = np.where(label == class_chosen)
        ignore_pix = np.where(label == 255)
        label[:,:] = 0
        if target_pix[0].shape[0] > 0:
            label[target_pix[0],target_pix[1]] = 1 
        label[ignore_pix[0],ignore_pix[1]] = 255           


        file_class_chosen = self.sub_class_file_list[class_chosen]
        num_file = len(file_class_chosen)

        support_image_path_list = []
        support_label_path_list = []
        support_idx_list = []
        for k in range(self.shot):
            support_idx = random.randint(1,num_file)-1
            support_image_path = image_path
            support_label_path = label_path
            while((support_image_path == image_path and support_label_path == label_path) or support_idx in support_idx_list):
                support_idx = random.randint(1,num_file)-1
                support_image_path, support_label_path = file_class_chosen[support_idx]                
            support_idx_list.append(support_idx)
            support_image_path_list.append(support_image_path)
            support_label_path_list.append(support_label_path)

        support_image_list = []
        support_label_list = []
        support_padding_list = []
        subcls_list = []
        for k in range(self.shot):  
            if self.mode == 'train':
                subcls_list.append(self.sub_list.index(class_chosen))
            else:
                subcls_list.append(self.sub_val_list.index(class_chosen))
            support_image_path = support_image_path_list[k]
            support_label_path = support_label_path_list[k] 
            support_image = cv2.imread(support_image_path, cv2.IMREAD_COLOR)      
            support_image = cv2.cvtColor(support_image, cv2.COLOR_BGR2RGB)
            support_image = np.float32(support_image)
            support_label = cv2.imread(support_label_path, cv2.IMREAD_GRAYSCALE)
            target_pix = np.where(support_label == class_chosen)
            ignore_pix = np.where(support_label == 255)
            support_label[:,:] = 0
            support_label[target_pix[0],target_pix[1]] = 1 
            support_label[ignore_pix[0],ignore_pix[1]] = 255
            if support_image.shape[0] != support_label.shape[0] or support_image.shape[1] != support_label.shape[1]:
                raise (RuntimeError("Support Image & label shape mismatch: " + support_image_path + " " + support_label_path + "\n"))     

            if not self.use_coco:
                support_padding_label = np.zeros_like(support_label)
                support_padding_label[support_label==255] = 255
            else:
                support_padding_label = np.zeros_like(support_label)
            support_image_list.append(support_image)
            support_label_list.append(support_label)
            support_padding_list.append(support_padding_label)
        assert len(support_label_list) == self.shot and len(support_image_list) == self.shot                    
        
        raw_label = label.copy()
        if self.transform is not None:
            image, label, padding_mask = self.transform(image, label, padding_mask)
            for k in range(self.shot):
                support_image_list[k], support_label_list[k], support_padding_list[k] = self.transform(support_image_list[k], support_label_list[k], support_padding_list[k])

        s_xs = support_image_list
        s_ys = support_label_list
        s_x = s_xs[0].unsqueeze(0)
        for i in range(1, self.shot):
            s_x = torch.cat([s_xs[i].unsqueeze(0), s_x], 0)
        s_y = s_ys[0].unsqueeze(0)
        for i in range(1, self.shot):
            s_y = torch.cat([s_ys[i].unsqueeze(0), s_y], 0)

        if support_padding_list is not None:
            s_eys = support_padding_list
            s_ey = s_eys[0].unsqueeze(0)
            for i in range(1, self.shot):
                s_ey = torch.cat([s_eys[i].unsqueeze(0), s_ey], 0)

        n_unlabel = self.num_unlabel
        unlabel_image_path_list = []
        unlabel_label_path_list = []
        unlabel_idx_list = []
        for k in range(n_unlabel):
            unlabel_idx = random.randint(1, num_file) - 1
            unlabel_image_path = image_path
            unlabel_label_path = label_path
            while ((unlabel_image_path == image_path and unlabel_label_path == label_path) or unlabel_idx in unlabel_idx_list):
                unlabel_idx = random.randint(1, num_file) - 1
                unlabel_image_path, unlabel_label_path = file_class_chosen[unlabel_idx]
            unlabel_idx_list.append(unlabel_idx)
            unlabel_image_path_list.append(unlabel_image_path)
            unlabel_label_path_list.append(unlabel_label_path)

        unlabel_image_list = []
        unlabel_label_list = []
        unlabel_padding_list = []
        unstrong_image_list = []
        for k in range(n_unlabel):
            unlabel_image_path = unlabel_image_path_list[k]
            unlabel_label_path = unlabel_label_path_list[k]
            unlabel_image = cv2.imread(unlabel_image_path, cv2.IMREAD_COLOR)
            unlabel_image = cv2.cvtColor(unlabel_image, cv2.COLOR_BGR2RGB)
            unlabel_image = np.float32(unlabel_image)
            unlabel_label = cv2.imread(unlabel_label_path, cv2.IMREAD_GRAYSCALE)
            target_pix = np.where(unlabel_label == class_chosen)
            ignore_pix = np.where(unlabel_label == 255)
            unlabel_label[:, :] = 0
            unlabel_label[target_pix[0], target_pix[1]] = 1
            unlabel_label[ignore_pix[0], ignore_pix[1]] = 255
            if unlabel_image.shape[0] != unlabel_label.shape[0] or unlabel_image.shape[1] != unlabel_label.shape[1]:
                raise (RuntimeError(
                    "unlabel Image & label shape mismatch: " + unlabel_image_path + " " + unlabel_label_path + "\n"))

            if not self.use_coco:
                unlabel_padding_label = np.zeros_like(unlabel_label)
                unlabel_padding_label[unlabel_label == 255] = 255
            else:
                unlabel_padding_label = np.zeros_like(unlabel_label)
            unlabel_image_list.append(unlabel_image)
            unlabel_label_list.append(unlabel_label)
            unlabel_padding_list.append(unlabel_padding_label)
            unstrong_image_list.append(unlabel_image)
        assert len(unlabel_label_list) == n_unlabel and len(unlabel_image_list) == n_unlabel

        if self.transform is not None:
            for k in range(n_unlabel):
                unlabel_image_list[k], unlabel_label_list[k], unlabel_padding_list[k] = self.transform(
                    unlabel_image_list[k], unlabel_label_list[k], unlabel_padding_list[k])
                if self.strong_transform is not None:
                    while True:
                        unstrong_image_list[k] = self.strong_transform(unlabel_image_list[k])
                        if torch.isnan(unstrong_image_list[k]).int().sum() == 0:
                            break
                #assert (torch.isnan(unstrong_image_list[k]).int().sum() == 0), 'dataout nan' + unlabel_image_path_list[k]

        u_xs = unlabel_image_list
        u_x = u_xs[0].unsqueeze(0)
        for i in range(1, n_unlabel):
            u_x = torch.cat([u_xs[i].unsqueeze(0), u_x], 0)
        #print(len(u_x)) 10
        #print(u_x[0].size())  [3,473,473]
        if self.strong_transform is not None:
            u_strongs = unstrong_image_list
            u_strong = u_strongs[0].unsqueeze(0)
            for i in range(1, n_unlabel):
                u_strong = torch.cat([u_strongs[i].unsqueeze(0), u_strong], 0)

        if unlabel_padding_list is not None:
            u_eys = unlabel_padding_list
            u_ey = u_eys[0].unsqueeze(0)
            for i in range(1, self.num_unlabel):
                u_ey = torch.cat([u_eys[i].unsqueeze(0), u_ey], 0)

        if self.mode == 'train':
            return image, label, s_x, s_y, padding_mask, s_ey, u_ey, u_x, u_strong, subcls_list
            #return image, label, s_x, s_y, padding_mask, s_ey, u_x, subcls_list
        else:
            return image, label, s_x, s_y, padding_mask, s_ey, u_ey, u_x, subcls_list, raw_label

