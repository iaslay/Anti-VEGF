import glob
import random
import os

import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torch



class ImageDataset(Dataset):
    def __init__(self, root,noise_level,count = None,transforms_1=None,transforms_2=None, unaligned=False):
        self.transform1 = transforms.Compose(transforms_1)
        self.transform2 = transforms.Compose(transforms_2)
        self.files_A = sorted(glob.glob("%s/A/*" % root))
        self.files_B = sorted(glob.glob("%s/B/*" % root))
        self.unaligned = unaligned
        self.noise_level =noise_level
        
    def __getitem__(self, index):
        if self.noise_level == 0:
            # if noise =0, A and B make same transform
            seed = np.random.randint(2147483647) # make a seed with numpy generator 
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            item_A = self.transform2(np.load(self.files_A[index % len(self.files_A)]).astype(np.float32))
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            item_B = self.transform2(np.load(self.files_B[index % len(self.files_B)]).astype(np.float32))
        else:
            # if noise !=0, A and B make different transform
            item_A = self.transform1(np.load(self.files_A[index % len(self.files_A)]).astype(np.float32))
            item_B = self.transform1(np.load(self.files_B[index % len(self.files_B)]).astype(np.float32))
            
        return {'A': item_A, 'B': item_B}
    def __len__(self):
        return max(len(self.files_A), len(self.files_B))


class ValDataset(Dataset):
    def __init__(self, root,count = None,transforms_=None, unaligned=False):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned
        self.files_A = sorted(glob.glob("%s/A/*" % root))
        self.files_B = sorted(glob.glob("%s/B/*" % root))
        self.root = root
        
    def __getitem__(self, index):
        item_A = self.transform(np.load(self.files_A[index % len(self.files_A)]).astype(np.float32))
        if self.unaligned:
            item_B = self.transform(np.load(self.files_B[random.randint(0, len(self.files_B) - 1)]))
        else:
            item_B = self.transform(np.load(self.files_B[index % len(self.files_B)]).astype(np.float32))
        return {'A': item_A, 'B': item_B}
    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

class EyeDataset(Dataset):
    def __init__(self, root,noise_level,count = None,transforms_1=None,transforms_2=None, unaligned=False, type='train'):
        self.transform1 = transforms.Compose(transforms_1)
        self.transform2 = transforms.Compose(transforms_2)
        self.unaligned = unaligned
        self.root = root
        self.type = type
        self.noise_level = noise_level
        self.label = pd.read_csv(f'{root}/label.csv')
        if type == 'train':
            self.filename = np.load(f'{self.root}/train.npy', allow_pickle=True)
        elif type=='val':
            self.filename = np.load(f'{self.root}/validation.npy', allow_pickle=True)
        elif type=='test':
            self.filename = np.load(f'{self.root}/test.npy', allow_pickle=True)
        else:
            self.filename = np.load(f'{self.root}/exter_test.npy', allow_pickle=True)

        print(len(self.filename))

    def __getitem__(self, item):
        if self.noise_level == 0 and self.type=='train':
            # if noise =0, A and B make same transform
            seed = np.random.randint(2147483647) # make a seed with numpy generator
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            img = cv2.imread(os.path.join(f'{self.root}/before' ,self.filename[item]), 0)
            img = (img-127.5)/127.5
            item_A = self.transform2(img.astype(np.float32))
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            img = cv2.imread(os.path.join(f'{self.root}/after',self.filename[item]), 0)
            img = (img - 127.5) / 127.5
            item_B = self.transform2(img.astype(np.float32))
        else:
            img_A = cv2.imread(os.path.join(f'{self.root}/before' ,self.filename[item]), 0)
            img_B = cv2.imread(os.path.join(f'{self.root}/after',self.filename[item]), 0)
            img_A = (img_A - 127.5) / 127.5
            img_B = (img_B - 127.5) / 127.5
            item_A = self.transform1(img_A.astype(np.float32))
            item_B = self.transform1(img_B.astype(np.float32))
        class_label = int(self.label.iloc[int(self.filename[item][:-4])]['label'])
        return {'A': item_A, 'B': item_B, 'name':self.filename[item], 'class_label':class_label}

    def __len__(self):
        return len(self.filename)

class fundusDataset(Dataset):
    def __init__(self, root,noise_level,count = None,transforms_1=None,transforms_2=None, transforms_3=None, unaligned=False, type='train'):
        self.transform1 = transforms.Compose(transforms_1)
        self.transform2 = transforms.Compose(transforms_2)
        self.transform3 = transforms.Compose(transforms_3)
        self.unaligned = unaligned
        self.root = root
        self.type = type
        self.noise_level = noise_level
        self.label = pd.read_csv(f'{root}/label.csv')
        if type == 'train':
            self.filename = np.load(f'{self.root}/train.npy', allow_pickle=True)
        else:
            self.filename = np.load(f'{self.root}/test.npy', allow_pickle=True)
        print(len(self.filename))

    def __getitem__(self, item):
        if self.noise_level == 0 and self.type=='train':
            # if noise =0, A and B make same transform
            seed = np.random.randint(2147483647) # make a seed with numpy generator
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            img = cv2.imread(os.path.join(f'{root}/before' ,self.filename[item]), 0)
            img = (img-127.5)/127.5
            item_A = self.transform2(img.astype(np.float32))
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            img = cv2.imread(os.path.join(f'{self.root}/after',self.filename[item]), 0)
            img = (img - 127.5) / 127.5
            item_B = self.transform2(img.astype(np.float32))
            fundus_idx = str(self.label.iloc[int(self.filename[item][:-4])]['fundus'])
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            img_C = cv2.imread(os.path.join(f'{self.root}/fundus', fundus_idx), 0)
            img_C = (img_C - 127.5) / 127.5
            img_C = self.transform3(img_C.astype(np.float32))
        else:
            # print(self.filename[item])
            fundus_idx = str(self.label.iloc[int(self.filename[item][:-4])]['fundus'])
            img_A = cv2.imread(os.path.join(f'{self.root}/before' ,self.filename[item]), 0)
            img_B = cv2.imread(os.path.join(f'{self.root}/after',self.filename[item]), 0)
            img_C = cv2.imread(os.path.join(f'{self.root}/fundus' , fundus_idx), 0)
            img_A = (img_A - 127.5) / 127.5
            img_B = (img_B - 127.5) / 127.5
            img_C = (img_C - 127.5) / 127.5
            item_A = self.transform1(img_A.astype(np.float32))
            item_B = self.transform1(img_B.astype(np.float32))
            img_C = self.transform1(img_C.astype(np.float32))
        class_label = int(self.label.iloc[int(self.filename[item][:-4])]['label'])
        return {'A': item_A, 'B': item_B,'fundus':img_C ,'name':self.filename[item], 'class_label':class_label}

    def __len__(self):
        return len(self.filename)

class My_dataset(Dataset):
    def __init__(self, root,noise_level,count = None,transforms_1=None,transforms_2=None, transforms_3=None, unaligned=False, type='train'):
        super().__init__()
        if type == 'all':
            self.data = pd.read_csv(f'{root}/extert/label.csv')[['image', 'label']]
        else:
            self.data = pd.read_csv(f'{root}/label.csv')[['image', 'label']]
        if type == 'train':
            self.idx = np.load(f'{root}/train.npy')
        elif type == 'val':
            self.idx = np.load(f'{root}/val.npy')
        elif type == 'all':
            self.idx = np.arange(len(self.data))
        else:
            self.idx = np.load(f'{root}/test.npy')
        self.transform1 = transforms.Compose(transforms_1)
        self.transform2 = transforms.Compose(transforms_2)
        self.transform3 = transforms.Compose(transforms_3)
        self.noise_level = noise_level
        self.type = type
        if type == 'all':
            self.root = f'{root}/extert'
        else:
            self.root = root

    def __getitem__(self, item):
        if self.noise_level == 0 and self.type == 'train':
            seed = np.random.randint(2147483647)  # make a seed with numpy generator
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            imageidx = self.data.loc[self.idx[item]]['image']
            label = int(self.data.loc[self.idx[item]]['label'])
            beforeimage = cv2.imread(os.path.join(f'{self.root}/before', imageidx), 0)
            afterimage = cv2.imread(os.path.join(f'{self.root}/after', imageidx), 0)
            beforeimage = (beforeimage - 127.5) / 127.5
            afterimage = (afterimage - 127.5) / 127.5
            beforeimage = self.transform2(beforeimage.astype(np.float32))
            afterimage = self.transform2(afterimage.astype(np.float32))
        else:
            imageidx = self.data.loc[self.idx[item]]['image']
            label =  int(self.data.loc[self.idx[item]]['label'])
            beforeimage = cv2.imread(os.path.join(f'{self.root}/before', imageidx),0)
            afterimage = cv2.imread(os.path.join(f'{self.root}/after', imageidx),0)
            beforeimage = (beforeimage - 127.5) / 127.5
            afterimage = (afterimage - 127.5) / 127.5
            beforeimage = self.transform1(beforeimage.astype(np.float32))
            afterimage = self.transform1(afterimage.astype(np.float32))
        return {'A': beforeimage, 'B': afterimage,'name':imageidx, 'class_label':label}

    def __len__(self):
        return len(self.idx)

# class My_dataset(Dataset):
#     def __init__(self, root,noise_level,count = None,transforms_1=None,transforms_2=None, transforms_3=None, unaligned=False, type='train'):
#         super().__init__()
#         file = open(f'{root}/label.txt', 'r')
#         self.data = file.read().splitlines()
#         if type == 'train':
#             self.idx = np.load(f'{root}/train_idx.npy')
#         elif type == 'val':
#             self.idx = np.load(f'{root}/val_idx.npy')
#         elif type == 'all':
#             self.idx = np.arange(len(self.data))
#         else:
#             self.idx = np.load(f'{root}/test_idx.npy')
#         self.transform1 = transforms.Compose(transforms_1)
#         self.transform2 = transforms.Compose(transforms_2)
#         self.transform3 = transforms.Compose(transforms_3)
#         self.noise_level = noise_level
#         self.type = type
#         self.root = f'{root}/image'
#
#     def __getitem__(self, item):
#         name1, name2, name3, name4, label, label1, label2 = self.data[self.idx[item]].split()
#         if name4[-1] == ',':
#             name4 = name4[:-1]
#
#         if self.noise_level == 0 and self.type == 'train':
#             seed = np.random.randint(2147483647)  # make a seed with numpy generator
#             torch.manual_seed(seed)
#             torch.cuda.manual_seed(seed)
#             beforeimage = cv2.imread(os.path.join(self.root, name1), 0)
#             afterimage = cv2.imread(os.path.join(self.root, name4), 0)
#             beforeimage = (beforeimage - 127.5) / 127.5
#             afterimage = (afterimage - 127.5) / 127.5
#             beforeimage = self.transform2(beforeimage.astype(np.float32))
#             afterimage = self.transform2(afterimage.astype(np.float32))
#         else:
#             beforeimage = cv2.imread(os.path.join(self.root, name1), 0)
#             afterimage = cv2.imread(os.path.join(self.root, name4), 0)
#             Cimage = cv2.imread(os.path.join(self.root, name3),0)
#             beforeimage = (beforeimage - 127.5) / 127.5
#             afterimage = (afterimage - 127.5) / 127.5
#             Cimage = (Cimage - 127.5)/127.5
#
#             Cimage = self.transform1(Cimage.astype(np.float32))
#             beforeimage = self.transform1(beforeimage.astype(np.float32))
#             afterimage = self.transform1(afterimage.astype(np.float32))
#         return {'A': beforeimage, 'B': afterimage, 'C':Cimage, 'name':name1, 'class_label':int(label2)}
#
#     def __len__(self):
#         return len(self.idx)

def sort_by_number(elem):
    underscore_positions = np.where(np.array(list(elem)) == '_')[0]
    return float(elem[underscore_positions[0]+1:-4])

class EyeDataset1(Dataset):
    def __init__(self, root,noise_level,count = None,transforms_1=None,transforms_2=None, unaligned=False, type='train'):
        self.transform1 = transforms.Compose(transforms_1)
        self.transform2 = transforms.Compose(transforms_2)
        self.unaligned = unaligned
        self.root = root
        self.type = type
        self.noise_level = noise_level
        self.bname = os.listdir(f'{root}/before')
        self.pname = os.listdir(f'{root}/after')
        if type == 'train':
            self.filename = np.load(f'{self.root}/train.npy', allow_pickle=True)
        elif type=='validation':
            self.filename = np.load(f'{self.root}/val.npy', allow_pickle=True)
        else:
            self.filename = np.load(f'{self.root}/test.npy', allow_pickle=True)
        self.bname = sorted(self.bname, key=sort_by_number)
        self.pname = sorted(self.pname, key=sort_by_number)

    def __getitem__(self, item):
        if self.noise_level == 0 and self.type=='train':
            # if noise =0, A and B make same transform
            seed = np.random.randint(2147483647) # make a seed with numpy generator
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            img = cv2.imread(os.path.join(f'{root}/before', self.bname[self.filename[item]]), 0)
            img = (img-127.5)/127.5
            item_A = self.transform2(img.astype(np.float32))
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            img = cv2.imread(os.path.join(f'{self.root}/after',self.pname[self.filename[item]]), 0)
            img = (img - 127.5) / 127.5
            item_B = self.transform2(img.astype(np.float32))
        else:
            # print(self.filename[item])
            img_A = cv2.imread(os.path.join(f'{self.root}/before',self.bname[self.filename[item]]), 0)
            img_B = cv2.imread(os.path.join(f'{self.root}/after',self.pname[self.filename[item]]), 0)
            img_A = (img_A - 127.5) / 127.5
            img_B = (img_B - 127.5) / 127.5
            item_A = self.transform1(img_A.astype(np.float32))
            item_B = self.transform1(img_B.astype(np.float32))
        class_label = sort_by_number(self.pname[self.filename[item]])
        eye = sort_by_number(self.bname[self.filename[item]])
        return {'A': item_A, 'B': item_B, 'name':self.filename[item], 'class_label':class_label, 'eye':eye}

    def __len__(self):
        return len(self.filename)