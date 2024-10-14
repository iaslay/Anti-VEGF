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