import random
from os.path import join as opj

import cv2
import numpy as np
from PIL import Image

import torch
from torchvision import transforms
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader

class Train_Dataset(Dataset):
    def __init__(self, df, transform=None, zipper_transform=None, metalnut_transform=None, toothbrush_transform=None, label_encoder=None, is_training=False):
        self.id = df['file_name'].values
        self.target = df['label'].values
        self.le = label_encoder
        self.transform = transform
        self.zipper_transform = zipper_transform
        self.metalnut_transform = metalnut_transform
        self.toothbrush_transform = toothbrush_transform
        self.is_training = is_training
        self.data_path = '../data/train/'
        
        self.metalnut = self.le.transform([label for label in self.le.classes_ if 'metal_nut' in label])
        self.toothbrush = self.le.transform([label for label in self.le.classes_ if 'toothbrush' in label])
        self.zipper = self.le.transform([label for label in self.le.classes_ if 'zipper' in label])

        print(f'Dataset size:{len(self.id)}')

    def __getitem__(self, idx):        
        img_path = opj(self.data_path, self.id[idx])
        image = cv2.imread(img_path).astype(np.float32)
        target = self.target[idx]
        
        if self.is_training and random.random() > 0.5:
            image, target = self.aug_data(image, target)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0

        if (self.metalnut_transform) and (target in self.metalnut):
            image_transform = self.metalnut_transform
        elif (self.toothbrush_transform) and (target in self.toothbrush):
            image_transform = self.toothbrush_transform
        elif (self.zipper_transform) and (target in self.zipper):
            image_transform = self.zipper_transform
        else:
            image_transform = self.transform

        #image = image_transform(Image.fromarray(image))
        image = image_transform(torch.from_numpy(image.transpose(2,0,1)))

        return image, target#np.array(target)

    def __len__(self):
        return len(self.id)

    def aug_data(self, image, target):
        if target == 52: #pill-good
            return self.aug_pill(image, target)
        elif target == 84: #zipper-good
            return self.aug_zipper(image, target)
        return image, target

    def aug_pill(self, image, target):           
        def get_alpha(abnormal, good='pill-good'):
            if good != 'pill-good':
                return random.uniform(0.4,0.6)
            else:
                if abnormal == 'pill-color' or abnormal == 'pill-pill_type':
                    return random.uniform(0.1,0.6)
                else:
                    return random.uniform(0.05,0.15)

        li = [label for label in self.le.classes_ if 'pill' in label]
        li.remove('pill-good')
        li.remove('pill-combined')
        li = list(self.le.transform(li))

        if random.random() > 0.14:
            indexes = self.id[np.isin(self.target, li)] # good, combined 제외한 인덱스 추출
            idx = np.random.choice(indexes, 1).item() # 랜덤하게 하나 추출 (ex. 10970.png)
            new_target = self.target[np.where(self.id == idx)][0] # long

            alpha = get_alpha(self.le.inverse_transform([new_target]))
            img_path = opj(self.data_path, idx)
            image2 = cv2.imread(img_path).astype(np.float32)

            new_image = image * alpha + image2 * (1-alpha)
        
        else: #Combined 생성
            li.remove(53) #pill_type
            l1, l2 = np.random.choice(li, 2, replace=False) # 중복 제외하고 5가지 중 2가지 추출
            alpha = get_alpha(self.le.inverse_transform([l1]), self.le.inverse_transform([l2]))

            id1 = np.random.choice(self.id[self.target==l1], 1).item()
            id2 = np.random.choice(self.id[self.target==l2], 1).item()

            img_path1 = opj(self.data_path, id1)
            img_path2 = opj(self.data_path, id2)

            image1 = cv2.imread(img_path1).astype(np.float32)
            image2 = cv2.imread(img_path2).astype(np.float32)

            new_image = image1 * alpha + image2 * (1-alpha)
            new_target = self.le.transform(['pill-combined'])

        if random.random() > 0.5:
            gamma = random.uniform(0.8, 1.3)
            new_image = np.clip(np.power(new_image / 255.0, gamma) * 255, 0, 255)

        return new_image, new_target.item()

    def aug_zipper(self, image, target):      
        def get_alpha(abnormal, good='zipper-good'):
            if good != 'zipper-good':
                return random.uniform(0.4,0.6)
            else:
                return random.uniform(0.05,0.2)   

        li = [label for label in self.le.classes_ if 'zipper' in label]
        li.remove('zipper-good')
        li.remove('zipper-combined')
        li = list(self.le.transform(li))
        
        if target == 84: #good이면서 6/7의 확률로 combined를 제외한 다른 비정상 생성
            indexes = self.id[np.isin(self.target, li)] # good, combined 제외한 인덱스 추출
            idx = np.random.choice(indexes, 1).item() # 랜덤하게 하나 추출 (ex. 10970.png)
            new_target = self.target[np.where(self.id == idx)][0] #string (ex. 67)

            alpha = get_alpha(self.le.inverse_transform([new_target]))
            img_path = opj(self.data_path, idx)
            image2 = cv2.imread(img_path).astype(np.float32)

            new_image = image * alpha + image2 * (1-alpha)
            
        else: #Combined 생성
            l1, l2 = np.random.choice(li, 2, replace=False) # 중복 제외하고 6가지 중 2가지 추출
            alpha = get_alpha(self.le.inverse_transform([l1]), self.le.inverse_transform([l2]))

            id1 = np.random.choice(self.id[self.target==l1], 1).item()
            id2 = np.random.choice(self.id[self.target==l2], 1).item()

            img_path1 = opj(self.data_path, id1)
            img_path2 = opj(self.data_path, id2)

            image1 = cv2.imread(img_path1).astype(np.float32)
            image2 = cv2.imread(img_path2).astype(np.float32)

            new_image = image1 * alpha + image2 * (1-alpha)
            new_target = self.le.transform(['zipper-combined'])
                        
        return new_image, new_target.item()

class Test_Dataset(Dataset):
    def __init__(self, df, transform=None):
        self.id = df['file_name'].values
        self.transform = transform

        print(f'Test Dataset size:{len(self.id)}')

    def __getitem__(self, idx):        
        image = np.array(Image.open(f'../data/test/{self.id[idx]}').convert('RGB'))

        if self.transform is not None:
            image = self.transform(Image.fromarray(image))

        return image

    def __len__(self):
        return len(self.id)

def get_loader(df, phase: str, batch_size, shuffle, num_workers, transform, zipper_transform, metalnut_transform, toothbrush_transform, label_encoder=None, is_training=False):

    if phase == 'test':
        dataset = Test_Dataset(df, transform)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    else:
        dataset = Train_Dataset(df, transform, zipper_transform, metalnut_transform, toothbrush_transform, label_encoder=label_encoder, is_training=is_training)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True, drop_last=False)

    return data_loader

class GammaTransform:
    """Rotate by one of the given angles."""

    def __init__(self,a,b):
        self.a = a
        self.b = b

    def __call__(self, x):
        gam = random.uniform(self.a,self.b)
        return TF.adjust_gamma(x, gam)
    
    
def get_train_augmentation(img_size, ver):
    if ver==1:
        transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                ])   
    elif ver==2:
        transform = transforms.Compose([
                transforms.RandomRotation(degrees=20),
                transforms.RandomAffine(degrees=(10,30)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.Resize((img_size, img_size)),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
            ])   
    elif ver==3:
        transform = transforms.Compose([
                transforms.RandomRotation(degrees=180),
                transforms.RandomAffine(degrees=(10,30)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.Resize((img_size, img_size)),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
            ])  
    elif ver==4:
        transform = transforms.Compose([
                transforms.RandomRotation(degrees=20),
                transforms.RandomAffine(degrees=(10,30)),
                transforms.Resize((img_size, img_size)),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
            ])   
    elif ver==5:
        transform = transforms.Compose([
                transforms.RandomRotation(degrees=180),
                transforms.RandomAffine(degrees=(10,30)),
                transforms.Resize((img_size, img_size)),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
            ]) 
    elif ver==6:
        transform = transforms.Compose([
                transforms.RandomRotation(degrees=180),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomAffine(30),
                transforms.ColorJitter(),
                transforms.RandomInvert(),
                transforms.Resize((img_size, img_size)),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
            ])    
    elif ver==7:
        transform = transforms.Compose([
                transforms.RandomRotation(degrees=180),
                transforms.RandomAffine(degrees=45),
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),    
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
            ])   
    elif ver==8:
        transform = transforms.Compose([
                transforms.RandomRotation(degrees=10),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomAffine(30),
                transforms.Resize((img_size, img_size)),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
            ])
    elif ver==9:
        transform = transforms.Compose([
                transforms.RandomRotation(degrees=(0,180)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomAffine(30),
                GammaTransform(0.6,1.0),
                transforms.Resize((img_size, img_size)),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
            ])       
    elif ver==10:
        transform = transforms.Compose([
                transforms.RandomRotation(degrees=(180)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomAffine(30),
                transforms.ColorJitter(),
                transforms.RandomInvert(),
                transforms.Resize((img_size, img_size)),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
            ]) 
    elif ver==11:
        transform = transforms.Compose([
                transforms.RandomRotation(degrees=20),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.Resize((img_size, img_size)),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
            ])      
    return transform

