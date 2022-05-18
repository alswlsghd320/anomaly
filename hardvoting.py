# %%
from matplotlib.pyplot import axis
import os
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from tqdm import tqdm
from glob import glob
from easydict import EasyDict
from torch.cuda.amp import autocast, grad_scaler
from natsort import natsorted
from dataloader import *
from network import *
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from PIL import Image, ImageFile

device = torch.device('cpu')

#%%
"""
후처리를 진행하기 위해 submission파일과 train파일 test파일을 모두 불러옵니다. 

"""
df_sub = pd.read_csv('./best_ensemble2_not_toothzipper.csv')
df_train = pd.read_csv('../data/train_df.csv')
df_test = pd.read_csv('../data/test_df.csv') 
# %%
args = EasyDict({'encoder_name':'efficientnet_b1',
                 'drop_path_rate':0.2,
                 'use_weight_norm':None,
                })

#%%
def predict(args, le, type, test_loader, model_path):
    model = Network(args).to(device)
    model.load_state_dict(torch.load(opj(model_path, f'{type}_model.pth'))['state_dict'])
    model.eval()
    output = []
    with torch.no_grad():
        with autocast():
            for batch in tqdm(test_loader):
                images = torch.tensor(batch, dtype = torch.float32, device = device)
                preds = model(images)
                output.extend(torch.tensor(torch.argmax(preds, dim=1), dtype=torch.int32).cpu().numpy())

    return le.inverse_transform(output)
#%%
"""
총 3개의 모델의 예측값으로 hard voting 하여
새로운 라벨로 바꾸어줍니다. 
"""
def hardVoting(args, le, df_sub, nums, idxLst, test_loader):
    df = df_sub[df_sub['index'].isin(idxLst)]
    
    for num in nums:
        model_path = f'./results/{num}/'
        pred = predict(args, le, 'best', test_loader, model_path)
        df[f'pred_{num}'] = pred
    
    for i in range(len(df)):
        label_pred_list = [df.iloc[i,1],df.iloc[i,2],df.iloc[i,3],df.iloc[i,4]]
        newlabel = max(label_pred_list, key=label_pred_list.count)
        
        df_sub.loc[df.iloc[i]['index'],'label'] = newlabel
        
    return df_sub

#%%
"""
특정 클래스를 후처리하는 함수입니다. 

"""
def Post_Processing(cls,nums, df_sub,df_train,df_test,args):
    

    idxLst = [df_sub.iloc[idx]['index'] for idx in range(len(df_sub)) if f'{cls}' in df_sub.iloc[idx]['label'] ]

    df_cls = df_train[df_train['class']==f'{cls}']
    le = LabelEncoder()
    df_cls['label'] = le.fit_transform(df_cls['label'])
    
    df_test_cls = df_test[df_test['index'].isin(idxLst)]
    transform = get_train_augmentation(img_size=512, ver=1)
    test_dataset = Test_Dataset(df_test_cls, transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    df_new_sub = hardVoting(args, le, df_sub, nums, idxLst, test_loader)
    
    return df_new_sub


#%%
# best_ensemble2.csv


"""
학습완료 후에 validation set을 통해 잘 구분하지 못헀던 class를 살펴보았습니다. 
toothbrush, transistor, capsule, pill, zipper의 후처리를 진행했으나
toothbrush, zipper만이 성능 향상을 보여 두 가지 class만 후처리를 진행했습니다. 
"""
df_sub = Post_Processing('toothbrush',[294,295,296], df_sub,df_train,df_test,args)
df_sub = Post_Processing('zipper',[297,298,299], df_sub,df_train,df_test,args)
df_sub.to_csv('./best_ensemble2.csv',index=False)


# %%
# best_ensemble2_zipper_tooth.csv

"""
zipper는 hard voting을 하고 toothbrush는 단일 모델의 예측값으로 라벨을 바꾸어주는 후처리를 진행했습니다. 
"""
df_sub = Post_Processing('toothbrush',[294], df_sub,df_train,df_test,args)
df_sub = Post_Processing('zipper',[297,298,299], df_sub,df_train,df_test,args)
df_sub.to_csv('./best_ensemble2_zipper_tooth.csv',index=False)
