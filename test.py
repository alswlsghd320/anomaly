# %%
from os.path import join as opj

import ttach as tta
import pandas as pd
from tqdm import tqdm
from PIL import ImageFile
from easydict import EasyDict
from sklearn.preprocessing import LabelEncoder

from torch.cuda.amp import autocast

from dataloader import *
from network import *
# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args = EasyDict({'encoder_name':'efficientnet_b3',
                 'drop_path_rate':0.2,
                 'use_weight_norm':None,
                 'use_arcface':None,
                 'embedding_dim':512,
                 'use_mixup':None
                })

sub = pd.read_csv('submission/sample_submission.csv')   
df_train = pd.read_csv('train_df.csv')  
df_test = pd.read_csv('test_df.csv')  

le = LabelEncoder()
df_train['label'] = le.fit_transform(df_train['label'])

transform = get_train_augmentation(img_size=512, ver=1)
test_dataset = Test_Dataset(df_test, transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)

#%%
def predict(args, test_loader, model_path):
    _transforms = tta.Compose([
        tta.Rotate90(angles=[0, 90, 180, 270]),
    ])
    model = Network(args, is_training=False).to(device)
    model.load_state_dict(torch.load(opj(model_path, 'best_model.pth'))['state_dict'])
    #model.load_state_dict(torch.load(model_path))
    model = tta.ClassificationTTAWrapper(model, _transforms).to(device)
    model.eval()

    output = []
    with torch.no_grad():
        with autocast():
            for batch in tqdm(test_loader):
                images = torch.tensor(batch, dtype = torch.float32, device = device).clone().detach()
                preds = model(images)
                output.extend(torch.tensor(torch.argmax(preds, dim=1), dtype=torch.int32).detach().cpu().numpy())
    
    return le.inverse_transform(output)
#%%
def predict_ensemble(args, test_loader, exps):
    _transforms = tta.Compose([
        tta.Rotate90(angles=[0, 90, 180, 270]),
    ])
    
    model = [Network(args, is_training=False).to(device) for _ in range(len(exps))]
    
    for i, exp in enumerate(exps):
        model_path = opj(f'results/{exp.zfill(3)}', 'best_model.pth')
        model[i].load_state_dict(torch.load(model_path, map_location=device)['state_dict'])
        model[i] = tta.ClassificationTTAWrapper(model[i], _transforms).to(device)
        model[i].eval()

    output = []
    with torch.no_grad():
        with autocast():
            for batch in tqdm(test_loader):
                images = torch.tensor(batch, dtype = torch.float32, device = device).clone().detach()
                result = 0
                for m in model:
                    preds = m(images)
                    result += torch.softmax(preds, dim=-1)
                output.extend(torch.tensor(torch.argmax(result, dim=1), dtype=torch.int32).detach().cpu().numpy())    

    return le.inverse_transform(output)

# %%
def predict_softmax(args, test_loader, model_path):
    _transforms = tta.Compose([
        tta.Rotate90(angles=[0, 90, 180, 270]),
    ])
    model = Network(args, is_training=False).to(device)
    model.load_state_dict(torch.load(opj(model_path, 'best_model.pth'))['state_dict'])
    model = tta.ClassificationTTAWrapper(model, _transforms).to(device)
    model.eval()

    output = []
    with torch.no_grad():
        with autocast():
            for batch in tqdm(test_loader):
                images = torch.tensor(batch, dtype=torch.float32, device=device).clone().detach()
                preds = model(images)
                output.extend(torch.softmax(preds, dim=1).detach().cpu().numpy())

    return output
# %%

model_path = './' #'.' #'./results/034'

predicts = predict(args, test_loader, model_path)
#predicts = predict_ensemble2(args, test_loader, exps=None)
results = le.inverse_transform(np.argmax(predicts, axis=1))
sub['label']=results
sub.to_csv(opj(model_path, 'tmp1.csv'), index=False)
# %%
