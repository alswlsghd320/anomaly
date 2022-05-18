import os
import pprint
import random

import neptune
import numpy as np

import torch

from trainer import Trainer
from config import getConfig

import warnings
warnings.filterwarnings('ignore')

args = getConfig()

def main(args):
    print('<---- Training Params ---->')
    pprint.pprint(args)

    # Random Seed
    seed = args.seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

    save_path = os.path.join(args.model_path, (args.exp_num).zfill(3))

    if args.logging:
        api = ""
        neptune.init("noname/dacon-anomaly", api_token=api)
        temp = neptune.create_experiment(name=args.experiment, params=vars(args))
        experiment_num = str(temp).split('-')[-1][:-1] 
        neptune.append_tag(args.tag)
        save_path = os.path.join(args.model_path, experiment_num.zfill(3))

    # Create model directory
    os.makedirs(save_path, exist_ok=True)
    Trainer(args, save_path)

if __name__ == '__main__':
    main(args)
