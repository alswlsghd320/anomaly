import os
import time
import logging
from os.path import join as opj

import neptune
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import f1_score
from ptflops import get_model_complexity_info
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold

import torch.nn as nn
import torch_optimizer as optim
from torch.cuda.amp import autocast, grad_scaler

import utils
from dataloader import *
from network import *

import warnings
warnings.filterwarnings('ignore')

class Trainer():
    def __init__(self, args, save_path):
        '''
        args: arguments
        save_path: Model 가중치 저장 경로
        '''
        super(Trainer, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Logging
        log_file = os.path.join(save_path, 'log.log')
        self.logger = utils.get_root_logger(logger_name='IR', log_level=logging.INFO, log_file=log_file)
        self.logger.info(args)
        self.logger.info(args.tag)

        # Train, Valid Set load
        ########################################################################################################################################################
        df_train = pd.read_csv(opj(args.data_path, args.file_name))
        le = LabelEncoder()
        df_train['label'] = le.fit_transform(df_train['label'])

        # Split Fold
        kf = StratifiedKFold(n_splits=args.Kfold)
        for fold, (_, val_idx) in enumerate(kf.split(df_train, y=df_train['label'])):
            df_train.loc[val_idx, 'fold'] = fold
        df_val = df_train[df_train['fold'] == args.fold].reset_index(drop=True)
        df_train = df_train[df_train['fold'] != args.fold].reset_index(drop=True)

        ########################################################################################################################################################
        # Augmentation
        self.train_transform = get_train_augmentation(img_size=args.img_size, ver=args.aug_ver)
        self.zipper_transform = get_train_augmentation(img_size=args.img_size, ver=args.zipper_aug) if args.zipper_aug != None else None
        self.metalnut_transform = get_train_augmentation(img_size=args.img_size, ver=args.metalnut_aug) if args.metalnut_aug != None else None
        self.toothbrush_transform = get_train_augmentation(img_size=args.img_size, ver=args.toothbrush_aug) if args.toothbrush_aug != None else None
        self.test_transform = get_train_augmentation(img_size=args.img_size, ver=1)

        # TrainLoader
        self.train_loader = get_loader(df_train, phase='train', batch_size=args.batch_size, shuffle=True,
                                       num_workers=args.num_workers, transform=self.train_transform, zipper_transform=self.zipper_transform, 
                                       metalnut_transform=self.metalnut_transform, toothbrush_transform=self.toothbrush_transform, label_encoder=le, is_training=args.use_aug)
        self.val_loader = get_loader(df_val, phase='train', batch_size=args.batch_size, shuffle=False,
                                       num_workers=args.num_workers, transform=self.test_transform, zipper_transform=self.test_transform, 
                                       metalnut_transform=self.test_transform, toothbrush_transform=self.test_transform, label_encoder=le, is_training=False)

        # Network
        self.model = Network(args).to(self.device)
        macs, params = get_model_complexity_info(self.model, (3, args.img_size, args.img_size), as_strings=True,
                                                 print_per_layer_stat=False, verbose=False)
        self.logger.info('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        self.logger.info('{:<30}  {:<8}'.format('Number of parameters: ', params))

        # Loss
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer & Scheduler
        self.optimizer = optim.Lamb(self.model.parameters(), lr=args.initial_lr, weight_decay=args.weight_decay)
        
        iter_per_epoch = len(self.train_loader)
        self.warmup_scheduler = utils.WarmUpLR(self.optimizer, iter_per_epoch * args.warm_epoch)

        if args.scheduler == 'step':
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=args.milestone, gamma=args.lr_factor, verbose=True)
        elif args.scheduler == 'cos':
            tmax = args.tmax # half-cycle 
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max = tmax, eta_min=args.min_lr, verbose=True)
        elif args.scheduler == 'cycle':
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=args.max_lr, steps_per_epoch=iter_per_epoch, epochs=args.epochs)

        load_epoch=0
        if args.re_training_exp is not None:
            pth_files = torch.load(f'./results/{args.re_training_exp}/best_model.pth')
            load_epoch = pth_files['epoch']
            self.model.load_state_dict(pth_files['state_dict'])
            self.optimizer.load_state_dict(pth_files['optimizer'])

            sch_dict = pth_files['scheduler']
            sch_dict['total_steps'] = sch_dict['total_steps'] + args.epochs * iter_per_epoch
            self.scheduler.load_state_dict(sch_dict)

            print(f'Start {load_epoch+1} Epoch Re-training')
            for i in range(args.warm_epoch+1, load_epoch+1):
                self.scheduler.step()

        # if args.multi_gpu:
        #     self.model = nn.DataParallel(self.model).to(self.device)

        # Train / Validate
        best_loss = np.inf
        best_acc = 0
        best_f1 = 0
        best_epoch = 0
        early_stopping = 0
        start = time.time()
        for epoch in range(load_epoch+1, args.epochs+1):
            self.epoch = epoch

            if args.scheduler == 'cos':
                if epoch > args.warm_epoch:
                    self.scheduler.step()

            # Training
            train_loss, train_acc, train_f1 = self.training(args)

            # Model weight in Multi_GPU or Single GPU
            state_dict = self.model.state_dict()

            # Validation
            val_loss, val_acc, val_f1 = self.validate(phase='val')

            if args.logging == True:
                neptune.log_metric('Train loss', train_loss)
                neptune.log_metric('Train acc', train_acc)
                neptune.log_metric('Train f1', train_f1)

                neptune.log_metric('val loss', val_loss)
                neptune.log_metric('val acc', val_acc)
                neptune.log_metric('val f1', val_f1)

            # Save models
            if val_loss < best_loss:
                early_stopping = 0
                best_epoch = epoch
                best_loss = val_loss
                best_acc = val_acc
                best_f1 = val_f1

                torch.save({'epoch':epoch,
                            'state_dict':state_dict,
                            'optimizer': self.optimizer.state_dict(),
                            'scheduler': self.scheduler.state_dict(),
                    }, os.path.join(save_path, 'best_model.pth'))
                self.logger.info(f'-----------------SAVE:{best_epoch}epoch----------------')
            else:
                early_stopping += 1

            # Early Stopping
            if early_stopping == args.patience:
                break

            if epoch == args.epochs:
                torch.save({'epoch':epoch,
                            'state_dict':state_dict,
                            'optimizer': self.optimizer.state_dict(),
                            'scheduler': self.scheduler.state_dict(),
                    }, os.path.join(save_path, 'last_model.pth'))
                self.logger.info('-----------------SAVE: last epoch----------------')

        self.logger.info(f'\nBest Val Epoch:{best_epoch} | Val Loss:{best_loss:.4f} | Val Acc:{best_acc:.4f} | Val F1:{best_f1:.4f}')
        end = time.time()
        self.logger.info(f'Total Process time:{(end - start) / 60:.3f}Minute')
        neptune.stop()

    # Training
    def training(self, args):
        self.model.train()
        train_loss = utils.AvgMeter()
        train_acc = 0
        preds_list = []
        targets_list = []

        scaler = grad_scaler.GradScaler()
        for i, (images, targets) in enumerate(tqdm(self.train_loader)):
            images = torch.tensor(images, device=self.device, dtype=torch.float32)
            targets = torch.tensor(targets, device=self.device, dtype=torch.long)
            
            if self.epoch <= args.warm_epoch:
                self.warmup_scheduler.step()

            self.model.zero_grad(set_to_none=True)
            if args.amp:
                with autocast():
                    preds = self.model(images)
                    loss = self.criterion(preds, targets)
                scaler.scale(loss).backward()

                # Gradient Clipping
                if args.clipping is not None:
                    scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.clipping)

                scaler.step(self.optimizer)
                scaler.update()

            else:
                preds = self.model(images)
                loss = self.criterion(preds, targets)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), args.clipping)
                self.optimizer.step()

            if args.scheduler == 'cycle':
                if self.epoch > args.warm_epoch:
                    self.scheduler.step()

            # Metric
            train_acc += (preds.argmax(dim=1) == targets).sum().item()
            preds_list.extend(preds.argmax(dim=1).cpu().detach().numpy())
            targets_list.extend(targets.cpu().detach().numpy())
            # log
            train_loss.update(loss.item(), n=images.size(0))

        train_acc /= len(self.train_loader.dataset)
        train_f1 = f1_score(np.array(targets_list), np.array(preds_list), average='macro')

        self.logger.info(f'Epoch:[{self.epoch:03d}/{args.epochs:03d}]')
        self.logger.info(f'Train Loss:{train_loss.avg:.3f} | Acc:{train_acc:.4f} | F1:{train_f1:.4f}')

        return train_loss.avg, train_acc, train_f1
            
    # Validation or Dev
    def validate(self, phase='val'):
        self.model.eval()
        with torch.no_grad():
            val_loss = utils.AvgMeter()
            val_acc = 0
            preds_list = []
            targets_list = []

            for images, targets in self.val_loader:
                images = torch.tensor(images, device=self.device, dtype=torch.float32)
                targets = torch.tensor(targets, device=self.device, dtype=torch.long)

                preds = self.model(images)
                loss = self.criterion(preds, targets)

                # Metric
                val_acc += (preds.argmax(dim=1) == targets).sum().item()
                preds_list.extend(preds.argmax(dim=1).cpu().detach().numpy())
                targets_list.extend(targets.cpu().detach().numpy())
                # log
                val_loss.update(loss.item(), n=images.size(0))
            val_acc /= len(self.val_loader.dataset)
            val_f1 = f1_score(np.array(targets_list), np.array(preds_list), average='macro')

            self.logger.info(f'{phase} Loss:{val_loss.avg:.3f} | Acc:{val_acc:.4f} | F1:{val_f1:.4f}')
            
        return val_loss.avg, val_acc, val_f1

