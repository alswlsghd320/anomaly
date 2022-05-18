############################################### bad ###############################################
# bad0
python main.py --batch_size=16 --drop_path_rate=0.2 --encoder_name=efficientnet_b4 \
			   --img_size=384 --aug_ver=11 --metalnut_aug=4 --num_classes=30 \
               --scheduler=cycle --weight_decay=1e-3 --initial_lr=3e-4 --Kfold=5 --fold=0 \
               --tag=re_bad0 --max_lr=1e-3 --epochs=100 --warm_epoch=3 --logging=True --file_name=train_df_bad.csv 
# bad1
python main.py --batch_size=16 --drop_path_rate=0.2 --encoder_name=efficientnet_b4 \
			   --img_size=384 --aug_ver=11 --metalnut_aug=4 --num_classes=30 \
               --scheduler=cycle --weight_decay=1e-3 --initial_lr=3e-4 --Kfold=5 --fold=1 \
               --tag=re_bad1 --max_lr=1e-3 --epochs=100 --warm_epoch=3 --logging=True --file_name=train_df_bad.csv
# bad2
python main.py --batch_size=16 --drop_path_rate=0.2 --encoder_name=efficientnet_b4 \
			   --img_size=384 --aug_ver=11 --metalnut_aug=4 --num_classes=30 \
               --scheduler=cycle --weight_decay=1e-3 --initial_lr=3e-4 --Kfold=5 --fold=2 \
               --tag=re_bad2 --max_lr=1e-3 --epochs=100 --warm_epoch=3 --logging=True --file_name=train_df_bad.csv
# bad3
python main.py --batch_size=16 --drop_path_rate=0.2 --encoder_name=efficientnet_b4 \
			   --img_size=384 --aug_ver=11 --metalnut_aug=4 --num_classes=30 \
               --scheduler=cycle --weight_decay=1e-3 --initial_lr=3e-4 --Kfold=5 --fold=3 \
               --tag=re_bad3 --max_lr=1e-3 --epochs=100 --warm_epoch=3 --logging=True --file_name=train_df_bad.csv
# bad4
python main.py --batch_size=16 --drop_path_rate=0.2 --encoder_name=efficientnet_b4 \
			   --img_size=384 --aug_ver=11 --metalnut_aug=4 --num_classes=30 \
               --scheduler=cycle --weight_decay=1e-3 --initial_lr=3e-4 --Kfold=5 --fold=4 \
               --tag=re_bad4 --max_lr=1e-3 --epochs=100 --warm_epoch=3 --logging=True --file_name=train_df_bad.csv                              


############################################# 내가누구게 #############################################
# 148
python main.py --batch_size=32 --drop_path_rate=0.2 --encoder_name=efficientnet_b1 \
			   --img_size=384 --aug_ver=11 --metalnut_aug=4 \
               --scheduler=cycle --weight_decay=1e-3 --initial_lr=3e-4 --Kfold=5 --fold=0 \
               --tag=re_exp148 --max_lr=1e-3 --epochs=100 --warm_epoch=3 --logging=True

# 156
python main.py --batch_size=32 --drop_path_rate=0.2 --encoder_name=efficientnet_b1 \
			   --img_size=384 --aug_ver=11 --metalnut_aug=4 \
               --scheduler=cycle -weight_decay=1e-3 --initial_lr=3e-4 --Kfold=5 --fold=4 \
               --tag=re_exp156 --max_lr=1e-3 --epochs=100 --warm_epoch=3 --logging=True

# pillzip 266
python main.py --batch_size=32 --drop_path_rate=0.2 --encoder_name=efficientnet_b1 \
			   --img_size=384 --aug_ver=3 --metalnut_aug=4 \
               --scheduler=cycle --weight_decay=1e-3 --initial_lr=3e-4 --Kfold=5 --fold=0 \
               --tag=re_exp266 --max_lr=1e-3 --epochs=100 --warm_epoch=3 --logging=True --use_aug=True

# pillzip 274
python main.py --batch_size=32 --drop_path_rate=0.2 --encoder_name=efficientnet_b1 \
			   --img_size=384 --aug_ver=3 --metalnut_aug=4 \
               --scheduler=cycle -weight_decay=1e-3 --initial_lr=3e-4 --Kfold=5 --fold=4 \
               --tag=re_exp274 --max_lr=1e-3 --epochs=100 --warm_epoch=3 --logging=True --use_aug=True

############################################### thdus ###############################################
# 123
python main.py --batch_size=32 --drop_path_rate=0.2 --encoder_name=efficientnet_b2 \
			   --img_size=512 --aug_ver=2 --metalnut_aug=4 \
               --scheduler=cycle --weight_decay=1e-3 --initial_lr=3e-4 --Kfold=5 --fold=0 \
               --tag=re_exp123 --max_lr=1e-3 --epochs=100 --warm_epoch=3 --logging=True

# # 133
python main.py --batch_size=32 --drop_path_rate=0.2 --encoder_name=efficientnet_b2 \
			   --img_size=512 --aug_ver=2 --metalnut_aug=4 \
               --scheduler=cycle -weight_decay=1e-3 --initial_lr=3e-4 --Kfold=5 --fold=2 \
               --tag=re_exp133 --max_lr=1e-3 --epochs=100 --warm_epoch=3 --logging=True

# exp123_tooth_zipper 
python main.py --batch_size=20 --drop_path_rate=0.2 --encoder_name=efficientnet_b2 \
			   --img_size=512 --aug_ver=2 --metalnut_aug=4 --zipper_aug=6 --toothbrush_aug=6 \
               --scheduler=cycle --weight_decay=1e-3 --initial_lr=3e-4 --Kfold=5 --fold=0 \
               --tag=re_exp266 --max_lr=1e-3 --epochs=100 --warm_epoch=3 --logging=True

# exp148_tooth_zipper 
python main.py --batch_size=20 --drop_path_rate=0.2 --encoder_name=efficientnet_b2 \
			   --img_size=512 --aug_ver=3 --metalnut_aug=5 --zipper_aug=6 --toothbrush_aug=6 \
               --scheduler=cycle --weight_decay=1e-3 --initial_lr=3e-4 --Kfold=5 --fold=2 \
               --tag=re_exp272 --max_lr=1e-3 --epochs=100 --warm_epoch=3 --logging=True

############################################### dada ###############################################
# toothbrush model 1 294
python main.py --batch_size=16 --drop_path_rate=0.2 --encoder_name=efficientnet_b1 \
            --img_size=512 --aug_ver=10 --scheduler=cycle --Kfold=5 --fold=0\
               --weight_decay=1e-3 --initial_lr=3e-4 --patience=50 \
               --max_lr=1e-3 --epochs=150 --warm_epoch=3 --logging=True --file_name=toothbrush_df.csv 

# toothbrush model 2 295
python main.py --batch_size=16 --drop_path_rate=0.2 --encoder_name=efficientnet_b1 \
            --img_size=512 --aug_ver=9 --scheduler=cycle --Kfold=5 --fold=0\
               --weight_decay=1e-3 --initial_lr=3e-4 --patience=50 \
               --max_lr=1e-3 --epochs=150 --warm_epoch=3 --logging=True --file_name=toothbrush_df.csv 

# toothbrush model 3 296
python main.py --batch_size=16 --drop_path_rate=0.2 --encoder_name=efficientnet_b1 \
            --img_size=512 --aug_ver=8 --scheduler=cycle --Kfold=5 --fold=0\
               --weight_decay=1e-3 --initial_lr=3e-4 --patience=50 \
               --max_lr=1e-3 --epochs=150 --warm_epoch=3 --logging=True --file_name=toothbrush_df.csv 

# zipper model 1 297
python main.py --batch_size=16 --drop_path_rate=0.2 --encoder_name=efficientnet_b1 \
            --img_size=512 --aug_ver=10 --scheduler=cycle --Kfold=5 --fold=0\
               --weight_decay=1e-3 --initial_lr=3e-4 --patience=50 \
               --max_lr=1e-3 --epochs=150 --warm_epoch=3 --logging=True --file_name=zipper_df.csv 

# zipper model 2 298
python main.py --batch_size=16 --drop_path_rate=0.2 --encoder_name=efficientnet_b1 \
            --img_size=512 --aug_ver=8 --scheduler=cycle --Kfold=5 --fold=0\
               --weight_decay=1e-3 --initial_lr=3e-4 --patience=50 \
               --max_lr=1e-3 --epochs=150 --warm_epoch=3 --logging=True --file_name=zipper_df.csv 

# zipper model 3 299
python main.py --batch_size=16 --drop_path_rate=0.2 --encoder_name=efficientnet_b1 \
            --img_size=512 --aug_ver=9 --scheduler=cycle --Kfold=5 --fold=0\
               --weight_decay=1e-3 --initial_lr=3e-4 --patience=50 \
               --max_lr=1e-3 --epochs=150 --warm_epoch=3 --logging=True --file_name=zipper_df.csv 
