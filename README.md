# dacon-anomaly

### Private score 3th 0.90~~

* 주최 : DACON
* 주관 : DACON
* [https://dacon.io/competitions/official/235870/overview/description](https://dacon.io/competitions/official/235894/overview/description)

<br>

### Directory Structure
```
/workspace
├── data
│   ├── train
│   │    ├── 10001.png
│   │    ├── ...
│   │    └── 67678
│   ├── test
│   │    ├── 10000
│   │    ├── ...
│   │    └── 67677
│   │    
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
│
├── dacon-anomaly
│   ├── config.py
│   ├── dataloader.py
│   ├── hardvoting.py
│   ├── main.py
│   ├── make_df.ipynb
│   ├── multi_train.sh
│   ├── network.py
│   ├── prediction_ensemble.ipynb
│   ├── test.py
│   ├── trainer.py
│   ├── files
│   │    ├── effb4_bad_5fold.npy
│   │    ├── softmax_142.npy
│   │    ├── softmax_156.npy
│   │    ├── softmax_pillzip_266.npy
│   │    ├── softmax_pillzip_274.npy
│   │    ├── softmax_sy_123.npy
│   │    ├── softmax_sy_133.npy
│   │    ├── softmax_sy_266.npy
│   │    ├── softmax_sy_272.npy
│   ├── utils
│   │    ├── __init__.py
│   │    ├── image_utils.py
│   │    ├── logger_utils.py
│   │    ├── scheduler_utils.py
```
<br>

## Jupyter Notebook Usage
1. Install Library
    ```
    pip3 install -r requirement.txt
    pip3 install jupyter
    ```


2. Download data.zip from[ https://dacon.io/competitions/official/235870/data](https://dacon.io/competitions/official/235894/data) to data path.
    ```bash
    #./workspace
    mkdir data
    cd data
    (Download data to ./workspace/data/)
    ```
3. Unzip train, test data
    ```bash
    #./workspace
    unzip data.zip
    unzip train.zip
    unzip test.zip
    ```
4. Train `multi_train.sh`
   sh multi_train.sh

5. Submit 
`submission.csv`
