# Computer Vision 이상치 탐지 알고리즘 경진대회

### Public score 3th 0.90875 | Private score 3th 0.90494

* 주최 : DACON
* 주관 : DACON
* [https://dacon.io/competitions/official/235870/overview/description](https://dacon.io/competitions/official/235894/overview/description)

### Overview
    - efficientnet-b1,b2를 이용하여 학습을 진행했습니다.
    - arcface loss, mixup 등 시도에서 별다른 성능 향상을 가져오지는 않았습니다.
    - 다른 팀들과는 다르게 efficientnet-b6,b7에 대해서 성능이 좋지 않아서 사용하지 않았습니다.
    - 따라서, 기존에 학습했던 모델들을 이용해서 앙상블 및 후처리에 대하여 고민을 했습니다.
    - 첫번째로 클래스 불균형으로 인한 good의 과한 예측을 피하고자 하였습니다.
        1. 기본적으로 good인 이미지들은 어떠한 기하학적 변형을 해도 매우 확실하게 good이라고 예측할거라 가정했습니다.
        2. 너무 과하게 good으로 예측하는 경우를 조금이라도 완화시키고자 각 모델들 output에 softmax를 취한 후 앙상블했습니다.
        3. 비정상의 경우를 bad로 통일한 후 모델을 학습한 결과를 이용하여 후처리를 진행했습니다.
    - 두번째로 위의 결과를 이용해도 헷갈려하는 클래스들(pill, zipper, toothbrush, transistor, capsule)에 대해 추가 학습을 진행했습니다.
        1. 한 개의 클래스에 대해서만 학습을 한 후 하드보팅 또는 단일 모델의 결과를 가지고 후처리를 진행했습니다. 

<br>

### Directory Structure
```
/workspace
├── data
│   ├── train
│   │    ├── 10001.png
│   │    ├── ...
│   │    └── 14276.png
│   ├── test
│   │    ├── 20001.png
│   │    ├── ...
│   │    └── 22153.png
│   │    
│   ├── train_df.csv
│   ├── test_df.csv
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

## Usage
- `make_df.ipynb`를 통해 state를 good과 bad로만 구분하는 `train_df_bad.csv`, one class만을 저장하는 `{class}_df.csv` 를 생성할 수 있습니다.
- `solution.ipynb` 에서 `test.py`와 Inference & Post-processing 과정에 관한 설명을 확인할 수 있습니다.

1. Install Library
    ```
    pip install -r requirements.txt
    pip install jupyter
    ```
2. Download data.zip from[ https://dacon.io/competitions/official/235870/data](https://dacon.io/competitions/official/235894/data) to data path.
    ```bash
    #./workspace
    mkdir data
    cd data
    (Download data to ./workspace/data/)
    unzip data.zip
    unzip train.zip
    unzip test.zip
    ```
3. Prepare data : You can run this section in this [ipynb](make_df.ipynb)

4. Training
   ```bash
   sh multi_train.sh
   ```
5. Test : You can run this section in [test.py](test.py)

6. Inference & Post-processing : You can run this section in [solution.ipynb](solution.ipynb)

