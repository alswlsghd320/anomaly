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
│   ├── main.py
│   ├── trainer.py 
│   ├── dataset.py 
│   ├── model.py 
│   ├── loss.py
│   ├── single_gpu_inference.py
│   ├── image_model_list.txt
│   ├── requirement.txt

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
    (Download data to ./LG_Plant_Disease_Diagnosis/data/)
    ```
3. Unzip train, test data
    ```bash
    #./workspace
    unzip data.zip
    unzip train.zip
    unzip test.zip
    ```
4. Train `main.py`

5. Submit 
`./submission_xxx.csv`
