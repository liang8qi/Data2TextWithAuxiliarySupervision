# Data2TextWithAuxiliarySupervision


This repository provides the implementation for our paper "Improving Encoder by Auxiliary Supervision Tasks for Table-to-Text Generation" **ACL2021 main conference**. This code is based on https://github.com/ernestgong/data2text-three-dimensions/.

We provide the scrips (`preprocess.sh`, `train.sh`, and `translate.sh`) to preprocess the dataset, train models, and test. **Please refer to these scripts for more details about parameters setting**. The ouputs of our model are saved at `./our_results`.

## Citations

Please kindly cite this work if it helps your research:

```
@inproceedings{li-etal-2021-improving-encoder,
    title = "Improving Encoder by Auxiliary Supervision Tasks for Table-to-Text Generation",
    author = "Li, Liang  and
      Ma, Can  and
      Yue, Yinliang  and
      Hu, Dayong",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.acl-long.466",
    doi = "10.18653/v1/2021.acl-long.466",
    pages = "5979--5989",
}
```



## 1. Requirements

All dependencies can be installed via:

```
pip install -r 3.7.requirements.txt
```

Note that `3.7.requirements.txt` contains unnecessary packages and Python version is **3.7**.

## 2. Dataset

Please refer to https://github.com/ernestgong/data2text-three-dimensions/ and https://github.com/wanghm92/rw_fg to obtain ROTOWIRE and RW-FG datasets. And we provide the necessary config files in `./dataset`.

## 3. Preprocess

The following command will preprocess the data:

```
bash preprocess.sh
```

## 4. Train

The following command will train the model:

```
bash train.sh
```

## 5. Translate

The following command will generate on development or test datasets given trained model:

```
bash translate.sh
```

## 6. Evaluate

To obtain extractive evaluation metrics, please refer to https://github.com/ratishsp/data2text-1 and https://github.com/wanghm92/rw_fg for details.

The following command will compute BLEU score:

```
perl ref/multi-bleu.perl ref/test.txt < ./our_results/model_pred_rw_test.txt
```

