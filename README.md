# LADA
This repo contains codes for the following paper: 



*Jiaao Chen\*, Zhenghui Wang\*, Ran Tian, Zichao Yang, Diyi Yang*:  Local Additivity Based Data Augmentation for Semi-supervised NER. In Proceedings of The 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP'2020)

If you would like to refer to it, please cite the paper mentioned above. 


## Getting Started
These instructions will get you running the codes of LADA.

### Requirements
* Python 3.6 or higher
* Pytorch >= 1.4.0
* Pytorch_transformers (also known as transformers)
* Pandas, Numpy, Pickle, faiss, sentence-transformers



### Code Structure
```
├── code/
│   ├── BERT/
│   │   ├── back_translate.ipynb --> Jupyter Notebook for back translating the dataset
│   │   ├── bert_models.py --> Codes for LADA-based BERT models
│   │   ├── eval_utils.py --> Codes for evaluations
│   │   ├── knn.ipynb --> Jupyter Notebook for building the knn index file
│   │   ├── read_data.py --> Codes for data pre-processing
│   │   ├── train.py --> Codes for trianing BERT model
│   │   └── ...
│   ├── flair/
│   │   ├── train.py --> Codes for trianing flair model
│   │   ├── knn.ipynb --> Jupyter Notebook for building the knn index file
│   │   ├── flair/ --> the flair library
│   │   │   └── ...
│   │   ├── resources/
│   │   │   ├── docs/ --> flair library docs
│   │   │   ├── taggers/ --> save evaluation results for flair model
│   │   │   └── tasks/
│   │   │       └── conll_03/
│   │   │           ├── sent_id_knn_749.pkl --> knn index file
│   │   │           └── ... -> CoNLL-2003 dataset
│   │   └── ...
├── data/
│   └── conll2003/
│       ├── de.pkl -->Back translated training dataset with German as middle language
│       ├── labels.txt --> label index file
│       ├── sent_id_knn_700.pkl
│       └── ...  -> CoNLL-2003 dataset
├── eval/
│   └── conll2003/ --> save evaluation results for BERT model
└── README.md
```
## BERT models

### Downloading the data
Please download the CoNLL-2003 dataset and save under `./data/conll2003/` as `train.txt`, `dev.txt`, and `test.txt`.

### Pre-processing the data


We utilize [Fairseq](https://github.com/pytorch/fairseq) to perform back translation on the training dataset. Please refer to `./code/BERT/back_translate.ipynb` for details.

Here, we have put one example of back translated data, `de.pkl`, in `./data/conll2003/` . You can directly use it for CoNLL-2003 or generate your own back translated data following  `./code/BERT/back_translate.ipynb`.

We also provide the kNN index file for the first 700 training sentences (5%) `./data/conll2003/sent_id_knn_700.pkl`. You can directly use it for CoNLL-2003 or generate your own kNN index file following `./code/BERT/knn.ipynb`


### Training models
These section contains instructions for training models on CoNLL-2003 using 5% training data.

#### Training BERT+Intra-LADA model
```shell
python ./code/BERT/train.py --data-dir 'data/conll2003' --model-type 'bert' \
--model-name 'bert-base-multilingual-cased' --output-dir 'eval/conll2003' --gpu '0,1' \
--labels 'data/conll2003/labels.txt' --max-seq-length 164 --overwrite-output-dir \
--do-train --do-eval --do-predict --evaluate-during-training --batch-size 16 \
--num-train-epochs 20 --save-steps 750 --seed 1 --train-examples 700  --eval-batch-size 128 \
--pad-subtoken-with-real-label --eval-pad-subtoken-with-first-subtoken-only --label-sep-cls \
--mix-layers-set 8 9 10  --beta 1.5 --alpha 60  --mix-option --use-knn-train-data \
--num-knn-k 5 --knn-mix-ratio 0.5 --intra-mix-ratio 1 
```
#### Training BERT+Inter-LADA model
```shell
python ./code/BERT/train.py --data-dir 'data/conll2003' --model-type 'bert' \
--model-name 'bert-base-multilingual-cased' --output-dir 'eval/conll2003' --gpu '0,1' \
--labels 'data/conll2003/labels.txt' --max-seq-length 164 --overwrite-output-dir \
--do-train --do-eval --do-predict --evaluate-during-training --batch-size 16 \
--num-train-epochs 20 --save-steps 750 --seed 1 --train-examples 700  --eval-batch-size 128 \ 
--pad-subtoken-with-real-label --eval-pad-subtoken-with-first-subtoken-only --label-sep-cls \ 
--mix-layers-set 8 9 10  --beta 1.5 --alpha 60  --mix-option --use-knn-train-data \
--num-knn-k 5 --knn-mix-ratio 0.5 --intra-mix-ratio -1  

```
#### Training BERT+Semi-Intra-LADA model
```shell
python ./code/BERT/train.py --data-dir 'data/conll2003' --model-type 'bert' \
--model-name 'bert-base-multilingual-cased' --output-dir 'eval/conll2003' --gpu '0,1' \
--labels 'data/conll2003/labels.txt' --max-seq-length 164 --overwrite-output-dir \
--do-train --do-eval --do-predict --evaluate-during-training --batch-size 16 \
--num-train-epochs 20 --save-steps 750 --seed 1 --train-examples 700  --eval-batch-size 128 \
--pad-subtoken-with-real-label --eval-pad-subtoken-with-first-subtoken-only --label-sep-cls \
--mix-layers-set 8 9 10  --beta 1.5 --alpha 60  --mix-option --use-knn-train-data \
--num-knn-k 5 --knn-mix-ratio 0.5 --intra-mix-ratio 1 \
--u-batch-size 32 --semi --T 0.6 --sharp --weight 0.05 --semi-pkl-file 'de.pkl' \
--semi-num 10000 --semi-loss 'mse' --ignore-last-n-label 4  --warmup-semi --num-semi-iter 1 \
--semi-loss-method 'origin' 
```
#### Training BERT+Semi-Inter-LADA model
```shell
python ./code/BERT/train.py --data-dir 'data/conll2003' --model-type 'bert' \
--model-name 'bert-base-multilingual-cased' --output-dir 'eval/conll2003' --gpu '0,1' \
--labels 'data/conll2003/labels.txt' --max-seq-length 164 --overwrite-output-dir \
--do-train --do-eval --do-predict --evaluate-during-training --batch-size 16 \
--num-train-epochs 20 --save-steps 750 --seed 1 --train-examples 700  --eval-batch-size 128 \ 
--pad-subtoken-with-real-label --eval-pad-subtoken-with-first-subtoken-only --label-sep-cls \
--mix-layers-set 8 9 10  --beta 1.5 --alpha 60  --mix-option --use-knn-train-data \
--num-knn-k 5 --knn-mix-ratio 0.5 --intra-mix-ratio -1 \
--u-batch-size 32 --semi --T 0.6 --sharp --weight 0.05 --semi-pkl-file 'de.pkl' \
--semi-num 10000 --semi-loss 'mse' --ignore-last-n-label 4  --warmup-semi --num-semi-iter 1 \
--semi-loss-method 'origin' 

```


#### 
## flair models

[flair](https://github.com/flairNLP/flair) is a BiLSTM-CRF sequence labeling model, and we provide code for flair+Inter-LADA 

### Downloading the data
Please download the CoNLL-2003 dataset and save under `./code/flair/resources/tasks/conll_03/` as `eng.train`, `eng.testa` (dev), and `eng.testb` (test).

### Pre-processing the data

We also provide the kNN index file for the first 749 training sentences (5%, including the `-DOCSTART-` seperator) `./code/flair/resources/tasks/conll_03/sent_id_knn_749.pkl`. You can directly use it for CoNLL-2003 or generate your own kNN index file following `./code/flair/knn.ipynb`

### Training models
These section contains instructions for training models on CoNLL-2003 using 5% training data.

#### Training flair+Inter-LADA  model
```shell
CUDA_VISIBLE_DEVICES=1 python ./code/flair/train.py --use-knn-train-data --num-knn-k 5 \
--knn-mix-ratio 0.6 --train-examples 749 --mix-layer 2  --mix-option --alpha 60 --beta 1.5 \
--exp-save-name 'mix'  --mini-batch-size 64  --patience 10 --use-crf 
```


