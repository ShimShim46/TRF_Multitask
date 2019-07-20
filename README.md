Text Categorization by Learning Predominant Sense of Words as Auxiliary Task
==
There are five models:

* XML-CNN ([Liu+ '17](http://nyc.lti.cs.cmu.edu/yiming/Publications/jliu-sigir17.pdf)) : XML-CNN proposed by Liu'17 et al. 
* TRF-Single: A text categorization model based on the transformer encoder but without domain-specific sense prediction.
* TRF-Multi: A text categorization model based on the transformer encoder and is trained to simultaneously categorize texts and predicts a predominant sense for each word.
* TRF-Delay-Multi: A text categorization model to start learning predominant sense model at first until the stable, and after that it adapts text categorization simultaneously.
* TRF-Sequential: A text categorization model with fully separated training and TRF-Multi with fully simultaneously training.


### Feature of each model

|      Feature\Model     | XML-CNN | TRF-Single | TRF-Multi |                        TRF-Delay-Multi                       |        TRF-Sequential       |
|:--------------------:|:-------:|:----------:|:---------:|:------------------------------------------------------------:|:---------------------------:|
| Convolution?         |    ✔    |            |           |                                                              |                             |
| Single-Task?         |    ✔    |      ✔     |           |                                                              | ✔(To learn predominant sense model and text categorization separately) |
|      Multi-Task?     |         |            |     ✔     | ✔(To learn predominant sense model at first until the stable, and after that it adapts text categorization simultaneously) |                             |
| Transformer Encoder? |         |      ✔     |     ✔     |                               ✔                              |              ✔              |

## Requirements
In order to run the code, I recommend the following environment.
* Python 3.5.4 or higher.
* Chainer 4.0.0 or higher. ([chainer](http://chainer.org/))
* CuPy 4.0.0 or higher. ([cupy](https://cupy.chainer.org/))
* Optuna 0.8.0 or higher.  ([optuna](https://optuna.org/))

Requirements 
* The code requires GPU environment. Please see requirements.txt to run my code.

## Installation
* Download code from **clone or download**
* Install the requirements: requirements.txt
* You can also use Python data science platform, Anaconda([anaconda](https://www.anaconda.com/enterprise/)) as follows:
    1. Download Anaconda from (https://www.anaconda.com/download/)
        * Example: Anaconda 5.1 for Linux(x86 architecture, 64bit) Installer
            1. wget https://repo.continuum.io/archive/Anaconda3-5.1.0-Linux-x86_64.sh
            2. bash Anaconda3-5.1.0-Linux-x86_64.sh

    2. Create virtual environments with the Anaconda Python distribution
        ```conda env create -f=trf_multitask_env.yml```
    3. ```source activate trf_multitask_env```
    4. You can run my programme code in this environment

## Directory structure
```
|--Data ## Data (20news group corpus)
|  |--20news_train.xml ## Training data
|  |--20news_test1.xml ## Test data
|--README.md ## README
|--RESULT_TRF-Delay-Multi ## Saving directory for TRF-Delay-Multi results
|  |--TRF-Delay-Multi_opt.db ## Optimization database for TRF-Delay-Multi by Optuna
|--RESULT_TRF-Multi ## Saving directory for TRF-Multi results
|  |--TRF-Multi_opt.db ## Optimization database for TRF-Multi by Optuna
|--RESULT_TRF-Sequential ## Saving directory for TRF-Sequential results
|  |--TRF-Sequential_opt.db ## Optimization database for TRF-Multi by Optuna
|--RESULT_TRF-Single ## Saving directory for TRF-Single results
|  |--TRF-Single_opt.db  ## Optimization database for TRF-Single by Optuna
|--RESULT_XML-CNN  ## Saving directory for XML-CNN results
|  |--XML-CNN_opt.db  ## Optimization database for XML-CNN by Optuna
|--embedding  ## Directory of word embedding
|--hyper_parms_optuna.sh  ## shell script for optimizing hyper-parameters by Optuna
|--program  ## Programmes (Python)
|  |--__pycache__  ## cash
|  |  |--net.cpython-35.pyc
|  |  |--sentence_reader.cpython-35.pyc
|  |  |--xmlcnn.cpython-35.pyc
|  |--net.py  ##  TRF-XXX model (Single, Multi, Delay-Multi, Sequential)
|  |--opt_param.py  ##  Hyper-parameters optimization Programme by Optuna
|  |--sentence_reader.py  ##  programme for input data
|  |--train.py  ##  programm for training
|  |--xmlcnn.py  ## XML-CNN model
|--training.sh  ## shall script for training

```

## Quick-start
You can categorize sample data, 20news group by running training.sh, with XML-CNN.

The results are stored at CNN directory.

* RESULT_XXX :
    * RESULT_FILE_[N]EPOCH_TC: Results of model prediction and correct data for text categorization
    * RESULT_FILE_[N]EPOCH_TC_fscore: F score of text categorization
    * RESULT_FILE_[N]EPOCH_WSD: Results of model prediction and correct data for predominant word sense
    * RESULT_FILE_[N]EPOCH_WSD_fscore] F-score of domain-specific sense identification

### Training model change
You can change a training model by modifying the ```model``` in the file ```training.sh```
```                                                                                                                 
## hyper-params ##
epoch=100
batchSize=32
gpu=0
shuffle=yes
pretrained=0
multilabel=0
model=XML-CNN ## XML-CNN, TRF-Single, TRF-Multi, TRF-Delay-Multi, or TRF-Sequential ## <- change here
```

### Optimization of Hyper-parameters by Optuna ###
You can optimize hyper-parameters by running ````hyper_param_optuna.sh````.
You can optimize any models by changing ```model``` in ````hyper_param_optuna.sh````.
The results of the optimized hyper-parameters are stored `{model name}_opt.db` in the directory,
`RESULT_{model name}`. Here, `{model name}_opt.db` is a database and
the search process of the hyper parameters are stored in that file.

## hyper-params ##
epoch=100
batchSize=32
gpu=0
shuffle=yes
pretrained=0
multilabel=0
model=XML-CNN ## XML-CNN, TRF-Single, TRF-Multi, TRF-Delay-Multi, or TRF-Sequential ## <- change here


### Word embedding

You can use random vectors or vectors obtained by RCV1 corpus as word embedding by setting the argument, 0 or 1 of `--pretrained` in the file ```training.sh```
* 0: random vectors
* 1: vectors obtained by RCV1 corpus (my code utilize word embedding obtained by [fastText](https://github.com/facebookresearch/fastText))

```
## hyper-params ##
epoch=100
batchSize=32
gpu=0
shuffle=yes
pretrained=0 <-- change here (0 shows random vectors, 1 indicates word embedding obtained by fastText)
multilabel=0
model=XML-CNN ## XML-CNN, TRF-Single, TRF-Multi, TRF-Delay-Multi, or TRF-Sequential ##
```

## Datasets ##

20news group corpus is a default data. You can use your own data as
validation and training data by changing datapath as below:

`hyper_params_opt.sh`
```
DIR=/mnt/WD_Blue/Multitask_master/Corpus/ACL/5test/20news
valid_trainData=${DIR}/20news_train.xml <-- change here
valid_testData=${DIR}/20news_train.xml <--　change here
```
`training.sh`
```
DIR=/mnt/WD_Blue/Multitask_master/Corpus/ACL/5test/20news
trainData=${DIR}/20news_train.xml <-- change here
testData=${DIR}/20news_test1.xml <-- change here
```
When you use multi-labeled dataset such as RCV1 corpus, please set the argument `--multilabel` to 1.


```
## hyper-params ##
epoch=100
batchSize=32
gpu=0
shuffle=yes
pretrained=0
multilabel=0 <-- change here
model=XML-CNN ## XML-CNN, TRF-Single, TRF-Multi, TRF-Delay-Multi, or TRF-Sequential ##
```


