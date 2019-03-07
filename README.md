語義の曖昧さ解消とのマルチタスク学習に基づく文書の自動分類
==
このコードでは次の5種類のモデルが利用できます:

* XML-CNN ([Liu+ '17](http://nyc.lti.cs.cmu.edu/yiming/Publications/jliu-sigir17.pdf)) : Liuら'17 の提案したモデル
* TRF-Single: Transformer Encoderを用いて文書行列を作成, 文書分類タスクのみを学習
* TRF-Multi: Transformer Encoderを用いて文書行列を作成, 語義の曖昧さ解消タスクと文書分類タスクを同時に学習
* TRF-Delay-Multi: Transformer Encoderを用いて文書行列を作成, 語義の曖昧さ解消タスクを先行して学習, その後文書分類タスクと統合しマルチタスク学習
* TRF-Sequential: Transformer Encoderを用いて文書行列を作成, 語義の曖昧さ解消タスクを学習, その後文書分類タスクをシングルタスクとして学習


### 各モデルの特徴

|      特徴\モデル     | XML-CNN | TRF-Single | TRF-Multi |                        TRF-Delay-Multi                       |        TRF-Sequential       |
|:--------------------:|:-------:|:----------:|:---------:|:------------------------------------------------------------:|:---------------------------:|
| Convolution?         |    ✔    |            |           |                                                              |                             |
| Single-Task?         |    ✔    |      ✔     |           |                                                              | ✔(それぞれのタスクを順番に) |
|      Multi-Task?     |         |            |     ✔     | ✔(先に語義の曖昧さ解消タスク,  途中から文書分類タスクと統合) |                             |
| Transformer Encoder? |         |      ✔     |     ✔     |                               ✔                              |              ✔              |

## Requirements
このコードを実行するために必要なライブラリのうち、代表的なものを次に示します。
* Python 3.5.4 以降
* Chainer 4.0.0 以降 ([chainer](http://chainer.org/))
* CuPy 4.0.0 以降 ([cupy](https://cupy.chainer.org/))
* Optuna 0.8.0 以降 ([optuna](https://optuna.org/))

注意: 
* 現在のコードのバージョンでは**GPU**を利用することが前提となっています。
* コードを実行するために必要なライブラリの詳細はrequirements.txtをご参照ください。

## Installation
* このページの **clone or download** からコードをダウンロード
* requirements.txtに書かれたライブラリをインストールし、実行環境を構築
* もし必要であれば、次の手順でAnaconda([anaconda](https://www.anaconda.com/enterprise/))による仮想環境を構築
    1. [Anacondaのダウンロードページ](https://www.anaconda.com/download/)から自分の環境にあったものをインストール
        * 例: Linux(x86アーキテクチャ, 64bit)にインストールする場合:
            1. wget https://repo.continuum.io/archive/Anaconda3-5.1.0-Linux-x86_64.sh
            1. bash Anaconda3-5.1.0-Linux-x86_64.sh
            
            でインストールできます。
    3. Anacondaをインストール後、仮想環境を構築
        ```conda env create -f=trf_multitask_env.yml```
    4. ```source activate trf_multitask_env```　で仮想環境に切り替え
    5. この環境内でコードを実行することが可能

## ディレクトリ構造
```
|--Data ## データ(20news groupコーパス)
|  |--20news_train.xml ## 訓練データ
|  |--20news_test1.xml ## テストデータ
|--README.md ## README
|--RESULT_TRF-Delay-Multi ## TRF-Delay-Multiの結果保存ディレクトリ
|  |--TRF-Delay-Multi_opt.db ## TRF-Delay-MultiのOptunaの最適化データベース
|--RESULT_TRF-Multi ## TRF-Multiの結果保存ディレクトリ
|  |--TRF-Multi_opt.db ## TRF-MultiのOptunaの最適化データベース
|--RESULT_TRF-Sequential ## TRF-Sequentialの結果保存ディレクトリ
|  |--TRF-Sequential_opt.db ## TRF-MultiのOptunaの最適化データベース
|--RESULT_TRF-Single ## TRF-Singleの結果保存ディレクトリ
|  |--TRF-Single_opt.db  ## TRF-SingleのOptunaの最適化データベース
|--RESULT_XML-CNN  ## XML-CNNの結果保存ディレクトリ
|  |--XML-CNN_opt.db  ## XML-CNNのOptunaの最適化データベース
|--embedding  ## 事前学習済み分散表現をここに配置(初期は空ディレクトリ)
|--hyper_parms_optuna.sh  ## Optunaによるハイパーパラメータの最適化を行うシェルスクリプト
|--program  ## プログラム(Python)群
|  |--__pycache__  ## キャッシュ
|  |  |--net.cpython-35.pyc
|  |  |--sentence_reader.cpython-35.pyc
|  |  |--xmlcnn.cpython-35.pyc
|  |--net.py  ##  TRF-XXXモデル(Single, Multi, Delay-Multi, Sequential)
|  |--opt_param.py  ##  Optunaによるハイパーパラメータ最適化プログラム
|  |--sentence_reader.py  ##  データの読み込みプログラム
|  |--train.py  ##  本訓練プログラム
|  |--xmlcnn.py  ## XML-CNNモデル
|--training.sh  ## 本訓練を行うシェルスクリプト

```

## Quick-start
training.shを実行することでXML-CNNによる20news groupデータセットの文書分類タスクを実行できます.

学習後の結果はCNNディレクトリに保存されます.
* RESULT_XXX :
    * RESULT_FILE_[N]EPOCH_TC: 文書分類タスクの正解とモデル予測結果
    * RESULT_FILE_[N]EPOCH_TC_fscore: 文書分類タスクのF値
    * RESULT_FILE_[N]EPOCH_WSD: 語義の曖昧さ解消タスクの正解とモデル予測結果
    * RESULT_FILE_[N]EPOCH_WSD_fscore] 語義の曖昧さ解消タスクのF値

### 学習モデルの変更
```training.sh```内の ```model``` を変更することで学習するモデルを変更することができます.
```                                                                                                                 
## hyper-params ##
epoch=100
batchSize=32
gpu=0
shuffle=yes
pretrained=0
multilabel=0
model=XML-CNN ## XML-CNN, TRF-Single, TRF-Multi, TRF-Delay-Multi, or TRF-Sequential ## <-ここ
```

### Optunaによるハイパーパラメータの最適化 ###
````hyper_param_optuna.sh````を実行することでOptunaを使ったモデルのハイパーパラメータを最適化を行います.
````hyper_param_optuna.sh````内の```model```を変更することで最適化するモデルを変更することができます.
最適化したハイパーパラメータの結果は`RESULT_{モデル名}`ディレクトリ内の`{モデル名}_opt.db`に保存されます.
`{モデル名}_opt.db`はデータベースですので, ハイパーパラメータの探索過程が保存されています.
```                                                                                                                 
## hyper-params ##
epoch=100
batchSize=32
gpu=0
shuffle=yes
pretrained=0
multilabel=0
model=XML-CNN ## XML-CNN, TRF-Single, TRF-Multi, TRF-Delay-Multi, or TRF-Sequential ## <-ここ
```


### 単語の分散表現について

```training.sh```内の 引数`--pretrained`に0または1を設定することで利用する分散表現を変更します.
* 0: ランダムベクトル
* 1: RCV1コーパスを用いた分散表現(このコードでは単語の分散表現に[fastText](https://github.com/facebookresearch/fastText)の学習結果を利用しています)

```
## hyper-params ##
epoch=100
batchSize=32
gpu=0
shuffle=yes
pretrained=0 <-- ここ (0ならばランダムベクトル, 1ならばfastTextによる事前学習の分散表現)
multilabel=0
model=XML-CNN ## XML-CNN, TRF-Single, TRF-Multi, TRF-Delay-Multi, or TRF-Sequential ##
```

## データセットについて ##

デフォルトでは20news groupデータセットの学習を行います.
Optunaの最適化, 本訓練の利用データを変更するには`hyper_params_opt.sh`, `training.sh`のデータパスを変更してください.

`hyper_params_opt.sh`
```
DIR=/mnt/WD_Blue/Multitask_master/Corpus/ACL/5test/20news
valid_trainData=${DIR}/20news_train.xml <-- ここ
valid_testData=${DIR}/20news_train.xml <--　ここ
```
`training.sh`
```
DIR=/mnt/WD_Blue/Multitask_master/Corpus/ACL/5test/20news
trainData=${DIR}/20news_train.xml <-- ここ
testData=${DIR}/20news_test1.xml <-- ここ
```
マルチラベルのデータセット(RCV1コーパスなど)に変更した場合, 引数`--multilabel`に1を設定してください.

```
## hyper-params ##
epoch=100
batchSize=32
gpu=0
shuffle=yes
pretrained=0
multilabel=0 <-- ここ
model=XML-CNN ## XML-CNN, TRF-Single, TRF-Multi, TRF-Delay-Multi, or TRF-Sequential ##
```