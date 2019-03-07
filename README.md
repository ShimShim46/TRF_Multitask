語義の曖昧さ解消とのマルチタスク学習に基づく文書の自動分類
==
このコードでは次の5種類のモデルが利用できます:

* XML-CNN モデル ([Liu+ '17](http://nyc.lti.cs.cmu.edu/yiming/Publications/jliu-sigir17.pdf)) : Liuら'17 の提案したモデル
* TRF-Single: Transformer Encoderを用いて文書行列を作成, 文書分類タスクのみを学習
* TRF-Multi: Transformer Encoderを用いて文書行列を作成, 語義の曖昧さ解消タスクと文書分類タスクを同時に学習
* TRF-Delay-Multi: Transformer Encoderを用いて文書行列を作成, 語義の曖昧さ解消タスクを先行して学習, その後文書分類タスクと統合しマルチタスク学習
* TRF-Sequential: Transformer Encoderを用いて文書行列を作成, 語義の曖昧さ解消タスクを学習, その後文書分類タスクをシングルタスクとして学習


### 各モデルの特徴

|              特徴\手法 |   Flatモデル  |   WoFtモデル  |   HFTモデル   |    XML-CNNモデル    |
|-----------------------:|:-------------:|:-------------:|:-------------:|:-------------------:|
|              Hierarchycal Structure |               |       ✔       |       ✔       |                     |
|            Fine-tuning |               |       ✔       |       ✔       |                     |
|                Pooling Type | 1-max pooling | 1-max pooling | 1-max pooling | dynamic max pooling |
| Compact Representation |               |               |               |          ✔          |

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
        ```conda env create -f=hft_cnn_env.yml```
    4. ```source activate hft_cnn_env```　で仮想環境に切り替え
    5. この環境内でHFT-CNNのコードを実行することが可能

## ディレクトリ構造
```
|--CNN  ## 学習結果を保存されるディレクトリ
|  |--LOG     ## 学習ログ                                                                                                        
|  |--PARAMS  ## CNNの学習パラメータ
|  |--RESULT  ## 分類結果
|--cnn_model.py  ## CNNモデル
|--cnn_train.py  ## CNNの学習
|--data_helper.py  ## データ整形/操作
|--example.sh  ## 実行することでサンプルデータの分類が可能
|--hft_cnn_env.yml ## 依存関係(Anaconda)
|--LICENSE  ## MITライセンス
|--MyEvaluator.py  ## CNNの学習 validationの処理
|--MyUpdater.py  ## CNNの学習 1iterationの処理
|--README.md  ## README
|--requirements.txt  ## 依存関係(pip)
|--Sample_data  ## サンプルの文書データ(Amazon)
|  |--sample_test.txt  ## 評価
|  |--sample_train.txt  ## 訓練
|  |--sample_valid.txt  ## 検証
|--train.py  ## main関数
|--Tree
|  |--Amazon_all.tree   ## Amazon用の木構造ファイル
|--tree.py  ## 木構造の操作
|--Word_embedding  ## 単語の分散表現ディレクトリ
|--xml_cnn_model.py  ## LiuらのXML-CNNモデル(chainer実装)
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
python program/train.py -itrain ${trainData}  -itest ${testData} -m ${model}  \
 -e ${epoch}  -b ${batchSize} -g ${gpu} --filepath ${FP} \
--shuffle ${shuffle} --pretrained ${pretrained} --multilabel ${multilabel}
```
