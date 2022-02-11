![](img/alex-icon.png)

# ALEX(Logic Expression Learning)

論理式で表現する機械学習モデルです．

# How to use

1. Amplify AEのTokenの入手と保存
[https://amplify.fixstars.com/ja/](Amplify HP)

このリンクからサインアップし，アニーリングマシンのアクセストークンを入手してください．

そして，ローカルの`~/.amplify/token.json`に以下の形式でTokenを保存してください．
```json
{
  "AMPLIFY_TOKEN": "*****"
}
```

2ソースのダウンロード

```shell
git clone https://github.com/Chizuchizu/logic-expression-learning
mv logic-expression-learning
```

3環境構築

- poetryの人

```shell
poetry install
```

- それ以外の人
仮想環境の上で`pip install`してください．
```shell
pip install -r requirements.txt
```

4. 実行

サンプルコードは`src/example_golf.py`にあります．

```shell
cd logic_expression_learning
python example_golf.py
```

