# soft-masked-bert
An unofficial implementation of soft-masked bert by pytorch and transformers, welcome to issue and PR. ![img.png](img.png)

Paper: [Spelling Error Correction with Soft-Masked BERT](https://arxiv.org/abs/2005.07421)
Train data: [Baidu Cloud](https://pan.baidu.com/s/1Qnh5MtC3-8AKpce145AmKA) (password:l545)
Pretrained bert model: [Chinese-BERT-wwm](https://github.com/ymcui/Chinese-BERT-wwm)

You can add custom data following the data format:
```json
{'text': '但是我不能去参加，因为我有一点事情阿！', 'mistakes': [{'wrong': '阿', 'correct': '啊', 'loc': '18'}]}
```

You can train or inference the model by `train.py`
```shell
python train.py
```