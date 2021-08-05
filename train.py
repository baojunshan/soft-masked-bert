from transformers import BertTokenizer
import torch

from utils import DataGenerator
from model import SoftMaskedBert, Trainer

pretrained_model_path = "../chinese_wwm_ext_pytorch"
train_data_path = "../data/train_data.json"
dev_data_path = "../data/test_data.json"
cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")

tokenizer = BertTokenizer.from_pretrained(pretrained_model_path)


data_generator = DataGenerator(
    path=train_data_path,
    tokenizer=tokenizer,
    batch_size=28,
    max_length=256,
)

dev_generator = DataGenerator(
    path=dev_data_path,
    tokenizer=tokenizer,
    batch_size=28,
    max_length=256,
)

model = SoftMaskedBert(
    pretrained_bert_path=pretrained_model_path,
    device=device,
    mask_token_id=tokenizer.mask_token_id
)

# model.load_state_dict(torch.load("best_model.bin"))

trainer = Trainer(
    model=model,
    device=device,
    tokenizer=tokenizer,
    lr=4e-4,
    alpha=0.8,
    epoch=3
)

trainer.fit(data_generator, dev_generator=dev_generator)

# for d in data[:10]:
#     res = trainer.inference(d["text"])
#     print(res)