#coding: utf-8
#Created Time: 2020-06-16 21:21:53
#train
import torch
import transformers
import pandas as pd
from tqdm import tqdm
from data_process import Dataset

MODEL_PATH = r"/Users/zeehu/Work/Text-Classification/bert-base-chinese/"
tokenizer = \
transformers.BertTokenizer.from_pretrained(r"/Users/zeehu/Work/Text-Classification/bert-base-chinese/bert-base-chinese-vocab.txt") 
TRAINING_FILE = 'test'

def train_fn(data_loader, model, optimizer, device, scheduler=None):
    model.train()
    tk0 = tqdm(data_loader, total=len(data_loader))
    for bi, d in enumerate(tk0):
        ids = d["ids"]
        token_type_ids = d["token_type_ids"]
        mask = d["attention_mask"]

        ids = ids.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        model.zero_grad()
        start_scores, end_scores = model(ids=ids,
                                            mask=mask,
                                            token_type_ids=token_tyoe_ids)

def run(fold):
    dfx = pd.read_csv(TRAINING_FILE)
    df_train = dfx[dfx.kfold != fold].reset_index(drop=True)
    df_valid = dfx[dfx.kfold == fold].reset_index(drop=True)
    train_dataset = Dataset(tokenizer, df_train.src.values, \
            df_train.dst.values, df_train.label.values)
    train_data_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=2,
            num_workers=1)
    train_fn(train_data_loader, model, optimizer, device)
run(0)
run(2)
