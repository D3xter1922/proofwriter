#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import pandas as pd


# In[2]:


# df=pd.read_json("./proofwriter-dataset-V2020.12.3/CWA/depth-5/meta-train.jsonl", lines=True)


# In[3]:


# df


# In[4]:


# df.loc[16,'questions']['Q1']['proofsWithIntermediates'][2]['intermediates']


# In[5]:


# # s=df.loc[0,'questions']['Q3']['proofsWithIntermediates'][1]['representation']
# # s="((sent16 ((((sent1 ((sent1 -> sent19 % int5) -> sent5 % int4)) -> sent17 % int3)) -> sent8 % int2)) -> sent7 % int1)"
# s="triple12"
# print(s)


# In[20]:


def processProof(s, nfact):
    s=s.split()
#     print(s)
    p=0
    final=[]
    for idx,x in enumerate(s):
        for i,y in enumerate(x):
            if y=='(':
                s[idx]=s[idx].replace('(','',1)
                final.append(y)
            else:
                break
        v=0
        for i,y in enumerate(x):
            if x[len(x)-v-1] ==')':
                v+=1
                s[idx]=s[idx].replace(')','',1)
            else:
                final.append(s[idx])
                break
        while v!=0:
            final.append(')')
            v-=1

#     print((final))
    i=0
    Final=final.copy()
    while i<len(final)-1:
        if final[i]!='(' and final[i]!=')' and final[i]!='->' and final[i]!='%':
            if(final[i+1]=='('):
                final.insert(i+1,'&')
                i+=1
#                 print('1')
        elif final[i]==')':
            if final[i+1] !=')' and final[i+1]!='->' and final[i+1]!='%':
                final.insert(i+1,'&')
                i+=1
#                 print('2')
        elif final[i]!='(' and final[i] !=')' and final[i]!='->' and final[i]!='%':
            if final[i+1]!='(' and final[i+1] !=')' and final[i+1]!='->' and final[i+1]!='%':
                final.insert(i+1,'&')
                i+=1
#                 print('3')
        i+=1
    for i,e in enumerate(final):
        if e=='%':
            final[i-1 : i+2] = [''.join(final[i-1 : i+2])]
    for i,b in enumerate(final):
        if b!='(' and b!=')' and b!='&' and b!='->':
            if b[0]=='t':
                print('xx')
                c='sent'+b[6:]
                final[i]=c
            elif b[0]=='r':
                c='sent'+str(int(b[4])+nfact)+str(b[5:])
                final[i]=c
    if(len(final)==1):
        return str(final[0])
    print(final)
    sym=[]
    nod=[]
    root=-1
    for i,e in enumerate(final):
        if e=='(':
            sym.append(i)
#             print(sym)
#             print(nod)
        if e!='(' and e!=')' and e!='->' and e!='%' and e!='&':
            nod.append(i)
#             print(sym)
#             print(nod)
        if e=='->' or e=='%' or e=='&':
            sym.append(i)
#             print(sym)
#             print(nod)
        if e==')':
            z=sym.pop()
            if final[z]!='(' and final[z]!='->':
                right=nod.pop()
                left=nod.pop()

                root=Node(z)
                if(type(right)==Node):
                    root.right=right
                else:
                    root.right=Node(right)
                if(type(left)==Node):
                    root.left=left
                else:
                    root.left=Node(left)
                sym.pop()
                nod.append(root)
                print(root)
            elif final[z]=='->':
                left=nod.pop()
                right=nod.pop()

                root=Node(z)
                if(type(right)==Node):
                    root.right=right
                else:
                    root.right=Node(right)
                if(type(left)==Node):
                    root.left=left
                else:
                    root.left=Node(left)
                sym.pop()
                nod.append(root)
#                 print(root)
#             print(sym)
#             print(nod)
    def prefix(n):
        if n.left==None:
            return [n.val]
        ret=[]
        ret.insert(0,n.val)
        ret.extend(prefix(n.left))
        ret.extend(prefix(n.right))
        return ret
    lil=prefix(root)
#     print(lil)
    for i,e in enumerate(lil):
        lil[i]=final[e]
        if lil[i]=='->':
            lil[i]='#'
        if lil[i]=='%':
            lil[i]='@'
        
    return ' '.join([str(elem) for elem in lil])


# In[7]:


# processProof(s, 12)


# In[8]:


# pip install binarytree 
#


# In[22]:


from binarytree import Node


# In[10]:


# print(final)
# sym=[]
# nod=[]
# root=-1
# for i,e in enumerate(final):
#     if e=='(':
#         sym.append(i)
#         print(sym)
#         print(nod)
#     if e!='(' and e!=')' and e!='->' and e!='%' and e!='&':
#         nod.append(i)
#         print(sym)
#         print(nod)
#     if e=='->' or e=='%' or e=='&':
#         sym.append(i)
#         print(sym)
#         print(nod)
#     if e==')':
#         z=sym.pop()
#         if final[z]!='(' and final[z]!='->':
#             right=nod.pop()
#             left=nod.pop()
            
#             root=Node(z)
#             if(type(right)==Node):
#                 root.right=right
#             else:
#                 root.right=Node(right)
#             if(type(left)==Node):
#                 root.left=left
#             else:
#                 root.left=Node(left)
#             sym.pop()
#             nod.append(root)
#             print(root)
#         elif final[z]=='->':
#             left=nod.pop()
#             right=nod.pop()
            
#             root=Node(z)
#             if(type(right)==Node):
#                 root.right=right
#             else:
#                 root.right=Node(right)
#             if(type(left)==Node):
#                 root.left=left
#             else:
#                 root.left=Node(left)
#             sym.pop()
#             nod.append(root)
#             print(root)
#         print(sym)
#         print(nod)
# print(df.loc[0,'questions']['Q3']['proofsWithIntermediates'][1]['representation'])
# print(final)


# In[11]:


def prefix(n):
    if n.left==None:
        return [n.val]
    ret=[]
    ret.insert(0,n.val)
    ret.extend(prefix(n.left))
    ret.extend(prefix(n.right))
    return ret


# In[12]:


# import itertools
# lil=prefix(root)
# print(lil)
# for i,e in enumerate(lil):
#     lil[i]=final[e]
# print(lil)


# In[ ]:





# In[13]:


# '''
# '#' for <-
# % conjoins rules and conlcusions also '@'
# &

# ''' 

# s=df.loc[0,'questions']['Q3']['proofsWithIntermediates'][1]['representation']
# stack=[]
# arr=[]
# op=[]
# s=s[1:-1]
# # s=s.split()
# for e in s:
# #     print("_",e)

#     if(e[0]=='('):
#         stack.append(e[0])
# #         print(stack)
#     elif(e[0]==')'):
#         stack.pop()
# #         print(stack)
#         print(arr)
#         if(len(stack)==0):
# #             print("empty")
#             print(arr)
#             arr=[]
#     elif(len(stack)>0):
#         arr.append(e[0])
#     else:
#         op.append(e[0])
# print(op)


# In[14]:





# In[15]:


# s


# In[1]:


import argparse
import glob
import os
import json
import time
import logging
import random
import re
from itertools import chain
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from transformers import AdamW, T5ForConditionalGeneration, T5Tokenizer, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler


# In[2]:


def extract_data_rows(path):
    df=pd.read_json(path, lines=True)
    data_rows=[]
    for index, row in df.iterrows():
        context=row['theory']
        nfact=row['NFact']
        for question in row['questions'].items():
            if 'proofsWithIntermediates' in question[1].keys():
                Question = question[1]['question']
                answer = question[1]['answer']
                proof = question[1]['proofsWithIntermediates'][0]['representation']
                intermediates = question[1]['proofsWithIntermediates'][0]['intermediates']
                data_rows.append({
                    'question': Question,
                    'nfact':nfact,
                    'context': context,
                    'answer': answer,
                    'proof': proof,
                    'intermediate': intermediates
                })
    return pd.DataFrame(data_rows)


# In[3]:


path="./proofwriter-dataset-V2020.12.3/CWA/depth-5/meta-train.jsonl"
extract_data_rows(path)


# In[4]:


data=extract_data_rows(path)


# In[5]:


MODEL_NAME='t5-large'


# In[6]:


tokenizer=T5Tokenizer.from_pretrained(MODEL_NAME)


# In[7]:


tokenizer("would I rather be feared or loved")


# In[8]:


sample_question=data.iloc[200]
print(sample_question)


# In[9]:


# encoding =tokenizer(sample_question["question"], sample_question["context"], max_length=396, padding="max_length", truncation="only_second", return_attention_mask=True, add_special_tokens=True, return_tensors="pt")


# # In[10]:


# answer_encoding=tokenizer(sample_question["answer"], sample_question["context"], max_length=396, padding="max_length", truncation="only_second", return_attention_mask=True, add_special_tokens=True, return_tensors="pt")


# In[29]:


def T5Trainer(
    dataframe, source_text, target_text, model_params, output_dir="./outputs/"
):

    """
    T5 trainer

    """

    # Set random seeds and deterministic pytorch for reproducibility
    torch.manual_seed(model_params["SEED"])  # pytorch random seed
    np.random.seed(model_params["SEED"])  # numpy random seed
    torch.backends.cudnn.deterministic = True

    # logging
    console.log(f"""[Model]: Loading {model_params["MODEL"]}...\n""")

    # tokenzier for encoding the text
    tokenizer = T5Tokenizer.from_pretrained(model_params["MODEL"])

    # Defining the model. We are using t5-base model and added a Language model layer on top for generation of Summary.
    # Further this model is sent to device (GPU/TPU) for using the hardware.
    model = T5ForConditionalGeneration.from_pretrained(model_params["MODEL"])
    model = model.to(device)

    # logging
    console.log(f"[Data]: Reading data...\n")

    # Importing the raw dataset
    dataframe = dataframe[[source_text, target_text]]
    display_df(dataframe.head(2))

    # Creation of Dataset and Dataloader
    # Defining the train size. So 80% of the data will be used for training and the rest for validation.
    train_size = 0.8
    train_dataset = dataframe.sample(frac=train_size, random_state=model_params["SEED"])
    val_dataset = dataframe.drop(train_dataset.index).reset_index(drop=True)
    train_dataset = train_dataset.reset_index(drop=True)

    console.print(f"FULL Dataset: {dataframe.shape}")
    console.print(f"TRAIN Dataset: {train_dataset.shape}")
    console.print(f"TEST Dataset: {val_dataset.shape}\n")

    # Creating the Training and Validation dataset for further creation of Dataloader
    training_set = YourDataSetClass(
        train_dataset,
        tokenizer,
        model_params["MAX_SOURCE_TEXT_LENGTH"],
        model_params["MAX_TARGET_TEXT_LENGTH"],
        source_text,
        target_text,
    )
    val_set = YourDataSetClass(
        val_dataset,
        tokenizer,
        model_params["MAX_SOURCE_TEXT_LENGTH"],
        model_params["MAX_TARGET_TEXT_LENGTH"],
        source_text,
        target_text,
    )

    # Defining the parameters for creation of dataloaders
    train_params = {
        "batch_size": model_params["TRAIN_BATCH_SIZE"],
        "shuffle": True,
        "num_workers": 0,
    }

    val_params = {
        "batch_size": model_params["VALID_BATCH_SIZE"],
        "shuffle": False,
        "num_workers": 0,
    }

    # Creation of Dataloaders for testing and validation. This will be used down for training and validation stage for the model.
    training_loader = DataLoader(training_set, **train_params)
    val_loader = DataLoader(val_set, **val_params)

    # Defining the optimizer that will be used to tune the weights of the network in the training session.
    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=model_params["LEARNING_RATE"]
    )

    # Training loop
    console.log(f"[Initiating Fine Tuning]...\n")

    for epoch in range(model_params["TRAIN_EPOCHS"]):
        train(epoch, tokenizer, model, device, training_loader, optimizer)

    console.log(f"[Saving Model]...\n")
    # Saving the model after training
    path = os.path.join(output_dir, "model_files")
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)

    # evaluating test dataset
    console.log(f"[Initiating Validation]...\n")
    for epoch in range(model_params["VAL_EPOCHS"]):
        predictions, actuals = validate(epoch, tokenizer, model, device, val_loader)
        final_df = pd.DataFrame({"Generated Text": predictions, "Actual Text": actuals})
        final_df.to_csv(os.path.join(output_dir, "predictions.csv"))

    console.save_text(os.path.join(output_dir, "logs.txt"))

    console.log(f"[Validation Completed.]\n")
    console.print(
        f"""[Model] Model saved @ {os.path.join(output_dir, "model_files")}\n"""
    )
    console.print(
        f"""[Validation] Generation on Validation data saved @ {os.path.join(output_dir,'predictions.csv')}\n"""
    )
    console.print(f"""[Logs] Logs saved @ {os.path.join(output_dir,'logs.txt')}\n""")


# In[30]:


def validate(epoch, tokenizer, model, device, loader):

  """
  Function to evaluate model for predictions

  """
  model.eval()
  predictions = []
  actuals = []
  with torch.no_grad():
      for _, data in enumerate(loader, 0):
          y = data['target_ids'].to(device, dtype = torch.long)
          ids = data['source_ids'].to(device, dtype = torch.long)
          mask = data['source_mask'].to(device, dtype = torch.long)

          generated_ids = model.generate(
              input_ids = ids,
              attention_mask = mask, 
              max_length=150, 
              num_beams=2,
              repetition_penalty=2.5, 
              length_penalty=1.0, 
              early_stopping=True
              )
          preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
          target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True)for t in y]
          if _%10==0:
              console.print(f'Completed {_}')

          predictions.extend(preds)
          actuals.extend(target)
  return predictions, actuals


# In[46]:


import sys
print(sys.executable)
print(sys.version)
print(sys.version_info)

print(torch.__version__)
# In[39]:


from torch.cuda.amp import GradScaler, autocast
def train(epoch, tokenizer, model, device, loader, optimizer):

    """
    Function to be called for training with the parameters passed from main function

    """

    


    scaler = GradScaler()


    
    model.train()
    for _, data in enumerate(loader, 0):
        y = data["target_ids"].to(device, dtype=torch.long)
        y_ids = y[:, :-1].contiguous()
        lm_labels = y[:, 1:].clone().detach()
        lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
        ids = data["source_ids"].to(device, dtype=torch.long)
        mask = data["source_mask"].to(device, dtype=torch.long)

        outputs = model(
            input_ids=ids,
            attention_mask=mask,
            decoder_input_ids=y_ids,
            labels=lm_labels,
        )
        loss = outputs[0]

        if _ % 10 == 0:
            training_logger.add_row(str(epoch), str(_), str(loss))
            console.print(training_logger)

        optimizer.zero_grad()
#         scaler.scale(loss).backward()
        loss.backward()
#         scaler.step(optimizer)
        optimizer.step()
#         scaler.update()


# In[32]:


class YourDataSetClass(Dataset):
    """
    Creating a custom dataset for reading the dataset and
    loading it into the dataloader to pass it to the
    neural network for finetuning the model

    """

    def __init__(
        self, dataframe, tokenizer, source_len, target_len, source_text, target_text
    ):
        """
        Initializes a Dataset class

        Args:
            dataframe (pandas.DataFrame): Input dataframe
            tokenizer (transformers.tokenizer): Transformers tokenizer
            source_len (int): Max length of source text
            target_len (int): Max length of target text
            source_text (str): column name of source text
            target_text (str): column name of target text
        """
        self.tokenizer = tokenizer
        self.data = dataframe
        self.source_len = source_len
        self.summ_len = target_len
        self.target_text = self.data[target_text]
        self.source_text = self.data[source_text]

    def __len__(self):
        """returns the length of dataframe"""

        return len(self.target_text)

    def __getitem__(self, index):
        """return the input ids, attention masks and target ids"""

        source_text = str(self.source_text[index])
        target_text = str(self.target_text[index])

        # cleaning data so as to ensure data is in string type
        source_text = " ".join(source_text.split())
        target_text = " ".join(target_text.split())

        source = self.tokenizer.batch_encode_plus(
            [source_text],
            max_length=self.source_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        target = self.tokenizer.batch_encode_plus(
            [target_text],
            max_length=self.summ_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()
        target_ids = target["input_ids"].squeeze()
        target_mask = target["attention_mask"].squeeze()

        return {
            "source_ids": source_ids.to(dtype=torch.long),
            "source_mask": source_mask.to(dtype=torch.long),
            "target_ids": target_ids.to(dtype=torch.long),
            "target_ids_y": target_ids.to(dtype=torch.long),
        }


# In[33]:


model_params = {
    "MODEL": "t5-large",  # model_type: t5-base/t5-large
    "TRAIN_BATCH_SIZE": 1,  # training batch size
    "VALID_BATCH_SIZE": 1,  # validation batch size
    "TRAIN_EPOCHS": 3,  # number of training epochs
    "VAL_EPOCHS": 1,  # number of validation epochs
    "LEARNING_RATE": 1e-4,  # learning rate
    "MAX_SOURCE_TEXT_LENGTH": 512,  # max length of source text
    "MAX_TARGET_TEXT_LENGTH": 64,  # max length of target text
    "SEED": 42,  # set seed for reproducibility
}


# In[34]:


data


# In[15]:


G=[]
for i,e in enumerate(data['context']):
    f=e.split('. ')
    e=""
    for i,x in enumerate(f):
        e=e+"sent"+str(i+1)+": "+x+". "
    e=e[:-2]
    G.append(e)
data['context2']=G


# In[16]:


print(data['context2'][0])


# In[17]:


data["text"]="$answer$ ; $proof$ ; $question$ = "+ data['question']+" ; $context$ = "+data["context2"]


# In[18]:


for q in data['intermediate'][20].items():
    print(q[1])


# In[23]:



G=[]
for i,x in enumerate(data['proof']):
    e=processProof(x, data['nfact'][i])
    if(len(data['intermediate'][i])!=0):
        e=e+" ; with "
        h=1
        for q in data['intermediate'][20].items():
            e+="int"+str(h)+": "+ q[1]['text']+" ; "
        e=e[:-3]
    G.append(e)
data['proof2']=G


# In[24]:


G=[]
for i,x in enumerate(data['proof2']):
    G.append(str("$answer$ = "+str(data['answer'][i])+" ; $proof$ = "+x))

data["target"]=G


# In[25]:


data


# In[26]:


import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import os

# Importing the T5 modules from huggingface/transformers
from transformers import T5Tokenizer, T5ForConditionalGeneration

# rich: for a better display on terminal
from rich.table import Column, Table
from rich import box
from rich.console import Console

# define a rich console logger
console = Console(record=True)


# In[27]:


def display_df(df):
    """display dataframe in ASCII format"""

    console = Console()
    table = Table(
        Column("source_text", justify="center"),
        Column("target_text", justify="center"),
        title="Sample Data",
        pad_edge=False,
        box=box.ASCII,
    )

    for i, row in enumerate(df.values.tolist()):
        table.add_row(row[0], row[1])

    console.print(table)

# training logger to log training progress
training_logger = Table(
    Column("Epoch", justify="center"),
    Column("Steps", justify="center"),
    Column("Loss", justify="center"),
    title="Training Status",
    pad_edge=False,
    box=box.ASCII,
)


# In[37]:



device = 'cuda'
T5Trainer(
    dataframe=data,
    source_text="text",
    target_text="target",
    model_params=model_params,
    output_dir="outputs",
)


# In[48]:


import torch
torch.cuda.empty_cache()


# In[36]:


from GPUtil import showUtilization as gpu_usage
gpu_usage()


# In[55]:


from GPUtil import showUtilization as gpu_usage
from numba import cuda

def free_gpu_cache():
    print("Initial GPU Usage")
    gpu_usage()                             

    torch.cuda.empty_cache()

    cuda.select_device(0)
    cuda.close()
    cuda.select_device(0)

    print("GPU Usage after emptying the cache")
    gpu_usage()

free_gpu_cache()


# In[ ]:




