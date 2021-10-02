# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 23:23:30 2021

@author: VXhpUS
"""
import os
import datetime as dt
import re
import pandas as pd
import numpy as np
import torch
import matplotlib as mp
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, HTML
import matplotlib.font_manager as fm


#%% Import Registration
df = pd.read_json(path_reg + '/records/contents_of_registrations.json')
df = dy_reg.Refined_Records(df)

#%% Identify Current Issue
curr_issue_name = df.sort_values(['公示日期', '公示批次名称'], ascending=False)['公示批次名称'].iloc[0]
df_curr =  df.loc[df.公示批次名称 == curr_issue_name]
issue_name = df_curr['公示批次起始'].iloc[0][0] + '年' + df_curr['公示批次起始'].iloc[0][1] +'月'
issue_name += df_curr['公示批次起始'].iloc[0][2]
print('Current Issue: {}'.format(issue_name))

#%% Predict Title
from transformers import BertTokenizer, BartForConditionalGeneration

# assign device
if torch.cuda.device_count() > 0:
  device = 'cuda:' + str(torch.cuda.current_device())
else:
  device = 'cpu'

# Instantiate tokenizer and model
checkpoint = "/content/drive/MyDrive/Github/Content/tools/models/PredTitle-10000"

tokenizer = BertTokenizer.from_pretrained(checkpoint)
model = BartForConditionalGeneration.from_pretrained(checkpoint)
model.to(device)
model.eval()

batch_size = 8
i = 0
ls = df_curr['梗概'].tolist()
L = df_curr.shape[0]
test_predictions = []

while i < L:
  inputs = tokenizer(ls[i:i+batch_size],
                           padding=True,
                           max_length=512, 
                           truncation=True, 
                           return_tensors='pt')
  inputs.to(device)
  summary_ids = model.generate(input_ids=inputs['input_ids'],
                             num_beams=4,
                             min_length=0,
                             max_length=32
                             )
  
  ret = [tokenizer.decode(g, 
                         skip_specical_tokens=True, 
                         clean_up_tokenization_spaces=True) for g in summary_ids]
  test_predictions.extend(ret)
  i += batch_size
  
  df_curr['预测片名'] = test_predictions

def remove_specials(x):
  x = re.sub(' ', '', x)
  x = re.sub('\[CLS\]', '', x)
  x = re.sub('\[PAD\]', '', x)
  x = re.sub('\[SEP\]', '', x)
  return x

df_curr['预测片名'] = df_curr['预测片名'].apply(remove_specials)