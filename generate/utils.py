# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 22:11:14 2021

@author: VXhpUS
"""
import sys
import pandas as pd
import numpy as np
import re
import matplotlib as mp
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, HTML
import matplotlib.font_manager as fm

path_fonts = '/content/drive/MyDrive/Github/Article/fonts'
fontprop = fm.FontProperties(fname=path_fonts, size= 15)
font_dirs = [path_fonts, ]
font_files = fm.findSystemFonts(fontpaths=font_dirs)
font_list = fm.createFontList(font_files)
for font in font_files:
  fm.fontManager.addfont(font)
plt.rcParams['figure.figsize'] = [15, 9]
mp.rcParams['font.family'] = ['Microsoft YaHei']

import wordcloud
from PIL import Image as pil
import matplotlib as mp
from collections import Counter

# set path
path_img = '/content/drive/MyDrive/Github/Article/img'

#%%
def plot_type(df, issue_name:str, return_df=False):
  df = df.copy()
  df_n_type = df.groupby('电影类别')['电影类别'].count().rename('数量'
    ).sort_values(ascending=False)
  
  labels =[]
  for label, count in zip(df_n_type.index.map(lambda x: x[:2]),df_n_type):
    labels.append('{}片 {}'.format(label, count))

  explode = [0]*df_n_type.shape[0]
  explode[0] = 0.1
  explode = tuple(explode)
  plt.clf()
  plt.rcParams['figure.figsize'] = [8, 8]
  plt.rcParams['axes.facecolor'] = 'white'
  ax = df_n_type.plot.pie(
      #y='数量',
      #grid = True,
      fontsize = 12,
      colors=sns.color_palette("Set2"),
      explode = explode,
      labels=['']*df_n_type.shape[0],
      startangle = 45,
      radius=1.2,
      #autopct='%1.1f%%',
  )
  ax.set_title("电影类别",fontsize=24, pad=20)
  #ax.grid(color='gray', linestyle='-', linewidth=0.5)
  ax.legend(fontsize=14, labels=labels, loc='lower left')
  ax.set_xlabel('',fontsize= 18)
  #ax.set_xticklabels(df_n_time['年代'], fontsize=12)
  ax.set_ylabel("",fontsize= 18)

  fp = path_img + '/df_Reg_plot_type_{}.png'.format(issue_name)
  plt.savefig(fp, bbox_inches='tight')

  plt.show()
  if return_df:
    return fp, df_n_type
  return fp
  
#%% plot time
def plot_time(df, issue_name:str, return_df=False) -> str:
  """
  This function makes a genre plot with df
  df: a pandas DataFrame containing at least the following columns:
    ['年代','']

  returns
    str: filpath to saved plot
    df: grouped pandas DataFrame
  """
  df = df.copy()
  # plot genre in barh format
  plt.rcParams['figure.figsize'] = [6, 3]
 
  dft = df.groupby('年代')['年代'].count().rename('数量')
  idx_time = dft.index
  idx_ordered = []
  for time in ['古代', '近代', '现代', '当代']:
    if time in idx_time:
      idx_ordered.append(time)
  dft = dft.reindex(idx_ordered)
  x = ['古代\n<1840', '近代\n1840-1919', '现代\n1919-1949', '当代\n>1949']

  fig, ax = plt.subplots()
  ax.stem(x, dft, linefmt='lightsteelblue', 
          label = dft,
          basefmt='C4-', markerfmt='C9o',
          use_line_collection=True)

  for x_,y_ in zip(x,dft):
    ax.annotate('{}部'.format(y_), xy=(x_,y_+6), 
                fontsize=18, color='royalblue', ha="center")

  ax.set_title("年代",fontsize= 24, pad=20)
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.spines['bottom'].set_visible(False)
  ax.spines['left'].set_visible(False)
  ax.get_yaxis().set_visible(False)
  ax.set_xticklabels(x, fontsize=12)
  #ax.legend(fontsize=14)
  #ax.set_xlabel('年代',fontsize= 18)
  fp = path_img + '/df_Reg_time_{}.png'.format(issue_name)
  plt.savefig(fp, bbox_inches='tight')

  plt.show()
  if return_df:
    return fp, dft
  return fp

#%% plot genre
def plot_genre(df, name:str, stacked=False, return_df=False) -> str:
  """
  This function makes a genre plot with df
  df: a pandas DataFrame containing at least the following columns:
    ['电影类别','类型_ext']

  returns
    str: filpath to saved plot
    df: grouped pandas DataFrame
  """
  df = df.copy()

  # plot genre in barh format, stack movie types
  if stacked:
    sns.color_palette("Set2")
    plt.rcParams['figure.figsize'] = [6, 10]
    dfg = df.groupby(['类型_ext', '电影类别']).size().unstack()

    # check which types exist in current publication
    # conform movie types in the designated order
    col_types_ = df['电影类别'].unique()
    col_types = []
    for type_ in ['故事影片', '合拍影片', '动画影片', '纪录影片', '科教影片', '特种影片']:
      if type_ in col_types_:
        col_types.append(type_)
    
    dfg = dfg[col_types]
    dfg = dfg.fillna(0).sort_values(['故事影片'], ascending=True)
    ax = dfg.plot.barh(stacked=True, 
                         color=sns.color_palette("Set2"),
                         grid=True
          )
    ax.set_xlabel('数量',fontsize= 18)
    ax.set_ylabel("类型",fontsize= 18)
    ax.set_title('{}'.format('类型分布'),fontsize= 24, pad=20)
    ax.legend(fontsize=22)
  
  # plot genre in bar format
  else:
    dfg = df.groupby('类型_ext')['类型_ext'].count().rename('数量'
      ).reset_index().sort_values('数量', ascending=False)

    plt.clf()
    plt.rcParams['figure.figsize'] = [10, 4]
    plt.rcParams['axes.facecolor'] = 'white'
    ax = dfg.plot(
        kind = 'bar',
        grid = True,
        fontsize = 22,
        rot = 0,
        color = ['violet'],
    )
    ax.set_title('{}'.format(name),fontsize= 24, pad=20)
    ax.spines['top'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.grid(color='gray', linestyle='-', linewidth=0.5)
    ax.set_xlabel('类型',fontsize= 18)
    ax.set_xticklabels(df_n_genre['类型_ext'], fontsize=12)
    ax.set_ylabel("数量",fontsize= 18)
    ax.legend(fontsize=22)
  
  fp = path_img + '/df_Reg_plot_genre_{}.png'.format(name)
  plt.savefig(fp, bbox_inches='tight')
  plt.show()
  if return_df:
    return fp, dfg
  return fp

#%% generate_word_image
path_font = '/content/drive/MyDrive/Github/Article/fonts/STHUPO.TTF'
path_img = '/content/drive/MyDrive/Github/Article/img'

def generate_word_image(ls_words, 
                        fp_img:str,
                        fp_mask:str=None,
                        fp_prefix:str=None,
                        fp_suffix:str=None,
                        img_width:int=400
                        ):
  if not fp_mask: 
    fp_mask = fp_img

  mask = pil.open(fp_mask)
  #mask.thumbnail([img_width, sys.maxsize], pil.ANTIALIAS)
  image = pil.open(fp_img)
  image.thumbnail(mask.size, pil.ANTIALIAS)
  mask = np.array(mask)
  coloring = np.array(image) # Load Image for coloring
  image_colors = wordcloud.ImageColorGenerator(coloring, default_color=(79, 46, 47))

  wc = wordcloud.WordCloud(
      font_path=path_font,
      width = img_width,
      #height = 100,
      scale = 1,
      mask=mask, # set back ground mask image
      max_words=288,
      max_font_size=188,
      min_font_size=8,
      #mode="RGBA",
      mode="RGB",
      background_color='white', 
      #background_color="rgba(255, 255, 255, 50)", 
      contour_width=3, 
      contour_color='gold',
      repeat=True,
      color_func=image_colors,
  )
  wfreq = Counter(ls_words)
  wc.generate_from_frequencies(wfreq)


  plt.imshow(wc) # 显示词云
  plt.axis('off') # 关闭坐标轴
  plt.show()

  fp_generated_img = '{}/{}_{}_{}_{}.png'.format(
                  path_img,
                  fp_prefix,
                  ls_words[0],
                  fp_img.split('/')[-1].split('.')[0],
                  fp_suffix
                  )

  wc.to_file(fp_generated_img)

  image = pil.open(fp_generated_img)
  image.thumbnail([img_width, sys.maxsize], pil.ANTIALIAS)
  image.save(fp_generated_img)

  return fp_generated_img

#%% preidct_title()

def predict_title(ls_summary,
                  batch_size:int=8,
                 ):
    
    import torch
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
    
    
    i = 0
    L = len(ls_summary)
    predictions = []
    while i < L:
      inputs = tokenizer(ls_summary[i:i+batch_size],
                               padding=True,
                               #max_length=512, 
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
                              clean_up_tokenization_spaces=True
                              ) 
             for g in summary_ids
            ]
      
      predictions.extend(ret)
      i += batch_size
      
    def remove_specials(x):
        x = re.sub(' ', '', x)
        x = re.sub('\[CLS\]', '', x)
        x = re.sub('\[PAD\]', '', x)
        x = re.sub('\[SEP\]', '', x)
        return x
    
    predictions = [remove_specials(p) for p in predictions]
    return predictions

#%% predict_genre()

def predict_genre(ls_summary,
                  batch_size:int=8,
                 ):
    
    # Define Labels
    label2id_genre= {'传奇': 0, '传记': 1, '其它': 2, '军旅': 3, '农村': 4, '宫廷': 5,  
                  '武打': 6, '涉案': 7, '神话': 8, '科幻': 9, '都市': 10, '青少': 11,
                  '革命': 12}
    id2label_genre = {}
    for k, v in label2id_genre.items(): id2label_genre[v] = k
    
    # Genre Classifier
    N_LABELS = len(id2label_genre)
    ID2LABEL = id2label_genre
    LABEL2ID = label2id_genre
    
    from transformers import BertTokenizer, BertForSequenceClassification
    import torch
    ############
    model_name = '/content/drive/MyDrive/Github/Content/tools/models/chinese-bert-wwm-classification-NRTAgenre'
    ############
    if torch.cuda.device_count() > 0:
      device = 'cuda:' + str(torch.cuda.current_device())
    else:
      device = 'cpu'
    
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(
      model_name,
      num_labels=N_LABELS,
      problem_type="multi_label_classification",
      id2label=ID2LABEL,
      label2id=LABEL2ID,  
    ).to(device)
    model.eval() # set model to eval mode for faster prediction
    
    i = 0
    softmax = torch.nn.Softmax(dim=-1)
    L = len(ls_summary)
    predictions = []    
    while i < L:
      batch_test = tokenizer(ls_summary[i:i+batch_size],
                             padding=True,
                             #max_length=512, 
                             truncation=True, 
                             return_tensors='pt'
                            )
      batch_test.to(device)
      batch_outputs = model(**batch_test)
      batch_logtis = batch_outputs.logits
      batch_softmax = softmax(batch_logtis)
      batch_results = torch.argmax(batch_softmax, dim=1).cpu().numpy()
      predictions.extend(list(batch_results))
      i += batch_size
    
    predictions = [id2label_genre[p] for p in predictions]
    return predictions

#%% predict_time()

def predict_time(ls_summary,
                  batch_size:int=8,
                 ):
    
    # Define Labels
    label2id_time = {'古代': 0, '当代': 1, '现代': 2, '近代': 3}
    id2label_time = {}
    for k, v in label2id_time.items(): id2label_time[v] = k
        
    # Genre Classifier
    N_LABELS = len(id2label_time)
    ID2LABEL = id2label_time
    LABEL2ID = label2id_time
    
    from transformers import BertTokenizer, BertForSequenceClassification
    import torch
    ############
    model_name = '/content/drive/MyDrive/Github/Content/tools/models/chinese-bert-wwm-classification-NRTAtime'
    ############
    if torch.cuda.device_count() > 0:
      device = 'cuda:' + str(torch.cuda.current_device())
    else:
      device = 'cpu'
    
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(
      model_name,
      num_labels=N_LABELS,
      problem_type="multi_label_classification",
      id2label=ID2LABEL,
      label2id=LABEL2ID,  
    ).to(device)
    model.eval() # set model to eval mode for faster prediction
    
    i = 0
    softmax = torch.nn.Softmax(dim=-1)
    L = len(ls_summary)
    predictions = []    
    while i < L:
      batch_test = tokenizer(ls_summary[i:i+batch_size],
                             padding=True,
                             #max_length=512, 
                             truncation=True, 
                             return_tensors='pt'
                            )
      batch_test.to(device)
      batch_outputs = model(**batch_test)
      batch_logtis = batch_outputs.logits
      batch_softmax = softmax(batch_logtis)
      batch_results = torch.argmax(batch_softmax, dim=1).cpu().numpy()
      predictions.extend(list(batch_results))
      i += batch_size
    
    predictions = [id2label_time[p] for p in predictions]
    return predictions

#%% id_main_character(ls_summary)
import jieba
import jieba.posseg as pseg
jieba.enable_paddle() #启动paddle模式。 0.40版之后开始支持，早期版本不支持

def find_PER(x: pd.Series):
  # 识别主要角色
  words = pseg.cut(x, use_paddle=True)
  ls_PER = []
  for word, flag in words:
    if flag == 'PER':
      word = word.strip('[\s,，。、“”？；]')
      ls_PER.append(word)
  ls_PER = list(set(ls_PER))

  ls_PER1 = ls_PER.copy()
  ls_PER = []
  while ls_PER1:
    PER =  ls_PER1.pop(0)
    ls_tmp = re.split('[\s,，。、“”？；]', PER)
    ls_PER.extend(ls_tmp)

  ls_PER = list(set(ls_PER)) 
  ls_PER = [PER for PER in ls_PER if len(PER) > 1]
  ls_PER_sorted = sorted(ls_PER, reverse=False, key=lambda x: len(x))

  ans = []
  while ls_PER_sorted:
    PER = ls_PER_sorted.pop(0)
    i = 0
    while i < len(ls_PER_sorted):
      if re.search(PER, ls_PER_sorted[i]):
        ls_PER_sorted.pop(i)
        ans.append(PER)
      i += 1
    if PER not in ans:
      ans.append(PER)

  return ans