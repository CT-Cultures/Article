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

#%% generaate_word_image

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

#%% Predict Title

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
      

