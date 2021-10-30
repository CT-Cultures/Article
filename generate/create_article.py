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

import jieba.analyse
tfidf = jieba.analyse.extract_tags
textrank = jieba.analyse.textrank
#%% Import local libraries
import utils

#%% Set Path
# Set Path
path_wd = '/content/drive/MyDrive/Github/Article'
path_font = '/content/drive/MyDrive/Github/Article/fonts/STHUPO.TTF'
path_img = '/content/drive/MyDrive/Github/Article/img'
path_reg = '/content/drive/MyDrive/Github/Content/sources/ChinaFilm'

#%% Import Registration
df = pd.read_json(path_reg + '/records/contents_of_registrations.json')
df = dy_reg.Refined_Records(df)

#%% Identify Current Issue
curr_issue_name = df.sort_values(['公示日期', '公示批次名称'], ascending=False)['公示批次名称'].iloc[0]
df_curr =  df.loc[df.公示批次名称 == curr_issue_name]
issue_name = df_curr['公示批次起始'].iloc[0][0] + '年' + df_curr['公示批次起始'].iloc[0][1] +'月'
issue_name += df_curr['公示批次起始'].iloc[0][2]
print('Current Issue: {}'.format(issue_name))

#%% Apply Models
df_curr['预测片名'] = utils.predict_title(df_curr['梗概'].tolist())
df_curr['类型'] = utils.predict_genre(df_curr['梗概'].tolist())
df_curr['年代'] = utils.predict_time(df_curr['梗概'].tolist())
df_curr['kw'] = df_curr['梗概'].apply(textrank, topK=10)
df_curr['主要角色'] = df_curr['梗概'].apply(utils.find_PER)

df_curr = pd.read_pickle(path_wd + '/records/df_reg_{}.pkl'.format(issue_name))

#%% T1
# Title
T1 = '{year}年'.format(year=df_curr.iloc[0]['公示批次起始'][0])
T1 += '{month}月'.format(month=df_curr.iloc[0]['公示批次起始'][1])
if df_curr.iloc[0]['公示批次起始'][2] != '整月':
  T1 += '{duration}'.format(duration=df_curr.iloc[0]['公示批次起始'][2])
T1 += '电影备案公示划重点'

#%% P1
# Obtain Variables
df_curr.loc[:,'公示日期'] = df_curr.loc[:,'公示日期'].astype('datetime64')
pub_year = df_curr.iloc[0]['公示日期'].year
pub_month = df_curr.iloc[0]['公示日期'].month
pub_day = df_curr.iloc[0]['公示日期'].day
df_curr['备案申请年份'] = df_curr['备案申请年份'].astype('int')
df_curr_sorted = df_curr.sort_values(
    ['备案申请年份', '备案立项年度顺序号'], ascending=True
).reset_index(drop=True)
df_type = df_curr.groupby('电影类别')['电影类别'].count().sort_values(
    ascending=False).rename('数量').to_frame().reset_index()
# Write Content
P1 = ''
P1 += '{year}年{month}月{day}日，'.format(year=pub_year, month=pub_month, day=pub_day)
P1 += '{month}月{part_of_month}的备案公示新鲜出炉，'.format(
    month=df_curr.iloc[0]['公示批次起始'][1], part_of_month=df_curr.iloc[0]['公示批次起始'][2])
P1 += '共计影片{}部！'.format(df_curr.shape[0])
P1 += '这一批次中，最遥远的项目是《{}》，'.format(df_curr_sorted.loc[0, '片名']) 
P1 += '备案号为{}，'.format(df_curr_sorted.loc[0, '备案立项号'])
P1 += '最近期的项目是《{}》，'.format(df_curr_sorted.loc[df_curr.shape[0]-1, '片名'])
P1 += '备案号为{}。'.format(df_curr_sorted.loc[df_curr.shape[0]-1, '备案立项号'])

#%% P2
df_n_type = df_curr.groupby('电影类别')['电影类别'].count().rename('数量'
  ).reset_index().sort_values('数量', ascending=False)

plt.clf()
plt.rcParams['figure.figsize'] = [6, 3.5]
plt.rcParams['axes.facecolor'] = 'white'
ax = df_n_type.plot(
    kind = 'bar',
    grid = True,
    fontsize = 22,
    rot = 0,
    color = ['violet'],
)
ax.set_title("按类别划分",fontsize= 24, pad=20)
ax.spines['top'].set_color('black')
ax.spines['bottom'].set_color('black')
ax.spines['left'].set_color('black')
ax.spines['right'].set_color('black')
ax.grid(color='gray', linestyle='-', linewidth=0.5)
ax.set_xlabel('电影类别',fontsize= 18)
ax.set_xticklabels(df_n_type['电影类别'], fontsize=12)
ax.set_ylabel("数量",fontsize= 18)
ax.legend(fontsize=22)

fp_plot_type = path_img + '/df_Reg_plot_type_{}.png'.format(issue_name)
plt.savefig(fp_plot_type, bbox_inches='tight')

P2 = '按备案类别划分本次完成备案的共计'
for i, row in df_type.iterrows():
  if i == df_type.shape[0]-1:
    P2 = P2.rstrip('、')
    P2 += '以及{type}{n}部。'.format(type=row['电影类别'], n=row['数量'])
  else:
    P2 += '{type}{n}部、'.format(type=row['电影类别'], n=row['数量'])
#%% P3
df_n_genre = df_curr.groupby('修正类型')['修正类型'].count().rename('数量'
  ).reset_index().sort_values('数量', ascending=False)

plt.clf()
plt.rcParams['figure.figsize'] = [6, 3.5]
plt.rcParams['axes.facecolor'] = 'white'
ax = df_n_genre.plot(
    kind = 'bar',
    grid = True,
    fontsize = 22,
    rot = 0,
    color = ['violet'],
)
ax.set_title("按类型划分",fontsize= 24, pad=20)
ax.spines['top'].set_color('black')
ax.spines['bottom'].set_color('black')
ax.spines['left'].set_color('black')
ax.spines['right'].set_color('black')
ax.grid(color='gray', linestyle='-', linewidth=0.5)
ax.set_xlabel('类型',fontsize= 18)
ax.set_xticklabels(df_n_genre['修正类型'], fontsize=12)
ax.set_ylabel("数量",fontsize= 18)
ax.legend(fontsize=22)

fp_plot_genre = path_img + '/df_Reg_plot_genre_{}.png'.format(issue_name)
plt.savefig(fp_plot_genre, bbox_inches='tight')


P3 = ''
P3 = '按类型划分，都市题材最多,'
df_genre_sorted = df_curr.groupby('类型')['片名'].count().sort_values(ascending=False).reset_index()
df_genre_sorted.columns = ['类型', '数量']

P3 += '共计{}部。'.format(df_genre_sorted['数量'][0])
#%% P4
df_n_time = df_curr.groupby('年代')['年代'].count().rename('数量'
  ).reset_index().sort_values('数量', ascending=False)

plt.clf()
plt.rcParams['figure.figsize'] = [6, 3.5]
plt.rcParams['axes.facecolor'] = 'white'
ax = df_n_time.plot(
    kind = 'bar',
    grid = True,
    fontsize = 22,
    rot = 0,
    color = ['violet'],
)
ax.set_title("按年代划分",fontsize= 24, pad=20)
ax.spines['top'].set_color('black')
ax.spines['bottom'].set_color('black')
ax.spines['left'].set_color('black')
ax.spines['right'].set_color('black')
ax.grid(color='gray', linestyle='-', linewidth=0.5)
ax.set_xlabel('年代',fontsize= 18)
ax.set_xticklabels(df_n_time['年代'], fontsize=12)
ax.set_ylabel("数量",fontsize= 18)
ax.legend(fontsize=22)

fp_plot_time = path_img + '/df_Reg_plot_time_{}.png'.format(issue_name)
plt.savefig(fp_plot_time, bbox_inches='tight')

P4 = ''
P4 = '按年代划分，当代题材占主力位置,'
df_time_sorted = df_curr.groupby('年代')['片名'].count().sort_values(ascending=False).reset_index()
df_time_sorted.columns = ['年代', '数量']

P4 += '共计{}部。'.format(df_time_sorted['数量'][0])

#%% P5

#%% P6

#%%




