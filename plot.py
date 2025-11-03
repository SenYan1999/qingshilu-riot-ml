import json
import pandas as pd
from collections import Counter
import streamlit as st
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from palettable.tableau import *
from utils.location import Pref_Pinyin_dict
st.set_page_config(layout="centered")


plt.rcParams['axes.unicode_minus'] = False
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

def load_trible():
    with open('data/triple_infer_entries.json', 'r') as f:
        data = json.load(f)
    return data

def figure_1(low=1840, high=1911):
    mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=Tableau_20.mpl_colors)
    # Sample data for a bar plot with subclasses
    data = load_trible()

    classes = ['Shunzhi', 'Kangxi', 'Yongzheng', 'Qianlong', 'Jiaqing', 'Daoguang', 'Xianfeng', 'Tongzhi', 'Guangxu', 'Xuantong']
    subclasses = ['Peasant', 'Secret Party', 'Militia']

    # Generating random data for each subclass in each class
    dp = np.zeros((len(classes), len(subclasses)))
    for entry in data:
        dp[classes.index(entry['emperor'])][subclasses.index(entry['RiotType'])] += 1

    # Compute Percentage
    # dp = dp / dp.sum(axis=1, keepdims=True)
    print(dp)
    scale = np.array([17, 62, 13, 60, 25, 29, 11, 13, 34, 3]).reshape(-1, 1)
    dp = dp / scale
    print()
    print(dp)

    # Creating the bar plot with subclasses
    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)

    # Width of a bar
    bar_width = 0.3

    # Setting the position of bars on the X-axis
    r1 = np.arange(len(classes))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]

    # Making the plot
    ax.bar(r1, dp[:, 0], width=bar_width, label=subclasses[0])
    ax.bar(r2, dp[:, 1], width=bar_width, label=subclasses[1].replace('Party', 'Society'))
    ax.bar(r3, dp[:, 2], width=bar_width, label=subclasses[2])

    # Adding labels
    ax.set_ylabel('Avg. Social Unrest per Year', fontsize=12)
    ax.set_title('Social Unrest Breakdown by Emperor Reign', fontsize=14)
    ax.set_xticks([r + bar_width for r in range(len(classes))])
    ax.set_xticklabels(classes)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
              fancybox=True, shadow=True, ncol=5)
    
    fig.savefig('temp/abs_num_unrest_by_reign.png')
    return fig

if __name__ == '__main__':
    figure_1()