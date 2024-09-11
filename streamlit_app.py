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

pref_dict = {
    '库伦': '土谢图汗部',
    '吉林': '吉林副都统辖区',
    '永宁直隶州': '叙州府',
    '镇雄州': '昭通府',
    '绥远城厅': '归绥六厅',
    '澄江府': '云南府',
    '郑州': '开封府',
    '钦州': '廉州府',
    '乌里雅苏台': '土谢图汗部',
    '镇边厅': '顺宁府',
    '黑龙江城': ' 黑龙江副都统辖区',
    '讷河直隶厅': '墨尔根副都统辖区',
    '淅川厅': '南阳府',
    '朝阳府': '承德府',
    '固原州': '',
}

def filter_year(year, low=1000, high=2000):
    try:
        year = int(year)
    except:
        return False
    if low <= year and year <= high:
        return True
    else:
        return False

@st.cache_data
def load_binary():
    with open('data/web/filter_binary_infer_entries.json', 'r') as f:
        data = json.load(f)
    return data

@st.cache_data
def load_prefecture_names():
    from pypinyin import pinyin, lazy_pinyin
    data = load_binary()
    prefecture_names = []
    for line in data:
        prefecture_names += line['prefectures']
    prefecture_names = list(set(prefecture_names))
    sorted_prefecture_names = sorted(prefecture_names, key=lambda char: ''.join(lazy_pinyin(char)))
    sorted_prefecture_names_with_pinyin = [name + '(%s)' % ''.join(lazy_pinyin(name)) for name in sorted_prefecture_names]
    return sorted_prefecture_names_with_pinyin


@st.cache_data
def load_trible():
    with open('data/triple_infer_entries.json', 'r') as f:
        data = json.load(f)
    return data

# Load your geospatial data
@st.cache_resource
def load_data(low=1840, high=1900):
    # Replace this with the path to your geospatial dataset
    data = load_binary()
    filtered_data = list(filter(lambda p: filter_year(p['year'], low, high) == True and p['Riot'], data))
    filtered_dp = []
    for line in filtered_data:
        for pref in line['prefectures']:
            filtered_dp.append([pref_dict.get(pref, pref), line['year']])

    data = dict(Counter([line[0] for line in filtered_dp]))
    data = pd.DataFrame(list(data.items()), columns=['NAME_CH', 'Count'])

    # Load the map of China at the province level
    # This requires a shapefile (or similar file) that defines the boundaries of each province
    china_map = gpd.read_file('data/geography/v6_1820_pref_pgn_gbk.shx', encoding='gb18030')
    # china_map = china_map[china_map['NAME_CH'] != '万里长沙']

    Pref_Pinyin_dict['抚州府'] = 'Fuzhou-1 Fu'
    Pref_Pinyin_dict['叙州府'] = 'Xuzhou-1 Fu'
    china_map.loc[(china_map['NAME_PY'] == 'Xuzhou Fu') & (china_map['LEV1_PY'] == 'Sichuan'), 'NAME_PY'] = 'Xuzhou-1 Fu'
    china_map.loc[(china_map['NAME_PY'] == 'Fuzhou Fu') & (china_map['LEV1_PY'] == 'Jiangxi'), 'NAME_PY'] = 'Fuzhou-1 Fu'

    # Assuming you have a DataFrame 'data' with columns 'Province' and 'Value'
    # Merge this data with the china_map GeoDataFrame
    china_map = china_map.merge(data, left_on='NAME_CH', right_on='NAME_CH', how='left')
    china_map = china_map.fillna(0)

    # Plotting the heat map

    return china_map

# Plotting function
def plot_map(low=1840, high=1911):
    china_map = load_data(low=low, high=high)
    topk_lines = china_map.sort_values('Count')
    fig, ax = plt.subplots(figsize=(10, 8))
    china_map.plot(column='Count', ax=ax, legend=True, cmap='YlGnBu')  # 'OrRd' is a colormap, you can choose any
    ax.set_title('Unrest in China, %d to %d' % (low, high))
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    topk = [[topk_lines.iloc[-i-1]['NAME_CH'], topk_lines.iloc[-i-1]['Count']] for i in range(10)]
    return fig, topk

@st.cache_resource
def plot_year_dist_binary(low=1840, high=1911):
    # Load data
    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    data = load_binary()
    data = list(filter(lambda dp: filter_year(dp['year'], low, high) == True, data))
    years = np.arange(low, high+1, 1)  # Years from 2000 to 2020
    points = np.zeros_like(years)
    for entry in data:
        if bool(entry['Riot']) and entry['year'] is not None:
            try:
                points[int(entry['year']) - low] += max(len(entry['prefectures']), 1)
            except:
                print(entry['entry'])

    # Creating the plot
    ax.plot(years, points, marker='None', color='b', linestyle='-')
    ax.set_title("Unrest Entries Over Years (%d - %d)" % (low, high), fontsize=14)
    ax.set_xlabel("Years", fontsize=12)
    ax.set_ylabel("Number of Unrest Entries", fontsize=12)
    ax.xaxis.set_ticks(np.arange(low, high, 5))
    ax.xaxis.set_tick_params(rotation=90)
    plt.grid(True)
    return fig


@st.cache_resource
def plot_year_dist_three_classes(low=1840, high=1911):
    mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=Tableau_20.mpl_colors)
    # Sample data for a bar plot with subclasses
    data = load_trible()

    classes = ['Shunzhi', 'Kangxi', 'Yongzheng', 'Qianlong', 'Daoguang', 'Jiaqing', 'Tongzhi', 'Xianfeng', 'Guangxu', 'Xuantong']
    subclasses = ['Peasant', 'Secret Party', 'Militia']

    # Generating random data for each subclass in each class
    dp = np.zeros((len(classes), len(subclasses)))
    for entry in data:
        dp[classes.index(entry['emperor'])][subclasses.index(entry['RiotType'])] += 1

    # Compute Percentage
    dp = dp / dp.sum(axis=1, keepdims=True)


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
    ax.bar(r2, dp[:, 1], width=bar_width, label=subclasses[1])
    ax.bar(r3, dp[:, 2], width=bar_width, label=subclasses[2])

    # Adding labels
    ax.set_ylabel('Share of All Unrest Entries in Emperors\' Reign', fontsize=12)
    ax.set_title('Unrest Entries Breakdown by Emperor Reign', fontsize=14)
    ax.set_xticks([r + bar_width for r in range(len(classes))])
    ax.set_xticklabels(classes)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
              fancybox=True, shadow=True, ncol=5)
    return fig


def main():
    st.title('Qing Shi Lu')
    if 'selected_option' not in st.session_state:
        st.session_state.selected_option = 'A'

    # Sidebar with menu options
    with st.sidebar:
        st.write("Menu")
        if st.button('Display the diagrams'):
            st.session_state.selected_option = 'A'
        if st.button('Explore the unrest entries'):
            st.session_state.selected_option = 'B'
    
    if st.session_state.selected_option == 'A':

        # display maps
        map_table, count_table, three_class_table = st.tabs(['Heat Map', 'Unrest Year Dist', 'Unrest Three Classes Dist'])
        with map_table:
            low_year = st.number_input("Low Year", value=1636, min_value=1636, max_value=1912, step=1, format="%i", key='low_map')
            high_year = st.number_input("High Year", value=1911, min_value=1636, max_value=1912, step=1, format="%i", key='high_map')
            fig, topk = plot_map(low=low_year, high=high_year)
            st.pyplot(fig)
            st.markdown('**Top 10 Prefectures with Most Unrest**')
            for line in topk:
                if line[0] and line[1]:
                    st.markdown('**%s**: %4d' % (line[0], line[1]))
        
        with count_table:
            low_year = st.number_input("Low Year", value=1636, min_value=1636, max_value=1912, step=1, format="%i", key='low_binary')
            high_year = st.number_input("High Year", value=1911, min_value=1636, max_value=1912, step=1, format="%i", key='high_binary')
            st.pyplot(plot_year_dist_binary(low=low_year, high=high_year))
        
        with three_class_table:
            st.pyplot(plot_year_dist_three_classes())

    elif st.session_state.selected_option == 'B':
        # filter and display information
        data = load_binary()
        low_year = st.number_input("Low Year", value=1636, min_value=1636, max_value=1912, step=1, format="%i", key='low_text')
        high_year = st.number_input("High Year", value=1911, min_value=1636, max_value=1912, step=1, format="%i", key='high_text')
        str_search = st.text_input('Query the unrest entries based on the keyword', value='')
        prefecture_names = load_prefecture_names()
        prefecture = st.selectbox('Please select the prefecture', ["All Prefecture"] + prefecture_names)
        data = list(filter(lambda x: x['Riot'] == True, data))
        data = list(filter(lambda x: filter_year(x['year'], low_year, high_year) == True, data))
        if prefecture != 'All Prefecture':
            prefecture_data = list(filter(lambda x: prefecture.split('(')[0] in x['prefectures'], data))
            prefecture_data = list(filter(lambda x: str_search in x['entry'], prefecture_data))
        else:
            prefecture_data = data
        
        num_row = 10
        pages = len(prefecture_data) // num_row + 1
        page = st.selectbox('Please select the page', [i + 1 for i in range(pages)])
        page_data = prefecture_data[(page-1) * num_row: page * num_row]

        st.markdown('**Number of unrest entries in %s (%d - %d):** %7d' % (prefecture, low_year, high_year, len(prefecture_data)))
        st.markdown('**Number of unrest entries in total (%d - %d):** %7d' % (low_year, high_year, len(data)))

        for line in page_data:
            entry, emperor, year, chinese_year, prefecture_list = st.columns([4, 1, 1, 1, 1])
            with entry:
                st.write(line['entry'])
            with emperor:
                st.write(line['emperor'])
            with year:
                st.write(line['year'])
            with chinese_year:
                st.write(line['chinese_year'])
            with prefecture_list:
                st.write('  \n  '.join(line['prefectures']))

if __name__ == "__main__":
    main()
