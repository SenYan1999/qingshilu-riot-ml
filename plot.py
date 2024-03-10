import json
import pickle
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import tqdm
from palettable.tableau import *


mpl.rcParams['font.family'] = ['serif']
mpl.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42


def plot_year_dist_binary():
    # Load data
    with open('data/binary_infer_entries.json', 'r') as f:
        data = json.load(f)
    years = np.arange(1636, 1911, 1)  # Years from 2000 to 2020
    points = np.zeros_like(years)
    for entry in data:
        if bool(entry['Riot']) and entry['year'] is not None:
            try:
                points[entry['year'] - 1636] += 1
            except:
                print(entry['entry'])

    # Creating the plot
    plt.figure(figsize=(10, 6), dpi=1000)
    plt.plot(years, points, marker='None', color='b', linestyle='-')
    plt.title("Riots Over Years (1636 - 1911)", fontsize=14)
    plt.xlabel("Years", fontsize=12)
    plt.ylabel("Number of Riots", fontsize=12)
    plt.xticks(np.arange(1636, 1911, 5), rotation=90)
    plt.grid(True)

    # Show the plot
    plt.savefig('diagrams/year_dist_binary.png')

def plot_year_dist_three_classes():
    mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=Tableau_20.mpl_colors)
    # Sample data for a bar plot with subclasses
    with open('data/data.pkl', 'rb') as f:
        raw_data = pickle.load(f)
    with open('data/triple_infer_entries.json', 'r') as f:
        data = json.load(f)

    classes = raw_data.emperor_list
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
    ax.set_ylabel('Share of All Riots in Emperors\' Reign', fontsize=12)
    ax.set_title('Riot Breakdown by Emperor Reign', fontsize=14)
    ax.set_xticks([r + bar_width for r in range(len(classes))])
    ax.set_xticklabels(classes)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
              fancybox=True, shadow=True, ncol=5)

    # Show the plot
    plt.savefig('diagrams/year_dist_three_classes_bar.png')

def plot_chinese_heatmap(low=1000, high=2000, use_pre_data=True):
    import pinyin
    import pandas as pd
    import geopandas as gpd
    import matplotlib.pyplot as plt
    from utils.location import LocationIdentify, Pref_Pinyin_dict
    from collections import Counter

    if not use_pre_data:
        new_entries = []
        with open('data/binary_infer_entries.json', 'r') as f:
            data = json.load(f)
        pref2twn = {}
        provinces = []
        for entry in tqdm(data):
            if bool(entry['Riot']):
                current_p = LocationIdentify(entry['entry'])
                provinces += [(Pref_Pinyin_dict[p], entry['year']) for p in current_p]
                entry['Prefecture'] = current_p
            new_entries.append(entry)
        with open('data/binary_infer_pref.json', 'w') as f:
            json.dump(provinces, f)
        with open('data/binary_infer_entries.json', 'w') as f:
            json.dump(new_entries, f)

    with open('data/binary_infer_pref.json', 'r') as f:
        provinces = json.load(f)

    provinces = [(line[0], line[1]) for line in provinces]

    def filter_year(year, low=1000, high=2000):
        try:
            year = int(year)
        except:
            return False
        if low <= year and year <= high:
            return True

    filtered_dp = list(filter(lambda p: filter_year(p[1], low, high), provinces))

    data = dict(Counter([line[0] for line in filtered_dp]))
    data = pd.DataFrame(list(data.items()), columns=['NAME_PY', 'Count'])

    # Load the map of China at the province level
    # This requires a shapefile (or similar file) that defines the boundaries of each province
    china_map = gpd.read_file('data/geography/v6_1820_pref_pgn_gbk.shx')

    # same pinyin
    

    # Assuming you have a DataFrame 'data' with columns 'Province' and 'Value'
    # Merge this data with the china_map GeoDataFrame
    china_map = china_map.merge(data, left_on='NAME_PY', right_on='NAME_PY')
    china_map.to_csv('data/china_map.csv')
    print()

    '''
    # Plotting the heat map
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    china_map.plot(column='Count', ax=ax, legend=True, cmap='YlGnBu')  # 'OrRd' is a colormap, you can choose any

    plt.title(f'Riots in China, {low} to {high}')
    plt.savefig(f'diagrams/{low}-{high}_riot_china_heatmap.png')
    '''

def table_print_year_emperor():
    import tabulate

    with open('data/data.pkl', 'rb') as f:
        raw_data = pickle.load(f)
    with open('data/binary_infer_entries.json', 'r') as f:
        entries = json.load(f)

    data = {e: {'Riots': 0, 'Events': 0} for e in raw_data.emperor_list}
    for entry in entries:
        if bool(entry['Riot']):
            data[entry['emperor']]['Riots'] += 1
        data[entry['emperor']]['Events'] += 1

    headers = ['Emperor', 'Riots', 'Events', 'Ratio %']
    rows = [[e, data[e]['Riots'], data[e]['Events'], '%.2f' % float(data[e]['Riots'] / data[e]['Events'] * 100)] \
            for e in raw_data.emperor_list]

    with open('diagrams/emperor_reign_ratio.txt', 'w') as f:
        f.write(tabulate.tabulate(rows, headers, tablefmt="presto"))

    with open('diagrams/emperor_reign_ratio.csv', 'w') as f:
        f.write(','.join(headers) + '\n')
        for row in rows:
            f.write(','.join([str(r) for r in row]) + '\n')


if __name__ == '__main__':
    # plot_year_dist_binary()
    # plot_year_dist_three_classes()
    # plot_chinese_heatmap(low=1840, high=1860, use_pre_data=True)
    # plot_chinese_heatmap(low=1680, high=1700, use_pre_data=True)
    # plot_chinese_heatmap(low=1636, high=1911, use_pre_data=True)
    table_print_year_emperor()
