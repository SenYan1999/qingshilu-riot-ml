import json
import calendar
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm


# TODO: maybe we need to count the non-riot entry for the month? Currently, only riot entry are counted
# TODO: need to check with Carol about the regression
# TODO: warning when run the program

def export_dta():
    with open('data/triple_infer_entries.json', 'r') as f:
        data = json.load(f)

    # load pref2code
    raw_pref_index = pd.read_excel('data/Loc_Pinyin.xlsx')
    pref2code = {}
    for pref, code in zip(raw_pref_index['中俄尼布楚条约待议地区'], raw_pref_index['pref_code']):
        if str(code).replace('.', '').strip() == '':
            # pref2code[pref.strip()] = ''
            continue
        pref2code[pref.strip()] = str(code)
        # print('%10s: %s' % (pref.strip(), str(code).replace('.', '').strip()))
    code2pref = {code: pref for pref, code in pref2code.items()}
    
    # check prefs
    prefs = set()
    for line in data:
        prefs.update(line['Prefecture'])
    not_in_stata = [p for p in prefs if p not in pref2code]
    not_in_riot = [p for p in pref2code.keys() if p not in prefs]
    print('In Entries, Not In Loc_Pinyin.xlsx')
    print(';'.join(not_in_stata))
    print()
    print('In Loc_Pinyin.xlsx, Not in Entries')
    print(';'.join(not_in_riot))

    # create new pref_weather dta TODO: change 0 to nan if not found
    pref_year_count = {pref2code[pref]: {y: None for y in range(1643, 1913)} for pref in prefs if pref not in not_in_stata}
    type_idx = {'Secret Party': 0, 'Peasant': 1, 'Militia': 2}
    for line in data:
        if not line['Riot']:
            continue
        year, month = str(line['year']).replace('nan', '').replace('None', '').strip(), str(line['month']).replace('nan', '').replace('None', '').strip()
        if year == '' or month == '' or int(year) > 1912 or int(year) < 1643:
            continue
        year, month = int(year), int(month)
        for pref in line['Prefecture']:
            if pref not in pref2code:
                continue
            if pref_year_count[pref2code[pref]][year] == None:
                pref_year_count[pref2code[pref]][year] = [set([month]), 1, 0, 0, 0]
                pref_year_count[pref2code[pref]][year][type_idx[line['RiotType']]+2] += 1
            else:
                pref_year_count[pref2code[pref]][year][0].update([month])
                pref_year_count[pref2code[pref]][year][1] += 1
                pref_year_count[pref2code[pref]][year][type_idx[line['RiotType']]+2] += 1

    # save to stata
    prefectures, provinces, pref_codes, years, s_months, riot_counts, j_counts, p_counts, w_counts, ms, mcs = [], [], [], [], [], [], [], [], [], [], []
    for pref_code, all_year in pref_year_count.items():
        for year, year_info in all_year.items():
            if year_info != None:
                prefectures.append(code2pref[pref_code])
                pref_codes.append(pref_code)
                years.append(year)
                s_months.append(';'.join([calendar.month_name[m] for m in year_info[0]]))
                riot_counts.append(year_info[1])
                j_counts.append(year_info[2])
                p_counts.append(year_info[3])
                w_counts.append(year_info[4])
                mcs.append(len(year_info[0]))
    df = pd.DataFrame({'prefecture': prefectures, 'province': provinces, 'pref_code': pref_codes, 'year': years, 's_month': s_months, 'riot_count': riot_counts, 'Jieshe_count': j_counts, 'Peasant_count': p_counts, 'Wuzhuang_count': w_counts, 'month': ms, 'month_counts': mcs})
    df.to_stata('data/validation/guwen_bert_month_3topics.dta.dta')

def validate_existing_paper():
    def normalize(raw):
        # norm = [float(i)/sum(raw) for i in raw]
        norm = [float(i)/max(raw) for i in raw]
        return norm

    # only 1740 - 1839
    chan = normalize(pd.read_excel('existing_lit/chan_mass_disturbances.xlsx', index_col=0).sum(axis=0).tolist()) # [1796 ~ 1911]
    ho = normalize(pd.read_csv('existing_lit/ho_fung_hung.csv')['riot-count'].tolist()) # 1740 - 1839
    jia = normalize(pd.read_stata('existing_lit/jia.dta')['riot_count_4'].tolist()) # 1470 - 1900
    miller = normalize(pd.read_stata('existing_lit/miller_topics_sample.dta')['riot_count_2a'].tolist()) # 1723 - 1911

    with open('data/binary_infer_entries.json', 'r') as f:
        data = json.load(f)
    years = np.arange(1636, 1911, 1)  # Years from 2000 to 2020
    points = np.zeros_like(years)
    for entry in data:
        if bool(entry['Riot']) and entry['year'] is not None:
            try:
                points[entry['year'] - 1636] += 1
            except:
                continue
    sen = normalize(points)

    ws = 5
    chan = pd.Series(chan[:-(1911 - 1839)]).rolling(window=ws).mean()
    ho = pd.Series(ho[1796-1740:]).rolling(window=ws).mean()
    jia = pd.Series(jia[1796-1470:-(1900 - 1839)]).rolling(window=ws).mean()
    miller = pd.Series(miller[1796-1723-1:-(1911 - 1839)]).rolling(window=ws).mean()
    sen = pd.Series(sen[1796-1636: -(1911 - 1840)]).rolling(window=ws).mean()

    sns.lineplot(x=range(1796, 1840), y=chan, label='Mass Disturbances, Chan')
    sns.lineplot(x=range(1796, 1840), y=ho, label='Engaging / Non-Engaging, Ho Fung')
    sns.lineplot(x=range(1796, 1840), y=jia, label='Events, Jia')
    sns.lineplot(x=range(1796, 1840), y=miller, label='Bandit Topic, Miller')
    sns.lineplot(x=range(1796, 1840), y=sen, label='GUWEN-BERT Classifier')
    plt.legend()
    plt.savefig('diagrams/cross_validation_smothing_moving_avg_ws_5.png', dpi=500)

def export_dta_template():
    with open('data/triple_infer_entries.json', 'r') as f:
        data = json.load(f)

    # load pref2code
    raw_pref_index = pd.read_excel('data/Loc_Pinyin.xlsx')
    pref2code = {}
    for pref, code in zip(raw_pref_index['中俄尼布楚条约待议地区'], raw_pref_index['pref_code']):
        if str(code).replace('.', '').strip() == '':
            # pref2code[pref.strip()] = ''
            continue
        pref2code[pref.strip()] = str(code)
        # print('%10s: %s' % (pref.strip(), str(code).replace('.', '').strip()))
    code2pref = {code: pref for pref, code in pref2code.items()}
    
    # check prefs
    prefs = set()
    for line in data:
        prefs.update(line['Prefecture'])
    not_in_stata = [p for p in prefs if p not in pref2code]
    not_in_riot = [p for p in pref2code.keys() if p not in prefs]
    print('In Entries, Not In Loc_Pinyin.xlsx')
    print(';'.join(not_in_stata))
    print()
    print('In Loc_Pinyin.xlsx, Not in Entries')
    print(';'.join(not_in_riot))

    # create new pref_weather dta TODO: change 0 to nan if not found
    pref_year_count = {pref2code[pref]: {y: None for y in range(1643, 1913)} for pref in prefs if pref not in not_in_stata}
    type_idx = {'Secret Party': 0, 'Peasant': 1, 'Militia': 2}
    for line in data:
        if not line['Riot']:
            continue
        year, month = str(line['year']).replace('nan', '').replace('None', '').strip(), str(line['month']).replace('nan', '').replace('None', '').strip()
        if year == '' or month == '' or int(year) > 1912 or int(year) < 1643:
            continue
        year, month = int(year), int(month)
        for pref in line['Prefecture']:
            if pref not in pref2code:
                continue
            if pref_year_count[pref2code[pref]][year] == None:
                pref_year_count[pref2code[pref]][year] = [set([month]), 1, 0, 0, 0]
                pref_year_count[pref2code[pref]][year][type_idx[line['RiotType']]+2] += 1
            else:
                pref_year_count[pref2code[pref]][year][0].update([month])
                pref_year_count[pref2code[pref]][year][1] += 1
                pref_year_count[pref2code[pref]][year][type_idx[line['RiotType']]+2] += 1

    # load template
    df = pd.read_stata('data/validation/month_3topics_10_18.dta')
    for i in tqdm(range(len(df)), 'Iterating Rows in Stata Template...'):
        line = df.iloc[i]
        if str(line['pref_code']) not in pref_year_count:
            # print(line['pref_code'])
            # only -1 and 322
            continue
        year_info = pref_year_count[str(line['pref_code'])][int(line['year'])]
        if year_info != None:
            df.iloc[i]['s_month'] = ';'.join([calendar.month_name[m] for m in year_info[0]])
            df.iloc[i, df.columns.get_loc('riot_count')] = year_info[1]
            df.iloc[i, df.columns.get_loc('Jieshe_count')] = year_info[2]
            df.iloc[i, df.columns.get_loc('Peasants_count')] = year_info[3]
            df.iloc[i, df.columns.get_loc('Wuzhuang_count')] = year_info[4]
            df.iloc[i, df.columns.get_loc('month')] = list(year_info[0])[-1]
            df.iloc[i, df.columns.get_loc('month_counts')] = len(year_info[0])
        else:
            df.iloc[i]['s_month'] = np.nan
            df.iloc[i, df.columns.get_loc('riot_count')] = 0
            df.iloc[i, df.columns.get_loc('Jieshe_count')] = 0
            df.iloc[i, df.columns.get_loc('Peasants_count')] = 0
            df.iloc[i, df.columns.get_loc('Wuzhuang_count')] = 0
            # df.iloc[i, df.columns.get_loc('month')] = np.nan
            # df.iloc[i, df.columns.get_loc('month_counts')] = np.nan
    df.to_stata('data/validation/guwen_bert_month_3topics.dta')

if __name__ == '__main__':
    # export_dta()
    export_dta_template()
 