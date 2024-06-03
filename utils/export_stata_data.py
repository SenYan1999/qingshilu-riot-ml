import json
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from .location import pref_prov_dict

def export_stata(create_weather_grain=False, grain_idx=0):
    raw_pref2code = pd.read_excel('data/stata/pref2code.xlsx')
    pref2code = {name: code for name, code in zip(raw_pref2code['中俄尼布楚条约待议地区'].tolist(), raw_pref2code['pref_code'].tolist())}
    for pref, code in pref2code.items():
        if str(code) == '.' or str(code).strip() == '':
            pref2code[pref] = -1
    pref2code['中俄尼布楚条约待议地区'] = -1

    pref_dict = {pref: pref2code[pref] for pref in set(pref_prov_dict.keys())}

    # china grain price
    if create_weather_grain:
        raw_grain = pd.read_stata('data/stata/China_pre_post.dta')
        grain_names = ['Wheat_max', 'province', 'Black_bean_mid', 'Millet_mid', 'Rice_husked_mid', 'Wheat_mid', 'Black_bean_max', 'Millet_max', 'Rice_husked_max', 'Black_bean_min', 'Millet_min', 'Rice_husked_min', 'Wheat_min', 'Soybean_mid', 'Soybean_max', 'Soybean_min', 'Rice_1st_mid', 'Rice_2nd_mid', 'Rice_1st_max', 'Rice_2nd_max', 'Rice_1st_min', 'Rice_2nd_min', 'dates', 'Sorghum_mid', 'Sorghum_min', 'Sorghum_max', 'Rice_er_min', 'Rice_er_max', 'Rice_er_mid', 'Rice_3rd_min', 'Rice_3rd_max', 'Rice_3rd_mid']
        pref_year_month_grain = {pref: {year: {month: np.nan for month in range(1, 13)} for year in range(1736, 1912)} for pref in pref2code.values()}
        raw_grain = raw_grain[raw_grain[grain_names[grain_idx]].notna()].reset_index()
        for i in tqdm(range(len(raw_grain)), desc='Parsing Grain Price...'):
            line = raw_grain.iloc[i]
            if line['prefecture'] not in pref2code.values():
                continue
            grain_price = line[grain_names[grain_idx]]
            if pd.isna(line['year']) or pd.isna(line['prefecture']) or pd.isna(grain_price):
                continue
            else:
                try:
                    pref_year_month_grain[int(line['prefecture'])][int(line['year'])][int(line['month'])] = grain_price
                except:
                    raise Exception('Error: %d | %d | %d' % (line['prefecture'], line['year'], line['month']))
        with open(f'data/stata/pref_year_month_{grain_idx}.json', 'wb') as f:
            pickle.dump(pref_year_month_grain, f)
    else:
        with open(f'data/stata/pref_year_month_{grain_idx}.json', 'rb') as f:
            pref_year_month_grain = pickle.load(f)

    # china weather information
    def recode_weather(code):
        if str(code) == '.' or len(str(code).strip()) == 0 or str(code) == 'nan':
            return np.nan
        elif code in [1, 5]:
            return 1
        elif code in [2, 4]:
            return 2
        elif code == 3:
            return 3
        else:
            raise Exception('Error Code' % code)

    if create_weather_grain:
        raw_weather = pd.read_stata('data/stata/pref_weather.dta')
        pref_year_weather = {pref: {year: recode_weather(raw_weather[raw_weather['pref_code'] == pref]['weather_%d' % year].iloc[0]) for year in range(1735, 1912)} for pref in tqdm(raw_weather['pref_code'].tolist(), desc='Parsing Weather Info...')}
        with open('data/stata/pref_year_weather.json', 'wb') as f:
            pickle.dump(pref_year_weather, f)
    else:
        with open('data/stata/pref_year_weather.json', 'rb') as f:
            pref_year_weather = pickle.load(f)

    # riot information
    with open('data/binary_infer_entries.json', 'r') as f:
        binary_data = json.load(f)
    with open('data/triple_infer_entries.json', 'r') as f:
        tri_data = json.load(f)

    years, months, riot_types, prefectures = [], [], [], []
    qing_range = list(range(1644, 1912))

    for entry in tri_data:
        if entry['month'] is None or entry['year'] is None or str(entry['year']) == 'None':
            continue
        elif int(entry['year']) not in qing_range:
            continue
        for prefecture in entry['Prefecture']:
            years.append(entry['year'])
            months.append(entry['month'])
            riot_types.append(entry['RiotType'])
            prefectures.append(prefecture)

    # get month count
    month_count = {}
    for year in qing_range:
        for month in range(1, 13):
            month_count['%d-%d' % (year, month)] = 0
    for year, month, riot_type in zip(years, months, riot_types):
        if riot_type != 'Nan':
            month_count['%d-%d' % (year, month)] += 1
    final_data = {}
    for year in qing_range:
        for month in range(1, 13):
            final_data['%d-%d' % (year, month)] = {pref: {'riot_count': 0, 'jieshe_count': 0, 'peasants_count': 0, 'wuzhuang_count': 0, 'month_counts': month_count['%d-%d' % (year, month)]} for pref in pref2code.values()}

    for year, month, rt, pref in zip(years, months, riot_types, prefectures):
        if rt == 'Nan':
            continue
        pref = pref2code[pref]
        final_data['%d-%d' % (year, month)][pref]['riot_count'] = final_data['%d-%d' % (year, month)][pref]['riot_count'] + 1
        if rt == 'Peasant':
            final_data['%d-%d' % (year, month)][pref]['peasants_count'] = final_data['%d-%d' % (year, month)][pref]['peasants_count'] + 1
        elif rt == 'Secret Party':
            final_data['%d-%d' % (year, month)][pref]['jieshe_count'] = final_data['%d-%d' % (year, month)][pref]['jieshe_count'] + 1
        elif rt == 'Militia':
            final_data['%d-%d' % (year, month)][pref]['wuzhuang_count'] = final_data['%d-%d' % (year, month)][pref]['wuzhuang_count'] + 1
        else:
            raise Exception('Wrong rt %s' % rt)


    all_years, all_months, all_prefs, all_rc, all_jc, all_pc, all_wc, all_mc = [], [], [], [], [], [], [], []
    for key, items in final_data.items():
        year, month = key.split('-')
        for pref, info in items.items():
            all_years.append(year)
            all_months.append(month)
            all_prefs.append(pref)
            all_rc.append(info['riot_count'])
            all_jc.append(info['jieshe_count'])
            all_pc.append(info['peasants_count'])
            all_wc.append(info['wuzhuang_count'])
            all_mc.append(info['month_counts'])
    
    # combine the weather and the grain
    all_weather_last_year, all_weather_current_year, all_grain_current_month, all_grain_min_last_year, all_grain_max_last_year, all_grain_min_this_year, all_grain_max_this_year, all_grain_last_summer, all_grain_this_summer, all_grain_last_winter, all_grain_this_winter, all_grain_last_spring, all_grain_last_fall, all_grain_this_spring, all_grain_this_fall = [], [], [], [], [], [], [], [], [], [], [], [], [], [], []

    def get_grain_min_year(grain):
        grain = [i for i in grain.values() if not pd.isna(i)]
        if len(grain) == 0:
            return np.nan
        return min(grain)
    def get_grain_max_year(grain):
        grain = [i for i in grain.values() if not pd.isna(i)]
        if len(grain) == 0:
            return np.nan
        return max(grain)
    def get_grain_spring(grain):
        grain = [i for m, i in grain.items() if m in [1, 2, 3] and not pd.isna(i)]
        if len(grain) == 0:
            return np.nan
        return np.mean(grain)
    def get_grain_summer(grain):
        grain = [i for m, i in grain.items() if m in [4, 5, 6] and not pd.isna(i)]
        if len(grain) == 0:
            return np.nan
        return np.mean(grain)
    def get_grain_fall(grain):
        grain = [i for m, i in grain.items() if m in [7, 8, 9] and not pd.isna(i)]
        if len(grain) == 0:
            return np.nan
        return np.mean(grain)
    def get_grain_winter(grain):
        grain = [i for m, i in grain.items() if m in [10, 11, 12] and not pd.isna(i)]
        if len(grain) == 0:
            return np.nan
        return np.mean(grain)

    for year, month, pref in tqdm(zip(all_years, all_months, all_prefs), desc='Creating New Dataframe...', total=len(all_years)):
        try:
            year, month, pref = int(year), int(month), int(pref)
        except:
            all_weather_current_year.append(np.nan)
            all_weather_last_year.append(np.nan)
            all_grain_current_month.append(np.nan)
            all_grain_min_last_year.append(np.nan)
            all_grain_min_this_year.append(np.nan)
            all_grain_max_last_year.append(np.nan)
            all_grain_max_this_year.append(np.nan)
            all_grain_last_spring.append(np.nan)
            all_grain_this_spring.append(np.nan)
            all_grain_last_summer.append(np.nan)
            all_grain_this_summer.append(np.nan)
            all_grain_last_fall.append(np.nan)
            all_grain_this_fall.append(np.nan)
            all_grain_last_winter.append(np.nan)
            all_grain_this_winter.append(np.nan)
            continue
        all_weather_current_year.append(pref_year_weather.get(pref, {}).get(year, np.nan))
        all_weather_last_year.append(pref_year_weather.get(pref, {}).get(year-1, np.nan))
        grain = pref_year_month_grain.get(pref, {}).get(year, {})
        grain_last_year = pref_year_month_grain.get(pref, {}).get(year-1, {})
        all_grain_current_month.append(pref_year_month_grain.get(pref, {}).get(year, {}).get(month, np.nan))
        all_grain_min_last_year.append(get_grain_min_year(grain_last_year))
        all_grain_min_this_year.append(get_grain_min_year(grain))
        all_grain_max_last_year.append(get_grain_max_year(grain_last_year))
        all_grain_max_this_year.append(get_grain_max_year(grain))
        all_grain_last_spring.append(get_grain_spring(grain_last_year))
        all_grain_this_spring.append(get_grain_spring(grain))
        all_grain_last_summer.append(get_grain_summer(grain_last_year))
        all_grain_this_summer.append(get_grain_summer(grain))
        all_grain_last_fall.append(get_grain_fall(grain_last_year))
        all_grain_this_fall.append(get_grain_fall(grain))
        all_grain_last_winter.append(get_grain_winter(grain_last_year))
        all_grain_this_winter.append(get_grain_winter(grain))
    
    df = pd.DataFrame({
        'year': all_years,
        'month': all_months,
        'prefecture': all_prefs,
        'month_counts': all_mc,
        'riot_count': all_rc,
        'jieshe_count': all_jc,
        'peasants_count': all_pc,
        'wuzhuang_count': all_wc,
        'weather_last_year': all_weather_last_year,
        'weather': all_weather_current_year,
        'grain_current_month': all_grain_current_month,
        'grain_min_last_year': all_grain_min_last_year,
        'grain_max_last_year': all_grain_max_last_year,
        'grain_min_current_year': all_grain_min_this_year,
        'grain_max_current_year': all_grain_max_this_year,
        'grain_last_spring': all_grain_last_spring,
        'grain_current_spring': all_grain_this_spring,
        'grain_last_summer': all_grain_last_summer,
        'grain_current_summer': all_grain_this_summer,
        'grain_last_fall': all_grain_last_fall,
        'grain_current_fall': all_grain_this_fall,
        'grain_last_winter': all_grain_last_winter,
        'grain_current_winter': all_grain_this_winter,
        })

    suffix = grain_names[grain_idx]
    df['year'].fillna(-1).astype('int64')
    df['prefecture'].fillna(-1).astype('int64')
    df.replace(-1, np.nan, inplace=True)
    df.to_stata(f'data/stata/export/{suffix}_stata_validation_weather_grain_month.dta', write_index=False)

    year_df = df.groupby(['year', 'prefecture']).agg({'riot_count':'sum', 'jieshe_count': 'sum',
                                                'peasants_count': 'sum',
                                                'wuzhuang_count': 'sum',
                                                'weather_last_year': 'first', 'weather':'first',
                                                'grain_min_last_year': 'first', 'grain_max_last_year': 'first',
                                                'grain_min_current_year': 'first', 'grain_max_current_year': 'first',
                                                'grain_last_spring': 'first', 'grain_current_spring': 'first',
                                                'grain_last_summer': 'first', 'grain_current_summer': 'first',
                                                'grain_last_fall': 'first', 'grain_current_fall': 'first',
                                                'grain_last_winter': 'first', 'grain_current_winter': 'first'}).reset_index()
    year_df.to_stata(f'data/stata/export/{suffix}_stata_validation_weather_grain_year.dta', write_index=False)


if __name__ == '__main__':
    '''
    for grain_idx in range(32):
        if grain_idx <= 22:
            continue
        export_stata(create_weather_grain=True, grain_idx=grain_idx)
    '''
    export_stata(create_weather_grain=True, grain_idx=17)

