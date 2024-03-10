import re
import glob
import pandas as pd


def check_if_six_classes_in_chen():
    with open('data/annotation_chen/Train_riots.txt', 'r') as f:
        data = [line.strip().replace('○', '') for line in f.readlines()]

    with open('data/annotation_chen/Train_nonriots.txt', 'r') as f:
        data += [line.strip().replace('○', '') for line in f.readlines()]

    for file in glob.glob('data/sixclasses/*.txt'):
        riot_type = file.split('/')[-1].replace('.txt', '').lower()
        fail, count = 0, 0
        with open(file, 'r') as f:
            for line in f.readlines():
                line = line.strip().replace('○', '')
                line = re.sub(r'\d+', '', line).strip()
                if len(line) == 0:
                    continue
                entries = [i for i, x in enumerate(data) if x[:100] == line[:100]]
                # entries = list(filter(lambda entry: entry[:100] == line[:100], data))
                if len(entries) == 0:
                    print(line)
                    fail += 1
                else:
                    a = 0
                count += 1
        print('Fail Rate: %2d / %2d = %.2f' % (fail, count, fail / count))

def check_three_classes_dataset():
    from collections import Counter
    from utils.dataset import ThreeClassesDataset
    data = ThreeClassesDataset('data/data.pkl')
    print(Counter([dp[1] for dp in data]))

def debug_location_extraction():
    from utils.location import LocationIdentify
    entry = '庚午。摄政和硕睿亲王师次辽河地方。以军事咨洪承畴。承畴上启曰。我兵之强天下无敌将帅同心。步伍整肃。流寇可一战而除。宇内可计日而定矣。今宜先遣官宣布王令。示以此行。特扫除乱逆、期于灭贼。有抗拒者必加诛戮。不屠人民不焚庐舍不掠财物之意。仍布告各府州县、有开门归降者官则加升。军民秋毫无犯若抗拒不服者城下之日官吏诛。百姓仍予安全。有首倡内应立大功者则破格封赏法在必行。此要务也况流寇初起时遇弱则战。遇强则遁。今得京城。财足志骄。已无固志。一旦闻我军至、必焚其宫殿府库。遁而西行贼之骡马不下三十余万。昼夜兼程可二三百里及我兵抵京贼已远去。财物悉空逆恶不得除。士卒无所获亦大可惜也。今宜计道里。限时日。辎重在后。精兵在前。出其不意。从蓟州密云近京处、疾行而前。贼走、则即行追巢□刀。傥仍坐据京城以拒我则伐之更易。如此庶逆贼扑灭、而神人之怒可回。更收其财畜、以赏士卒。殊有益也初明之守边者兵弱马疲。犹可轻入今恐贼遣精锐伏于山谷狭处、以步兵扼路。我国。骑兵不能履险宜于骑兵内选作步兵。从高处觇其埋伏俾步兵在前。骑兵在后。比及入边则步兵皆骑兵也孰能御之。若沿边仍复空虚则接踵而进不劳余力。抵京之日我兵连营城外侦探勿绝。庶可断陕西宣府大同真保诸路、以备来攻。则马首所至、计日功成矣流寇十余年来用兵已久。虽不能与大军相拒。亦未可以昔日汉兵轻视之也。'
    print(LocationIdentify(entry, {})[0])

def plot_pref2all():
    from utils.location import pref_pref_dict, cnty_pref_dict, twn_pref_dict
    level_list = [pref_pref_dict, cnty_pref_dict, twn_pref_dict]
    pref2all = {}
    for i, level in enumerate(level_list):
        for key, value in level.items():
            if value not in pref2all:
                pref2all[value] = [(key, i)]
            else:
                pref2all[value].append((key, i))
    return pref2all

def plot_yili():
    from streamlit_app import load_binary, load_prefecture_names
    data = load_binary()
    prefecture_names = load_prefecture_names()
    prefecture = '伊犁'
    data = list(filter(lambda x: x['Riot'], data))
    print(len(data))
    prefecture_data = list(filter(lambda x: prefecture in x['Prefecture'], data))
    print(len(prefecture_data))

def print_matched_loc():
    import json
    from tqdm import tqdm
    from collections import Counter
    from utils.location import LocationIdentify, LocationIdentifyDebug, debug
    with open('data/binary_infer_entries.json', 'r') as f:
        data = json.load(f)

    print(debug)
    new_entries = []
    for entry in tqdm(data):
        if bool(entry['Riot']):
            current_p = LocationIdentify(entry['entry'])
            entry['Prefecture'] = current_p
        new_entries.append(entry)
    print(Counter(debug).most_common(10))

    with open('data/binary_infer_entries.json', 'w') as f:
        json.dump(new_entries, f)
    with open('data/nm_loc.json', 'w') as f:
        json.dump(debug, f)

def print_pref_twn_count():
    import json
    from tqdm import tqdm
    from collections import Counter
    from utils.location import LocationIdentifyDebug, debug
    '''
    with open('data/binary_infer_entries.json', 'r') as f:
        data = json.load(f)

    print(debug)
    new_entries = []
    pref2twn = {}
    for entry in tqdm(data):
        if bool(entry['Riot']):
            current_p = LocationIdentifyDebug(entry['entry'], pref2twn=pref2twn)
            entry['Prefecture'] = current_p[0]
        new_entries.append(entry)
    print(Counter(debug).most_common(10))

    with open('data/binary_infer_entries.json', 'w') as f:
        json.dump(new_entries, f)

    with open('data/nm_loc.json', 'w') as f:
        json.dump(debug, f)
    
    with open('data/pref2twn.json', 'w') as f:
        json.dump(pref2twn, f)
    '''

    with open('data/pref2twn.json', 'r') as f:
        pref2twn = json.load(f)
    
    while True:
        pref = input('Enter the prefecture name:')
        if pref not in pref2twn:
            print('We dont have this prefecture')
        else:
            print(Counter(pref2twn[pref]).most_common(10))



def modify_pref_gpd():
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
        '镇边厅': '',
        '黑龙江城': '',
        '讷河直隶厅': '',
        '淅川厅': '',
        '朝阳府': '',
        '固原州': '',
    }

def gpd_miss_prefs():
    import json
    import geopandas as gpd
    from collections import Counter
    from utils.location import Pref_Pinyin_dict
    china_map = gpd.read_file('data/geography/v6_1820_pref_pgn_gbk.shx', encoding='gb18030')
    china_map_pinyin = china_map['NAME_PY'].tolist()

    with open('data/binary_infer_entries.json', 'r') as f:
        data = json.load(f)
    
    pref = set()
    pref_counter = Counter()
    for line in data:
        pref.update(line['Prefecture'])
        pref_counter.update(line['Prefecture'])
    
    no_ch = 0
    for p in list(pref):
        if p not in china_map['NAME_CH'].tolist() and pref_counter[p] > 100:
            no_ch += 1
            print(p)
    print(no_ch)
    print('------------------')
    
    no_pinyin = 0
    for p in list(pref):
        if Pref_Pinyin_dict[p] not in china_map_pinyin and pref_counter[p] > 100:
            no_pinyin += 1
            print(p)
    print(no_pinyin)
    print('------------------')

    for line in china_map.iterrows():
        line = line[1]
        if line['NAME_CH'] not in pref:
            print(line['NAME_CH'])
    
    print('------------------')

def print_province_china_map():
    import geopandas as gpd
    china_map = gpd.read_file('data/geography/v6_1820_pref_pgn_gbk.shx', encoding='gb18030')

    print('All available provinces are: %s' % '; '.join([str(i) for i in list(set(china_map['LEV1_CH'].tolist()))]))

    while True:
        province = input('Enter the province:')
        lines = [str(p) for p in china_map.loc[china_map['LEV1_CH'] == province, 'NAME_CH'].tolist()]
        print('; '.join(lines))

def convert_entries():
    import json
    with open("data/binary_infer_entries.json", 'r') as f:
        data = json.load(f)
    data = list(filter(lambda x: x['Riot'], data))
    with open("data/filter_binary_infer_entries.json", 'w') as f:
        json.dump(data, f)

def export_ling():
    import os
    import torch
    from main import args
    train_dataset, test_dataset = torch.load(args.train_dataset), torch.load(args.test_dataset)
    train_pos, train_neg, test_pos, test_neg = [], [], [], []

    for line in train_dataset:
        if line[1] == 0:
            train_neg.append(line[0])
        elif line[1] == 1:
            train_pos.append(line[0])
        else:
            raise 1==2

    for line in test_dataset:
        if line[1] == 0:
            test_neg.append(line[0])
        elif line[1] == 1:
            test_pos.append(line[0])
        else:
            raise 1==2
    
    root_dir = '/mnt/c/Users/Sen Yan/Dropbox/shilu_comp_language backup Oct 3 2016/archive/NaiveBayes/Training'
    with open(os.path.join(root_dir, 'train', 'pos', 'sen.txt'), 'w') as f:
        for line in train_pos:
            # line = '○' + line
            f.write(line + '\n')
    with open(os.path.join(root_dir, 'train', 'neg', 'sen.txt'), 'w') as f:
        for line in train_neg:
            # line = '○' + line
            f.write(line + '\n')
    with open(os.path.join(root_dir, 'test', 'pos', 'sen.txt'), 'w') as f:
        for line in test_pos:
            # line = '○' + line
            f.write(line + '\n')
    with open(os.path.join(root_dir, 'test', 'neg', 'sen.txt'), 'w') as f:
        for line in test_neg:
            # line = '○' + line
            f.write(line + '\n')

def export_stata():
    import json
    import pinyin
    from utils.location import pref_prov_dict

    pref_dict = {pred: i for i, pred in enumerate(set(pref_prov_dict.keys()))}
    prov_dict = {prov: i for i, prov in enumerate(set(pref_prov_dict.values()))}

    with open('data/binary_infer_entries.json', 'r') as f:
        binary_data = json.load(f)
    with open('data/triple_infer_entries.json', 'r') as f:
        tri_data = json.load(f)

    years, months, riot_types, prefectures = [], [], [], []
    count = 0
    for entry in binary_data:
        if bool(entry['Riot']) == True:
            continue
        elif entry['month'] is None or entry['year'] is None or str(entry['year']) == 'None':
            continue
        else:
            for prefecture in entry['Prefecture']:
                years.append(entry['year'])
                months.append(entry['month'])
                riot_types.append('Nan')
                prefectures.append(prefecture)
            else:
                count += 1

    for entry in tri_data:
        if entry['month'] is None or entry['year'] is None or str(entry['year']) == 'None':
            continue
        for prefecture in entry['Prefecture']:
            years.append(entry['year'])
            months.append(entry['month'])
            riot_types.append(entry['RiotType'])
            prefectures.append(prefecture)

    # get month count
    month_count = {}
    for year in set(years):
        for month in range(1, 13):
            month_count['%d-%d' % (year, month)] = 0
    for year, month, riot_type in zip(years, months, riot_types):
        if riot_type != 'Nan':
            month_count['%d-%d' % (year, month)] += 1
    final_data = {}
    for year in set(years):
        for month in range(1, 13):
            final_data['%d-%d' % (year, month)] = {pref_dict[pref]: {'province': prov_dict[prov], 'riot_count': 0, 'jieshe_count': 0, 'peasants_count': 0, 'wuzhuang_count': 0, 'month_counts': month_count['%d-%d' % (year, month)]} for pref, prov in pref_prov_dict.items()}

    for year, month, rt, pref in zip(years, months, riot_types, prefectures):
        if rt == 'Nan':
            continue
        pref = pref_dict[pref]
        final_data['%d-%d' % (year, month)][pref]['riot_count'] = final_data['%d-%d' % (year, month)][pref]['riot_count'] + 1
        if rt == 'Peasant':
            final_data['%d-%d' % (year, month)][pref]['peasants_count'] = final_data['%d-%d' % (year, month)][pref]['peasants_count'] + 1
        elif rt == 'Secret Party':
            final_data['%d-%d' % (year, month)][pref]['jieshe_count'] = final_data['%d-%d' % (year, month)][pref]['jieshe_count'] + 1
        elif rt == 'Militia':
            final_data['%d-%d' % (year, month)][pref]['wuzhuang_count'] = final_data['%d-%d' % (year, month)][pref]['wuzhuang_count'] + 1
        else:
            raise Exception('Wrong rt %s' % rt)


    all_years, all_months, all_prefs, all_provs, all_rc, all_jc, all_pc, all_wc, all_mc = [], [], [], [], [], [], [], [], []
    for key, items in final_data.items():
        year, month = key.split('-')
        for pref, info in items.items():
            all_years.append(year)
            all_months.append(month)
            all_prefs.append(pref)
            all_provs.append(info['province'])
            all_rc.append(info['riot_count'])
            all_jc.append(info['jieshe_count'])
            all_pc.append(info['peasants_count'])
            all_wc.append(info['wuzhuang_count'])
            all_mc.append(info['month_counts'])

    df = pd.DataFrame({
        'year': all_years,
        'month': all_months,
        'prefecture': all_prefs,
        'province': all_provs,
        'month_counts': all_mc,
        'riot_count': all_rc,
        'jieshe_count': all_jc,
        'peasants_count': all_pc,
        'wuzhuang_count': all_wc
        })

    df.to_stata('data/stata_validation_weather_grain.dta')
        

def convert_stata_pref():
    from utils.location import Pref_Pinyin_dict

if __name__ == '__main__':
    # check_if_six_classes_in_chen()
    # check_three_classes_dataset()
    # debug_location_extraction()
    # plot_yili()
    # print_matched_loc()
    # gpd_miss_prefs()
    # print_province_china_map()
    # print_pref_twn_count()
    # convert_entries()
    # export_ling()
    export_stata()
