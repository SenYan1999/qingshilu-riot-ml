import os
import re
import glob
from tqdm import tqdm
from pycnnum import cn2num
from .location import LocationIdentify

class Processor:
    def __init__(self, entry_dir, annotation_dir, sixclasses_dir):
        self.emperor_list = ['Shunzhi', 'Kangxi', 'Yongzheng', 'Qianlong', 'Jiaqing', 'Daoguang',
                        'Xianfeng', 'Tongzhi', 'Guangxu', 'Xuantong']
        self.emperor_pinyin = {'Shunzhi': '顺治', 'Kangxi': '康熙', 'Yongzheng': '雍正', 'Qianlong': '乾隆',
                               'Daoguang': '道光', 'Jiaqing': '嘉庆', 'Tongzhi': '同治', 'Xianfeng': '咸丰',
                               'Guangxu': '光绪', 'Xuantong': '宣统'}
        self.emperor_year = {
            "崇德": 1636,
            "顺治": 1644,
            "康熙": 1662,
            "雍正": 1723,
            "乾隆": 1736,
            "嘉庆": 1796,
            "道光": 1821,
            "咸丰": 1851,
            "同治": 1862,
            "光绪": 1875,
            "宣统": 1909,
        }
        self.emperor_full_name = {
            'Shunzhi': '大清世祖体天隆运定统建极英睿钦文显武大德弘功至仁纯孝章皇帝实录卷之',
            'Kangxi': '大清圣祖合天弘运文武睿哲恭俭宽裕孝敬诚信中和功德大成仁皇帝实录卷之',
            'Yongzheng': '大清世宗敬天昌运建中表正文武英明宽仁信毅大孝至诚宪皇帝实录卷之',
            'Qianlong': '大清高宗法天隆运至诚先觉体元立极敷文奋武孝慈神圣纯皇帝实录卷之',
            'Daoguang': '大清宣宗效天符运立中体正至文圣武智勇仁慈俭勤孝敏成皇帝实录卷之',
            'Jiaqing': '大清仁宗受天兴运敷化绥猷崇文经武孝恭勤俭端敏英哲睿皇帝实录卷之',
            'Tongzhi': '大清穆宗继天开运受中居正保大定功圣智诚孝信敏恭宽毅皇帝实录卷之',
            'Xianfeng': '大清文宗协天翊运执中垂谟懋德振武圣孝渊恭端仁宽敏显皇帝实录卷之',
            'Guangxu': '大清圣祖合天弘运文武睿哲恭俭宽裕孝敬诚信中和功德大成仁皇帝实录卷之',
            'Xuantong': '大清宣统政纪卷之'
        }
        self.entries = self.extract_entry(entry_dir)
        self.annotated_entries = self.extract_training_dataset(annotation_dir, sixclasses_dir)

    def extract_entry(self, entry_dir):
        # extract entry, the entry includes entry text, year
        cn_num = r'[闰〇初正元一二三四五六七八九十]{1,5}'

        ems = '|'.join(list(self.emperor_year.keys()))
        ems = f'({ems})'

        year_pattern = rf'{ems}.?{cn_num}年'
        month_pattern = r"(正月|腊月|[一二三四五六七八九十]{1,2}月)"
        entries = []
        for emperor in self.emperor_list:
            with open(os.path.join(entry_dir, emperor+'.txt')) as f:
                meta_info = f.read().strip().split('○')[0]
                for line in meta_info.split('\n'):
                    year_match = re.search(year_pattern, line[:20])
                    month_match = re.search(month_pattern, line[:20])
                    if year_match and month_match:
                        year = year_match.group(0)
                        year_num = year_match.group(0)[2:-1]
                        month_num = month_match.group(0)[:-1]
                        year_num = '一' if year_num in ['元', '正'] else year_num
                        month_num = '一' if month_num == '正' else month_num
                        month_num = '十二' if month_num == '腊' else month_num
                        year_num, month_num = self.emperor_year[self.emperor_pinyin[emperor]] + cn2num(year_num) - 1, cn2num(month_num)
                        break
                else:
                    raise Exception('Incorrect Meta Information in {}'.format(os.path.join(entry_dir, emperor+'.txt')))

            with open(os.path.join(entry_dir, emperor+'.txt')) as f:
                lines = f.read().strip()

            assert meta_info in lines
            lines = lines.replace(meta_info, '')
            assert len(lines.split('\n')) > 0

            for line in tqdm(lines.split('\n'), desc=f'Process Emperor {emperor}'):
                line = line.strip()
                if len(line) == 0:
                    continue
                if line.startswith(self.emperor_full_name[emperor]):
                    year_num, month_num, year = None, None, None
                if not line.startswith('○'):
                    year_match = re.search(year_pattern, line[:20])
                    month_match = re.search(month_pattern, line[:20])
                    if year_match and month_match:
                        year = year_match.group(0)
                        year_num = year_match.group(0)[2:-1].replace('。', '')
                        month_num = month_match.group(0)[:-1]
                        year_num = '一' if year_num in ['元', '正'] else year_num
                        month_num = '一' if month_num == '正' else month_num
                        month_num = '十二' if month_num == '腊' else month_num
                        year_num, month_num = self.emperor_year[self.emperor_pinyin[emperor]] + cn2num(year_num) - 1, cn2num(month_num)
                else:
                    # assert year_num != None and month_num != None and year != None
                    # may be some error, see the comment code in the above line
                    prefectures = LocationIdentify(line[1:])
                    entries.append({'entry': line[1:], 'emperor': emperor, 'year': year_num, 'month': month_num, 'chinese_year': year, 'prefectures': prefectures})
        return entries

    def extract_training_dataset(self, annotation_dir, sixclasses_dir):
        data = []
        fail, count = 0, 0
        with open(os.path.join(annotation_dir, 'Train_riots.txt'), 'r') as f:
            for line in tqdm(f.readlines(), 'Extracting Riots'):
                line = line.strip().replace('○', '')
                entries = list(filter(lambda x: x['entry'][:100] == line[:100], self.entries))
                if len(entries) == 0:
                    fail += 1
                else:
                    entry = entries[0]
                    entry['Riot'] = True
                    entry['RiotType'] = 'Unknown'
                    data.append(entry)
                count += 1
            print('Fail Rate (Process Annotation Chen): %2d / %2d = %.2f' % (fail, count, fail / count))

        fail, count = 0, 0
        with open(os.path.join(annotation_dir, 'Train_nonriots.txt'), 'r') as f:
            for line in tqdm(f.readlines(), desc='Extracting Non-Riots'):
                line = line.strip().replace('○', '')
                entries = list(filter(lambda x: x['entry'][:100] == line[:100], self.entries))
                if len(entries) == 0:
                    fail += 1
                else:
                    entry = entries[0]
                    entry['Riot'] = False
                    entry['RiotType'] = 'Non-Riot'
                    data.append(entry)
                count += 1
            print('Fail Rate (Process Six Classes Annotation): %2d / %2d = %.2f' % (fail, count, fail / count))

        for file in glob.glob(os.path.join(sixclasses_dir, '*.txt')):
            riot_type = file.split('/')[-1].replace('.txt', '').lower()
            fail, count = 0, 0
            with open(file, 'r') as f:
                for line in f.readlines():
                    line = line.strip().replace('○', '')
                    line = re.sub(r'\d+', '', line).strip()
                    if len(line) == 0:
                        continue
                    entries = [i for i, x in enumerate(data) if x['entry'][:100] == line[:100]]
                    if len(entries) == 0:
                        fail += 1
                    else:
                        data[entries[0]]['RiotType'] = riot_type
                    count += 1
            print('Fail Rate in %10s %2d / %2d = %.2f' % (riot_type + ':', fail, count, fail / count))

        return data
