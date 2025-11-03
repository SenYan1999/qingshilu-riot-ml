import json
import pandas as pd

def oct_20():
    with open('data/triple_infer_entries.json', 'r') as f:
        data = json.load(f)
    all_entries, all_labels = [], []
    for line in data:
        entry = line['entry']
        t = line['RiotType']
        if t in ['Peasant', 'Secret Party']:
            all_entries.append(entry)
            all_labels.append(t)
    
    print(len(data))
    print(len(all_entries))
    print(set(all_labels))

    pd.DataFrame({'Entry': all_entries, 'Type': all_labels}).to_csv('temp/oct_20_peasant_secret.csv')

if __name__ == '__main__':
    oct_20()