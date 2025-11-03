import json
import pickle
import torch.utils.data as data

class BaseDataset(data.Dataset):
    def __init__(self, file):
        self.data = self._load_data(file)

    def _load_data(self, file):
        with open(file, 'r') as f:
            raw_data = json.load(f)
        return raw_data['train'] + raw_data['test']

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        return self.data[i]

class BinaryDataset(BaseDataset):
    def __init__(self, file):
        super(BinaryDataset, self).__init__(file)

    def _load_data(self, file):
        with open(file, 'rb') as f:
            raw_data = pickle.load(f)
        data = [(entry['entry'], int(entry['Riot'])) for entry in raw_data.annotated_entries]
        return data

class ThreeClassesDataset(BaseDataset):
    def __init__(self, file):
        super(ThreeClassesDataset, self).__init__(file)

    def _load_data(self, file):
        self.class2label = {'disaster': 0, 'others_peasant': 0, 'rent': 0, 'tax': 0, 'jieshe': 1, 'wuzhuang': 2}
        with open(file, 'rb') as f:
            raw_data = pickle.load(f)
        data = [(entry['entry'], self.class2label[entry['RiotType']]) for entry in raw_data.annotated_entries if entry['RiotType'] not in ['Non-Riot', 'others']]
        return data

class SecondStepFourClassesDataset(BaseDataset):
    def __init__(self, file):
        super(SecondStepFourClassesDataset, self).__init__(file)

    def _load_data(self, file):
        self.class2label = {'disaster': 0, 'others_peasant': 0, 'rent': 0, 'tax': 0, 'jieshe': 1, 'wuzhuang': 2, 'others': 3}
        with open(file, 'rb') as f:
            raw_data = pickle.load(f)
        data = [(entry['entry'], self.class2label[entry['RiotType']]) for entry in raw_data.annotated_entries if entry['RiotType'] not in ['Non-Riot']]
        from collections import Counter
        print(Counter([entry['RiotType'] for entry in raw_data.annotated_entries]))
        return data

class FourClassesDataset(BaseDataset):
    def __init__(self, file):
        super(FourClassesDataset, self).__init__(file)

    def _load_data(self, file):
        self.class2label = {'Non-Riot': 0, 'disaster': 3, 'others_peasant': 3, 'rent': 3, 'tax': 3, 'jieshe': 1, 'wuzhuang': 2}
        with open(file, 'rb') as f:
            raw_data = pickle.load(f)
        data = [(entry['entry'], self.class2label[entry['RiotType']]) for entry in raw_data.annotated_entries if entry['RiotType'] != 'others']
        return data

class FiveClassesDataset(BaseDataset):
    def __init__(self, file):
        super(FiveClassesDataset, self).__init__(file)

    def _load_data(self, file):
        from collections import Counter
        self.class2label = {'Non-Riot': 0, 'disaster': 3, 'others_peasant': 3, 'rent': 3, 'tax': 3, 'jieshe': 1, 'wuzhuang': 2, 'others': 4}
        with open(file, 'rb') as f:
            raw_data = pickle.load(f)
        print(Counter([entry['RiotType'] for entry in raw_data.annotated_entries]))
        data = [(entry['entry'], self.class2label[entry['RiotType']]) for entry in raw_data.annotated_entries]
        print(Counter([line[1] for line in data]))
        return data

if __name__ == '__main__':
    from collections import Counter
    data = ThreeClassesDataset('data/data.pkl')
    print(Counter([dp[1] for dp in data]))

