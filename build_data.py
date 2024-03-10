import json
import pickle
import utils.process_data
from glob import glob

processor = utils.process_data.Processor('data/entries', 'data/annotation_chen', 'data/sixclasses')

with open('data/data.pkl', 'wb') as f:
    pickle.dump(processor, f)

training_pos = []
for filename in glob('data/Training/train/pos/*.txt'):
    with open(filename, 'r') as f:
        for line in f.readlines():
            training_pos.append(line.strip()[1:])
training_pos = list(set(training_pos))

training_neg = []
for filename in glob('data/Training/train/neg/*.txt'):
    with open(filename, 'r') as f:
        for line in f.readlines():
            training_neg.append(line.strip()[1:])
training_neg = list(set(training_neg))

test_pos = []
for filename in glob('data/Training/test/pos/*.txt'):
    with open(filename, 'r') as f:
        for line in f.readlines():
            test_pos.append(line.strip()[1:])
test_pos = list(set(test_pos))

test_neg = []
for filename in glob('data/Training/test/neg/*.txt'):
    with open(filename, 'r') as f:
        for line in f.readlines():
            test_neg.append(line.strip()[1:])
test_neg = list(set(test_neg))


print('Training POS: %d' % len(training_pos))
print('Training NEG: %d' % len(training_neg))
print('Test POS: %d' % len(test_pos))
print('Test NEG: %d' % len(test_neg))
print()

training_data = [(line, 1) for line in training_pos] + [(line, 0) for line in training_neg]
test_data = [(line, 1) for line in test_pos] + [(line, 0) for line in test_neg]

print('Training Dataset: %d' % len(training_data))
print('Test Data: %d' % len(test_data))

with open('data/riot_data.json', 'w') as f:
    json.dump({'train': training_data, 'test': test_data}, f)
