import torch
import os
import json
import pickle
import random
import argparse
import itertools
import numpy as np
import torch.nn as nn

from tqdm import tqdm
from model import BertClassifier
from utils.export_stata_data import export_stata, export_stata_mix_prob
from utils.process_data import Processor
from utils.dataset import BinaryDataset, ThreeClassesDataset, FourClassesDataset, FiveClassesDataset, SecondStepFourClassesDataset
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

random_seed = 42 # for reproduce
torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

parser = argparse.ArgumentParser()
parser.add_argument('--raw_data', type=str, default='data/data.pkl')
parser.add_argument('--process_raw_data', action='store_true')
parser.add_argument('--prepare_data', action='store_true')
parser.add_argument('--train', action='store_true')
parser.add_argument('--train_three_classes', action='store_true')
parser.add_argument('--train_second_four_classes', action='store_true')
parser.add_argument('--train_four_classes', action='store_true')
parser.add_argument('--train_five_classes', action='store_true')
parser.add_argument('--infer', action='store_true')
parser.add_argument('--infer_three_classes', action='store_true')
parser.add_argument('--infer_four_classes', action='store_true')
parser.add_argument('--plot_figures', action='store_true')
parser.add_argument('--export_data_stata', action='store_true')
parser.add_argument('--export_data_web_demo', action='store_true')
parser.add_argument('--benchmark', action='store_true')

parser.add_argument('--train_ratio', type=float, default=0.8)
parser.add_argument('--train_dataset', type=str, default='data/train.pt')
parser.add_argument('--val_dataset', type=str, default='data/val.pt')
parser.add_argument('--test_dataset', type=str, default='data/test.pt')
parser.add_argument('--train_tri_dataset', type=str, default='data/three_classes_train.pt')
parser.add_argument('--val_tri_dataset', type=str, default='data/three_classes_val.pt')
parser.add_argument('--test_tri_dataset', type=str, default='data/three_classes_test.pt')
parser.add_argument('--train_second_four_dataset', type=str, default='data/second_four_classes_train.pt')
parser.add_argument('--val_second_four_dataset', type=str, default='data/second_four_classes_val.pt')
parser.add_argument('--test_second_four_dataset', type=str, default='data/second_four_classes_test.pt')
parser.add_argument('--train_four_dataset', type=str, default='data/four_classes_train.pt')
parser.add_argument('--val_four_dataset', type=str, default='data/four_classes_val.pt')
parser.add_argument('--test_four_dataset', type=str, default='data/four_classes_test.pt')
parser.add_argument('--train_five_dataset', type=str, default='data/five_classes_train.pt')
parser.add_argument('--val_five_dataset', type=str, default='data/five_classes_val.pt')
parser.add_argument('--test_five_dataset', type=str, default='data/five_classes_test.pt')
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--lr', type=float, default=2e-5)
parser.add_argument('--transformer_name', type=str, default='ethanyt/guwenbert-base')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--num_classes', type=int, default=2)
parser.add_argument('--log_dir', type=str, default='default')
parser.add_argument('--binary_infer_file', type=str, default='data/binary_infer_entries.json')
parser.add_argument('--triple_infer_file', type=str, default='data/triple_infer_entries.json')
parser.add_argument('--four_infer_file', type=str, default='data/four_infer_entries.json')
args = parser.parse_args()

if not os.path.exists(os.path.join('logs', args.log_dir)):
    os.mkdir(os.path.join('logs', args.log_dir))
args.binary_save_checkpoint = os.path.join('logs', args.log_dir, 'guwen-bert.pt')
args.triple_save_checkpoint = os.path.join('logs', args.log_dir, 'triple-guwen-bert.pt')
args.second_four_save_checkpoint = os.path.join('logs', args.log_dir, 'second_four-guwen-bert.pt')
args.four_save_checkpoint = os.path.join('logs', args.log_dir, 'four-guwen-bert.pt')
args.five_save_checkpoint = os.path.join('logs', args.log_dir, 'five-guwen-bert.pt')

def process_raw_data():
    processor = Processor('data/entries', 'data/annotation_chen', 'data/sixclasses')

    with open(args.raw_data, 'wb') as f:
        pickle.dump(processor, f)

def prepare_data():
    from collections import Counter
    ratios = {'train': 0.7, 'val': 0.1, 'test': 0.2}

    def split_dataset(dataset, train_ratio, val_ratio):
        train_len = int(len(dataset) * train_ratio)
        val_len = int(len(dataset) * val_ratio)
        test_len = len(dataset) - train_len - val_len
        return torch.utils.data.random_split(dataset, [train_len, val_len, test_len])

    # Binary classes
    binary_data = BinaryDataset(args.raw_data)
    train_data, val_data, test_data = split_dataset(binary_data, ratios['train'], ratios['val'])
    torch.save(train_data, args.train_dataset)
    torch.save(val_data, args.val_dataset)
    torch.save(test_data, args.test_dataset)
    print('Binary')
    print(Counter([d[1] for d in train_data]))
    print(Counter([d[1] for d in val_data]))
    print(Counter([d[1] for d in test_data]))

    # Three classes
    three_data = ThreeClassesDataset(args.raw_data)
    train_data, val_data, test_data = split_dataset(three_data, ratios['train'], ratios['val'])
    torch.save(train_data, args.train_tri_dataset)
    torch.save(val_data, args.val_tri_dataset)
    torch.save(test_data, args.test_tri_dataset)
    print('Three')
    print(Counter([d[1] for d in train_data]))
    print(Counter([d[1] for d in val_data]))
    print(Counter([d[1] for d in test_data]))

    # Second four classes
    four_data = SecondStepFourClassesDataset(args.raw_data)
    train_data, val_data, test_data = split_dataset(four_data, ratios['train'], ratios['val'])
    torch.save(train_data, args.train_second_four_dataset)
    torch.save(val_data, args.val_second_four_dataset)
    torch.save(test_data, args.test_second_four_dataset)
    print('Second Four')
    print(Counter([d[1] for d in train_data]))
    print(Counter([d[1] for d in val_data]))
    print(Counter([d[1] for d in test_data]))

    # Four classes
    four_data = FourClassesDataset(args.raw_data)
    train_data, val_data, test_data = split_dataset(four_data, ratios['train'], ratios['val'])
    torch.save(train_data, args.train_four_dataset)
    torch.save(val_data, args.val_four_dataset)
    torch.save(test_data, args.test_four_dataset)
    print('Four')
    print(Counter([d[1] for d in train_data]))
    print(Counter([d[1] for d in val_data]))
    print(Counter([d[1] for d in test_data]))

    # Five classes
    five_data = FiveClassesDataset(args.raw_data)
    train_data, val_data, test_data = split_dataset(five_data, ratios['train'], ratios['val'])
    torch.save(train_data, args.train_five_dataset)
    torch.save(val_data, args.val_five_dataset)
    torch.save(test_data, args.test_five_dataset)
    print('Five')
    print(Counter([d[1] for d in train_data]))
    print(Counter([d[1] for d in val_data]))
    print(Counter([d[1] for d in test_data]))


def train():
    train_dataset, test_dataset = torch.load(args.train_dataset, weights_only=False), torch.load(args.val_dataset, weights_only=False)


    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    model = BertClassifier(args.num_classes, args.transformer_name)
    model.to(model.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_acc = 0

    for epoch in range(args.epochs):
        pbar = tqdm(desc='Training Epoch %d' % (epoch + 1), total=len(train_loader))
        model.train()
        for batch in train_loader:
            sent, label = batch
            label = label.to(model.device)
            logits = model(sent)
            loss = nn.functional.cross_entropy(logits, label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            pbar.update(1)
            pbar.set_postfix_str('Loss: %.2f' % loss.item())
        pbar.close()

        pred, target = [], []
        model.eval()
        for batch in test_loader:
            sent, label = batch
            logits = model(sent)
            pred += torch.argmax(logits, dim=-1).detach().cpu().tolist()
            target += label.tolist()

        acc = np.sum(np.array(pred) == np.array(target)) / len(pred)
        print('Epoch: %d' % epoch)
        print("Classification Report:")
        print(classification_report(target, pred, digits=4))

        if acc > best_acc:
            best_acc = acc
            print('Find Better Results, Saving Checkpoints...\n')
            torch.save(model.state_dict(), args.binary_save_checkpoint)

def train_three_classes():
    train_dataset, test_dataset = torch.load(args.train_tri_dataset, weights_only=False), torch.load(args.val_tri_dataset, weights_only=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    model = BertClassifier(num_classes=3, transformer_name=args.transformer_name)
    model.to(model.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_acc = 0
    for epoch in range(args.epochs):
        pbar = tqdm(desc='Training Epoch %d' % (epoch + 1), total=len(train_loader))
        model.train()
        for batch in train_loader:
            sent, label = batch
            label = label.to(model.device)
            logits = model(sent)
            loss = nn.functional.cross_entropy(logits, label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            pbar.update(1)
            pbar.set_description_str('Loss: %.2f' % loss.item())
        pbar.close()

        pred, target = [], []
        model.eval()
        for batch in test_loader:
            sent, label = batch
            logits = model(sent)
            pred += torch.argmax(logits, dim=-1).detach().cpu().tolist()
            target += label.tolist()

        acc = np.sum(np.array(pred) == np.array(target)) / len(pred)
        print('Epoch: %d | Accuracy: %.10f' % (epoch, acc))
        print("Classification Report:\n", classification_report(target, pred, digits=4))
        
        if best_acc < acc:
            best_acc = acc
            print('Find Better Results, Saving Checkpoints...\n')
            torch.save(model.state_dict(), args.triple_save_checkpoint)

def train_second_four_classes():
    train_dataset, test_dataset = torch.load(args.train_second_four_dataset, weights_only=False), torch.load(args.val_second_four_dataset, weights_only=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    model = BertClassifier(num_classes=4, transformer_name=args.transformer_name)
    model.to(model.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_acc = 0
    for epoch in range(args.epochs):
        pbar = tqdm(desc='Training Epoch %d' % (epoch + 1), total=len(train_loader))
        model.train()
        for batch in train_loader:
            sent, label = batch
            label = label.to(model.device)
            logits = model(sent)
            loss = nn.functional.cross_entropy(logits, label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            pbar.update(1)
            pbar.set_description_str('Loss: %.2f' % loss.item())
        pbar.close()

        pred, target = [], []
        model.eval()
        for batch in test_loader:
            sent, label = batch
            logits = model(sent)
            pred += torch.argmax(logits, dim=-1).detach().cpu().tolist()
            target += label.tolist()

        acc = np.sum(np.array(pred) == np.array(target)) / len(pred)
        print('Epoch: %d | Accuracy: %.10f' % (epoch, acc))
        print("Classification Report:\n", classification_report(target, pred, digits=4))
        
        if best_acc < acc:
            best_acc = acc
            print('Find Better Results, Saving Checkpoints...\n')
            torch.save(model.state_dict(), args.second_four_save_checkpoint)

def train_four_classes():
    train_dataset, test_dataset = torch.load(args.train_four_dataset, weights_only=False), torch.load(args.val_four_dataset, weights_only=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    model = BertClassifier(num_classes=4, transformer_name=args.transformer_name)
    model.to(model.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_acc = 0
    for epoch in range(args.epochs):
        pbar = tqdm(desc='Training Epoch %d' % (epoch + 1), total=len(train_loader))
        model.train()
        for batch in train_loader:
            sent, label = batch
            label = label.to(model.device)
            logits = model(sent)
            loss = nn.functional.cross_entropy(logits, label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            pbar.update(1)
            pbar.set_description_str('Loss: %.2f' % loss.item())
        pbar.close()

        pred, target = [], []
        model.eval()
        for batch in test_loader:
            sent, label = batch
            logits = model(sent)
            pred += torch.argmax(logits, dim=-1).detach().cpu().tolist()
            target += label.tolist()

        acc = np.sum(np.array(pred) == np.array(target)) / len(pred)
        print('Epoch: %d | Accuracy: %.10f' % (epoch, acc))
        print("Classification Report:\n", classification_report(target, pred, digits=4))
        
        if best_acc < acc:
            best_acc = acc
            print('Find Better Results, Saving Checkpoints...\n')
            torch.save(model.state_dict(), args.four_save_checkpoint)

def train_five_classes():
    train_dataset, test_dataset = torch.load(args.train_five_dataset, weights_only=False), torch.load(args.val_five_dataset, weights_only=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    model = BertClassifier(num_classes=5, transformer_name=args.transformer_name)
    model.to(model.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_acc = 0
    for epoch in range(args.epochs):
        pbar = tqdm(desc='Training Epoch %d' % (epoch + 1), total=len(train_loader))
        model.train()
        for batch in train_loader:
            sent, label = batch
            label = label.to(model.device)
            logits = model(sent)
            loss = nn.functional.cross_entropy(logits, label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            pbar.update(1)
            pbar.set_description_str('Loss: %.2f' % loss.item())
        pbar.close()

        pred, target = [], []
        model.eval()
        for batch in test_loader:
            sent, label = batch
            logits = model(sent)
            pred += torch.argmax(logits, dim=-1).detach().cpu().tolist()
            target += label.tolist()

        acc = np.sum(np.array(pred) == np.array(target)) / len(pred)
        print('Epoch: %d | Accuracy: %.10f' % (epoch, acc))
        print("Classification Report:\n", classification_report(target, pred, digits=4))
        
        if best_acc < acc:
            best_acc = acc
            print('Find Better Results, Saving Checkpoints...\n')
            torch.save(model.state_dict(), args.five_save_checkpoint)

def infer():
    import pickle
    with open(args.raw_data, 'rb') as f:
        data = pickle.load(f).entries

    total_tqdm = len(data) // args.batch_size + 1
    data = iter(data)

    model = BertClassifier(args.num_classes, args.transformer_name)
    state_dict = torch.load(args.binary_save_checkpoint, weights_only=False)
    model.load_state_dict(state_dict)
    model.to(model.device)
    model.eval()

    def gen_batch(data, batch_size):
        while True:
            part = tuple(itertools.islice(data, batch_size))
            if not part:
                break
            yield part

    pred_entries = []
    for batch in tqdm(gen_batch(data, args.batch_size), total=total_tqdm):
        texts = [b['entry'].replace(' ', '').replace('○', '') for b in batch]
        pred = torch.argmax(model(texts), dim=-1).cpu().detach().tolist()
        for i, p in enumerate(pred):
            label = True if int(p) == 1 else False
            current_entry = batch[i]
            current_entry['Riot'] = label
            pred_entries.append(current_entry)

    with open(args.binary_infer_file, 'w') as f:
        json.dump(pred_entries, f)


def infer_three_classes():
    with open(args.binary_infer_file, 'r') as f:
        data = [dp for dp in json.load(f) if bool(dp['Riot']) == True]

    total_tqdm = len(data) // args.batch_size + 1
    data = iter(data)

    model = BertClassifier(num_classes=3, transformer_name=args.transformer_name)
    state_dict = torch.load(args.triple_save_checkpoint, weights_only=False)
    model.load_state_dict(state_dict)
    model.to(model.device)
    model.eval()

    def gen_batch(data, batch_size):
        while True:
            part = tuple(itertools.islice(data, batch_size))
            if not part:
                break
            yield part

    pred2label = {0: 'Peasant', 1: 'Secret Party', 2: 'Militia'}
    pred_entries = []
    for batch in tqdm(gen_batch(data, args.batch_size), total=total_tqdm):
        texts = [b['entry'].replace(' ', '').replace('○', '') for b in batch]
        prob = torch.softmax(model(texts), dim=-1).cpu().detach()
        pred = torch.argmax(prob, dim=-1).tolist()
        for i, p in enumerate(pred):
            label = pred2label[p]
            current_entry = batch[i]
            current_entry['RiotType'] = label
            current_entry['RiotProb'] = prob.tolist()[i]
            pred_entries.append(current_entry)

    with open(args.triple_infer_file, 'w') as f:
        json.dump(pred_entries, f)

def infer_four_classes():
    with open(args.binary_infer_file, 'r') as f:
        data = [dp for dp in json.load(f)]

    total_tqdm = len(data) // args.batch_size + 1
    data = iter(data)

    model = BertClassifier(num_classes=4, transformer_name=args.transformer_name)
    state_dict = torch.load(args.four_save_checkpoint, weights_only=False)
    model.load_state_dict(state_dict)
    model.to(model.device)
    model.eval()

    def gen_batch(data, batch_size):
        while True:
            part = tuple(itertools.islice(data, batch_size))
            if not part:
                break
            yield part

    pred2label = {0: 'Non-Riot', 1: 'Secret Party', 2: 'Militia', 3: 'Peasant'}
    pred_entries = []
    for batch in tqdm(gen_batch(data, args.batch_size), total=total_tqdm):
        texts = [b['entry'].replace(' ', '').replace('○', '') for b in batch]
        pred = torch.argmax(model(texts), dim=-1).cpu().detach().tolist()
        for i, p in enumerate(pred):
            label = pred2label[p]
            current_entry = batch[i]
            current_entry['RiotType'] = label
            pred_entries.append(current_entry)

    with open(args.four_infer_file, 'w') as f:
        json.dump(pred_entries, f)

def export_data_stata():
    export_stata(True, 17)
    # export_stata_mix_prob(True, 17)

def export_data_web_demo():
    with open(args.binary_infer_file, 'r') as f:
        data = json.load(f)
    data = list(filter(lambda x: x['Riot'], data))
    with open("data/web/filter_binary_infer_entries.json", 'w') as f:
        json.dump(data, f)

def benchmark():
    from utils.base_ml_model import perceptron_bi, perceptron_four, naive_bayes_four, mnb_bi, perceptron_five, naive_bayes_five, textcnn_five, textrnn_five, textcnn_bi, textrnn_bi, textcnn_four, textrnn_four, textcnn_four_end, textrnn_four_end, textcnn_tri, textrnn_tri, perceptron_four_end, naive_bayes_four_end, perceptron_tri, naive_bayes_tri
    print('*' * 100)
    print('\nBinary Classification: Perceptron')
    perceptron_bi()

    print('*' * 100)
    print('\nBinary Classification: Naive Bayes')
    mnb_bi()

    print('*' * 100)
    print('\nBinary Classification: TextCNN')
    textcnn_bi()

    print('*' * 100)
    print('\nBinary Classification: TextRNN')
    textrnn_bi()

    print('*' * 100)
    print('\n Four Classification: Perceptron')
    perceptron_four()

    print('*' * 100)
    print('\n Four Classification: Naive Bayes')
    naive_bayes_four()

    print('*' * 100)
    print('\n Four Classification: TextCNN')
    textcnn_four()

    print('*' * 100)
    print('\n Four Classification: TextRNN')
    textrnn_four()

    print('*' * 100)
    print('\nFive Classification: Perceptron')
    perceptron_five()

    print('*' * 100)
    print('\n Five Classification: Naive Bayes')
    naive_bayes_five()

    print('*' * 100)
    print('\nFive Classification: TextCNN')
    textcnn_five()

    print('*' * 100)
    print('\nFive Classification: TextRNN')
    textrnn_five()

    print('*' * 100)
    print('\n Tri Classification: Perceptron')
    perceptron_tri()

    print('*' * 100)
    print('\n Tri Classification: Naive Bayes')
    naive_bayes_tri()

    print('*' * 100)
    print('\n Tri Classification: TextCNN')
    textcnn_tri()

    print('*' * 100)
    print('\n Tri Classification: TextRNN')
    textrnn_tri()

    print('*' * 100)
    print('\n Four End Classification: Perceptron')
    perceptron_four_end()

    print('*' * 100)
    print('\n Four End Classification: Naive Bayes')
    naive_bayes_four_end()

    print('*' * 100)
    print('\n Four End Classification: TextCNN')
    textcnn_four_end()

    print('*' * 100)
    print('\n Four End Classification: TextRNN')
    textrnn_four_end()

if __name__ == '__main__':
    if args.process_raw_data:
        process_raw_data()

    if args.prepare_data:
        prepare_data()

    if args.train:
        train()

    if args.train_three_classes:
        train_three_classes()
    
    if args.train_second_four_classes:
        train_second_four_classes()

    if args.train_four_classes:
        train_four_classes()

    if args.train_five_classes:
        train_five_classes()

    if args.infer:
        infer()

    if args.infer_three_classes:
        infer_three_classes()

    if args.infer_four_classes:
        infer_four_classes()
    
    if args.export_data_stata:
        export_data_stata()
    
    if args.export_data_web_demo:
        export_data_web_demo()
    
    if args.benchmark:
        benchmark()
    
