import torch
import json
import pickle
import random
import argparse
import itertools
import numpy as np
import torch.nn as nn

from tqdm import tqdm
from model import BertClassifier
from utils.export_stata_data import export_stata
from utils.process_data import Processor
from utils.dataset import BinaryDataset, ThreeClassesDataset
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

random_seed = 1000 # for reproduce
torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

parser = argparse.ArgumentParser()
parser.add_argument('--raw_data', type=str, default='data/data.pkl')
parser.add_argument('--process_raw_data', action='store_true')
parser.add_argument('--prepare_data', action='store_true')
parser.add_argument('--train', action='store_true')
parser.add_argument('--train_three_classes', action='store_true')
parser.add_argument('--infer', action='store_true')
parser.add_argument('--infer_three_classes', action='store_true')
parser.add_argument('--plot_figures', action='store_true')
parser.add_argument('--export_data_stata', action='store_true')
parser.add_argument('--export_data_web_demo', action='store_true')

parser.add_argument('--train_ratio', type=float, default=0.8)
parser.add_argument('--train_dataset', type=str, default='data/train.pt')
parser.add_argument('--test_dataset', type=str, default='data/test.pt')
parser.add_argument('--train_tri_dataset', type=str, default='data/three_classes_train.pt')
parser.add_argument('--test_tri_dataset', type=str, default='data/three_classes_test.pt')
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--lr', type=float, default=2e-5)
parser.add_argument('--transformer_name', type=str, default='hfl/chinese-roberta-wwm-ext')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--num_classes', type=int, default=2)
parser.add_argument('--binary_save_checkpoint', type=str, default='logs/guwen-bert.pt')
parser.add_argument('--triple_save_checkpoint', type=str, default='logs/triple-guwen-bert.pt')
parser.add_argument('--binary_infer_file', type=str, default='data/binary_infer_entries.json')
parser.add_argument('--triple_infer_file', type=str, default='data/triple_infer_entries.json')
args = parser.parse_args()

def process_raw_data():
    processor = Processor('data/entries', 'data/annotation_chen', 'data/sixclasses')

    with open(args.raw_data, 'wb') as f:
        pickle.dump(processor, f)

def prepare_data():
    # binary classes
    data = BinaryDataset(args.raw_data)
    # randomly split
    num_train = int(len(data) * args.train_ratio)
    train_data, test_data = torch.utils.data.random_split(data, (num_train, len(data) - num_train))
    torch.save(train_data, args.train_dataset)
    torch.save(test_data, args.test_dataset)

    # three classes
    data = ThreeClassesDataset(args.raw_data)
    # randomly split
    num_train = int(len(data) * args.train_ratio)
    train_dataset, test_dataset = torch.utils.data.random_split(data, (num_train, len(data) - num_train))

    # save for base ml
    torch.save(train_dataset, args.train_tri_dataset)
    torch.save(test_dataset, args.test_tri_dataset)

def train():
    train_dataset, test_dataset = torch.load(args.train_dataset), torch.load(args.test_dataset)


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
    train_dataset, test_dataset = torch.load(args.train_tri_dataset), torch.load(args.test_tri_dataset)
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

def infer():
    import pickle
    with open(args.raw_data, 'rb') as f:
        data = pickle.load(f).entries

    total_tqdm = len(data) // args.batch_size + 1
    data = iter(data)

    model = BertClassifier(args.num_classes, args.transformer_name)
    state_dict = torch.load(args.binary_save_checkpoint)
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
    state_dict = torch.load(args.triple_save_checkpoint)
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
        pred = torch.argmax(model(texts), dim=-1).cpu().detach().tolist()
        for i, p in enumerate(pred):
            label = pred2label[p]
            current_entry = batch[i]
            current_entry['RiotType'] = label
            pred_entries.append(current_entry)

    with open(args.triple_infer_file, 'w') as f:
        json.dump(pred_entries, f)

def export_data_stata():
    export_stata(True, 17)

def export_data_web_demo():
    with open(args.binary_infer_file, 'r') as f:
        data = json.load(f)
    data = list(filter(lambda x: x['Riot'], data))
    with open("data/web/filter_binary_infer_entries.json", 'w') as f:
        json.dump(data, f)

if __name__ == '__main__':
    if args.process_raw_data:
        process_raw_data()

    if args.prepare_data:
        prepare_data()

    if args.train:
        train()

    if args.train_three_classes:
        train_three_classes()

    if args.infer:
        infer()

    if args.infer_three_classes:
        infer_three_classes()
    
    if args.export_data_stata:
        export_data_stata()
    
    if args.export_data_web_demo:
        export_data_web_demo()

