import torch
import jieba
import torch.nn as nn
import xgboost as xgb
from main import args
from sklearn.linear_model import Perceptron
from skorch import NeuralNetClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import make_pipeline
from pytextclassifier import TextCNNClassifier, TextRNNClassifier

def perceptron_tri():
    # two class
    import numpy as np
    train_dataset, test_dataset = torch.load(args.val_tri_dataset), torch.load(args.test_tri_dataset)
    x_train, y_train = [d[0] for d in train_dataset], [d[1] for d in train_dataset]
    x_test, y_test = [d[0] for d in test_dataset], [d[1] for d in test_dataset]
    model = make_pipeline(TfidfVectorizer(tokenizer=jieba.lcut), Perceptron())

    # Training the model
    model.fit(x_train, y_train)

    # Predicting the labels for the test set
    predicted_labels = model.predict(x_test)

    # Evaluating the model
    print("Triple Perceptron Classification Report:\n", classification_report(y_test, predicted_labels, digits=4))

def naive_bayes_tri():
    # two class
    train_dataset, test_dataset = torch.load('data/three_classes_train.pt'), torch.load('data/three_classes_test.pt')
    x_train, y_train = [d[0] for d in train_dataset], [d[1] for d in train_dataset]
    x_test, y_test = [d[0] for d in test_dataset], [d[1] for d in test_dataset]
    model = make_pipeline(TfidfVectorizer(tokenizer=jieba.lcut), MultinomialNB())

    # Training the model
    model.fit(x_train, y_train)

    # Predicting the labels for the test set
    predicted_labels = model.predict(x_test)

    # Evaluating the model
    print("Triple Naive Bayes Classification Report:\n", classification_report(y_test, predicted_labels, digits=4))

def bert_tri():
    import torch
    from model.bert import BertClassifier
    model = BertClassifier(num_classes=3, transformer_name=args.transformer_name)
    state_dict = torch.load('temp/triple-guwen-bert.pt')
    model.load_state_dict(state_dict)
    model.to(model.device)
    model.eval()

    test_dataset = torch.load('data/three_classes_test.pt')
    preds, targets = [], []
    for line in test_dataset:
        texts = [b.replace(' ', '').replace('○', '') for b in [line[0]]]
        pred = torch.argmax(model(texts), dim=-1).cpu().detach().tolist()[0]
        preds.append(pred)
        targets.append(line[1])
    
    print("Triple GUWEN-BERT Classification Report:\n", classification_report(targets, preds, digits=4))

if __name__ == '__main__':
    perceptron_tri()
    # naive_bayes_tri()
    # bert_tri()