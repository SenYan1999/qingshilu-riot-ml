import torch
import jieba
import pickle
import xgboost as xgb
from main import args
from sklearn.linear_model import Perceptron
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import make_pipeline


def naive_bayes_bi():
    # two class
    train_dataset, test_dataset = torch.load(args.train_dataset), torch.load(args.test_dataset)
    x_train, y_train = [d[0] for d in train_dataset], [d[1] for d in train_dataset]
    x_test, y_test = [d[0] for d in test_dataset], [d[1] for d in test_dataset]
    model = make_pipeline(TfidfVectorizer(tokenizer=jieba.lcut), MultinomialNB())

    # Training the model
    model.fit(x_train, y_train)

    # Predicting the labels for the test set
    predicted_labels = model.predict(x_test)

    # Evaluating the model
    print("Accuracy:", accuracy_score(y_test, predicted_labels))
    print("Classification Report:", classification_report(y_test, predicted_labels))

def perceptron_bi():
    # two class
    import numpy as np
    train_dataset, test_dataset = torch.load(args.train_dataset), torch.load(args.test_dataset)
    x_train, y_train = [d[0] for d in train_dataset], [d[1] for d in train_dataset]
    x_test, y_test = [d[0] for d in test_dataset], [d[1] for d in test_dataset]
    model = make_pipeline(TfidfVectorizer(tokenizer=jieba.lcut), Perceptron())

    # Training the model
    model.fit(x_train, y_train)

    # Predicting the labels for the test set
    predicted_labels = model.predict(x_test)

    # Evaluating the model
    print("Accuracy:", accuracy_score(y_test, predicted_labels))
    print("Classification Report:", classification_report(y_test, predicted_labels, digits=4))

def xgboost_bi():
    # two class
    train_dataset, test_dataset = torch.load(args.train_dataset), torch.load(args.test_dataset)
    x_train, y_train = [d[0] for d in train_dataset], [d[1] for d in train_dataset]
    x_test, y_test = [d[0] for d in test_dataset], [d[1] for d in test_dataset]
    model = make_pipeline(TfidfVectorizer(jieba.lcut), xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss'))

    # Training the model
    model.fit(x_train, y_train)

    # Predicting the labels for the test set
    predicted_labels = model.predict(x_test)

    # Evaluating the model
    print("Accuracy:", accuracy_score(y_test, predicted_labels))
    print("Classification Report:", classification_report(y_test, predicted_labels))

def perceptron_onehot_bi():
    # two class
    train_dataset, test_dataset = torch.load(args.train_dataset), torch.load(args.test_dataset)
    x_train, y_train = [d[0] for d in train_dataset], [d[1] for d in train_dataset]
    x_test, y_test = [d[0] for d in test_dataset], [d[1] for d in test_dataset]
    encoder = CountVectorizer(binary=True, tokenizer=jieba.lcut)  # Set binary=True for one-hot encoding
    X = encoder.fit_transform(x_train).toarray()

    model = Perceptron()

    # Training the model
    model.fit(encoder.transform(x_train), y_train)

    # Predicting the labels for the test set
    predicted_labels = model.predict(encoder.transform(x_test))

    # Evaluating the model
    print("Accuracy:", accuracy_score(y_test, predicted_labels))

def mnb_bi():
    import numpy as np

    train_dataset, test_dataset = torch.load(args.train_dataset), torch.load(args.test_dataset)
    x_train, y_train = [d[0] for d in train_dataset], [d[1] for d in train_dataset]
    x_test, y_test = [d[0] for d in test_dataset], [d[1] for d in test_dataset]

    # 1. Tokenize the text
    tokens_train = [list(text) for text in x_train]

    # 2. Create vocabulary
    vocab = set(word for text in tokens_train for word in text)

    # 3. Count word frequencies
    word_counts = {word: np.zeros(2) for word in vocab}
    for text, label in zip(tokens_train, y_train):
        for word in text:
            if label == 1:
                word_counts[word][0] += 1
            else:
                word_counts[word][1] += 1

    # 4. Calculate class probabilities
    class_prob = np.array([
        sum(label == 1 for label in y_train) / len(y_train),
        sum(label == 0 for label in y_train) / len(y_train),
    ])

    # 5. Calculate conditional probabilities for each word
    total_words = [0, 0]
    for word in vocab:
        total_words[0] += word_counts[word][0]
        total_words[1] += word_counts[word][1]

    word_probs = {word: [(word_counts[word][0] + 1) / (total_words[0] + len(vocab)),
                     (word_counts[word][1] + 1) / (total_words[1] + len(vocab))] 
              for word in vocab}

    # 6. Classify new instances
    def classify(text):
        words = list(text)
        spam_prob = np.log(class_prob[0])
        ham_prob = np.log(class_prob[1])
        for word in words:
            if word in word_probs:
                spam_prob += np.log(word_probs[word][0])
                ham_prob += np.log(word_probs[word][1])
        return 1 if spam_prob > ham_prob else 0

    # Test the classifier on x_test
    predictions = [classify(text) for text in x_test]

    # Calculate accuracy
    accuracy = sum(pred == actual for pred, actual in zip(predictions, y_test)) / len(y_test)
    print(classification_report(y_test, predictions, digits=4))

def naive_bayes_ling():
    import os
    import math

    def filetowordlist(dataset):
        """Read a riot training file and returns all characters and punctuations as a list."""
        poswords, negwords = [], []
        for line in dataset:
            if line[1] == 0:
                for c in line[0].strip():
                    negwords.append(c)
            elif line[1] == 1:
                for c in line[0].strip():
                    poswords.append(c)
            else:
                raise Exception('Error label in dataset')
        return poswords, negwords

    def classifyfiles(dataset):
        """Classify several test files, keeping track of #correct."""
        guesses = correct = 0
        for line in dataset:
            pp = pn = 0
            line, label = line[0], line[1]
            if line != '\n':
                item = line.strip()
                for character in item:
                    if character in vocabulary:
                        pp += math.log(poscounts[character]/postotal)
                        pn += math.log(negcounts[character]/negtotal)
                if (pp > pn and label == 1) or (pn >= pp and label == 0):
                    correct += 1
                guesses += 1
            #print correct, guesses
        return correct, guesses
                        
    train_dataset, test_dataset = torch.load(args.train_dataset), torch.load(args.test_dataset)
    poswords, negwords = filetowordlist(train_dataset)

    vocabulary = set(poswords + negwords)

    PRIOR = 1.0 

    poscounts = {w: PRIOR for w in vocabulary} # Initialize counts with prior (0.5)
    negcounts = {w: PRIOR for w in vocabulary}

    for w in poswords:
        poscounts[w] = poscounts[w] + 1
    for w in negwords:
        negcounts[w] = negcounts[w] + 1

    postotal = sum(poscounts.values()) # Calculate normalization
    negtotal = sum(negcounts.values())

    numcorrectpos, numattemptspos = classifyfiles(test_dataset)

    print("ACCURACY: %f" % (numcorrectpos/float(numattemptspos)))

def perceptron_tri():
    # two class
    import numpy as np
    train_dataset, test_dataset = torch.load('data/three_classes_train.pt'), torch.load('data/three_classes_test.pt')
    x_train, y_train = [d[0] for d in train_dataset], [d[1] for d in train_dataset]
    x_test, y_test = [d[0] for d in test_dataset], [d[1] for d in test_dataset]
    model = make_pipeline(TfidfVectorizer(tokenizer=jieba.lcut), Perceptron())

    # Training the model
    model.fit(x_train, y_train)

    # Predicting the labels for the test set
    predicted_labels = model.predict(x_test)

    # Evaluating the model
    print("Accuracy:", accuracy_score(y_test, predicted_labels))
    print("Classification Report:", classification_report(y_test, predicted_labels, digits=4))

def naive_bayes_tri():
    # two class
    train_dataset, test_dataset = torch.load(args.train_dataset), torch.load(args.test_dataset)
    x_train, y_train = [d[0] for d in train_dataset], [d[1] for d in train_dataset]
    x_test, y_test = [d[0] for d in test_dataset], [d[1] for d in test_dataset]
    model = make_pipeline(TfidfVectorizer(tokenizer=jieba.lcut), MultinomialNB())

    # Training the model
    model.fit(x_train, y_train)

    # Predicting the labels for the test set
    predicted_labels = model.predict(x_test)

    # Evaluating the model
    print("Accuracy:", accuracy_score(y_test, predicted_labels))
    print("Classification Report:", classification_report(y_test, predicted_labels, digits=4))

def bert_tri():
    import torch
    from model.bert import BertClassifier
    model = BertClassifier(num_classes=3, transformer_name=args.transformer_name)
    state_dict = torch.load(args.triple_save_checkpoint)
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
    
    print("Classification Report:", classification_report(targets, preds, digits=4))



if __name__ == '__main__':
    # naive_bayes_bi()
    # perceptron_bi()
    # xgboost_bi()
    # perceptron_onehot_bi()
    # mnb_bi()
    # naive_bayes_ling()
    perceptron_tri()
    # naive_bayes_tri()
    # bert_tri()
