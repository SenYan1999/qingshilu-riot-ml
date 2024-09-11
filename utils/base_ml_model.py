import torch
import jieba
import xgboost as xgb
from main import args
from sklearn.linear_model import Perceptron
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import make_pipeline


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
    print("Classification Report:\n", classification_report(y_test, predicted_labels, digits=4))

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
    print("Binary Perceptron Classification Report:\n", classification_report(y_test, predicted_labels))

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
    print("Binary Naive Bayes Classification Report:\n", classification_report(y_test, predictions, digits=4))
    print()

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
    print("Triple Perceptron Classification Report:\n", classification_report(y_test, predicted_labels, digits=4))

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
    print("Triple Naive Bayes Classification Report:\n", classification_report(y_test, predicted_labels, digits=4))

def bert_bi():
    import torch
    from model.bert import BertClassifier
    model = BertClassifier(num_classes=2, transformer_name=args.transformer_name)
    state_dict = torch.load(args.binary_save_checkpoint)
    model.load_state_dict(state_dict)
    model.to(model.device)
    model.eval()

    test_dataset = torch.load('data/test.pt')
    preds, targets = [], []
    for line in test_dataset:
        texts = [b.replace(' ', '').replace('○', '') for b in [line[0]]]
        pred = torch.argmax(model(texts), dim=-1).cpu().detach().tolist()[0]
        preds.append(pred)
        targets.append(line[1])
    
    print("Binary GUWEN-BERT Classification Report:\n", classification_report(targets, preds, digits=4))

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
    
    print("Triple GUWEN-BERT Classification Report:\n", classification_report(targets, preds, digits=4))

def perceptron_four():
    # two class
    import numpy as np
    train_dataset, test_dataset = torch.load(args.train_four_dataset), torch.load(args.test_four_dataset)
    x_train, y_train = [d[0] for d in train_dataset], [d[1] for d in train_dataset]
    x_test, y_test = [d[0] for d in test_dataset], [d[1] for d in test_dataset]
    model = make_pipeline(TfidfVectorizer(tokenizer=jieba.lcut), Perceptron())

    # Training the model
    model.fit(x_train, y_train)

    # Predicting the labels for the test set
    predicted_labels = model.predict(x_test)

    # Evaluating the model
    print("Four Perceptron Classification Report:\n", classification_report(y_test, predicted_labels, digits=4))

def naive_bayes_four():
    # two class
    train_dataset, test_dataset = torch.load(args.train_four_dataset), torch.load(args.test_four_dataset)
    x_train, y_train = [d[0] for d in train_dataset], [d[1] for d in train_dataset]
    x_test, y_test = [d[0] for d in test_dataset], [d[1] for d in test_dataset]
    model = make_pipeline(TfidfVectorizer(tokenizer=jieba.lcut), MultinomialNB())

    # Training the model
    model.fit(x_train, y_train)

    # Predicting the labels for the test set
    predicted_labels = model.predict(x_test)

    # Evaluating the model
    print("Four Naive Bayes Classification Report:\n", classification_report(y_test, predicted_labels, digits=4))

def bert_four():
    import torch
    from model.bert import BertClassifier
    model = BertClassifier(num_classes=4, transformer_name=args.transformer_name)
    state_dict = torch.load(args.four_save_checkpoint)
    model.load_state_dict(state_dict)
    model.to(model.device)
    model.eval()

    test_dataset = torch.load(args.test_four_dataset)
    preds, targets = [], []
    for line in test_dataset:
        texts = [b.replace(' ', '').replace('○', '') for b in [line[0]]]
        pred = torch.argmax(model(texts), dim=-1).cpu().detach().tolist()[0]
        preds.append(pred)
        targets.append(line[1])
    
    print("Four GUWEN-BERT Classification Report:\n", classification_report(targets, preds, digits=4))


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
