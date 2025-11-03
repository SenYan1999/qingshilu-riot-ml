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
    train_dataset, test_dataset = torch.load(args.train_tri_dataset), torch.load(args.test_tri_dataset)
    x_train, y_train = [d[0] for d in train_dataset], [d[1] for d in train_dataset]
    x_test, y_test = [d[0] for d in test_dataset], [d[1] for d in test_dataset]
    model = make_pipeline(TfidfVectorizer(tokenizer=jieba.lcut), Perceptron())

    # Training the model
    model.fit(x_train, y_train)

    # Predicting the labels for the test set
    predicted_labels = model.predict(x_test)

    # Evaluating the model
    print("Tri Classification Report:\n", classification_report(y_test, predicted_labels, digits=4))

def naive_bayes_tri():
    # two class
    train_dataset, test_dataset = torch.load(args.train_tri_dataset), torch.load(args.test_tri_dataset)
    x_train, y_train = [d[0] for d in train_dataset], [d[1] for d in train_dataset]
    x_test, y_test = [d[0] for d in test_dataset], [d[1] for d in test_dataset]
    model = make_pipeline(TfidfVectorizer(tokenizer=jieba.lcut), MultinomialNB())

    # Training the model
    model.fit(x_train, y_train)

    # Predicting the labels for the test set
    predicted_labels = model.predict(x_test)

    # Evaluating the model
    print("Tri Naive Bayes Classification Report:\n", classification_report(y_test, predicted_labels, digits=4))

def perceptron_four_end():
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
    print("End Four Classification Report:\n", classification_report(y_test, predicted_labels, digits=4))

def naive_bayes_four_end():
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
    print("End Four Naive Bayes Classification Report:\n", classification_report(y_test, predicted_labels, digits=4))


def perceptron_four():
    # two class
    import numpy as np
    train_dataset, test_dataset = torch.load(args.train_second_four_dataset), torch.load(args.test_second_four_dataset)
    x_train, y_train = [d[0] for d in train_dataset], [d[1] for d in train_dataset]
    x_test, y_test = [d[0] for d in test_dataset], [d[1] for d in test_dataset]
    model = make_pipeline(TfidfVectorizer(tokenizer=jieba.lcut), Perceptron())

    # Training the model
    model.fit(x_train, y_train)

    # Predicting the labels for the test set
    predicted_labels = model.predict(x_test)

    # Evaluating the model
    print("Second Step Four Classification Report:\n", classification_report(y_test, predicted_labels, digits=4))

def naive_bayes_four():
    # two class
    train_dataset, test_dataset = torch.load(args.train_second_four_dataset), torch.load(args.test_second_four_dataset)
    x_train, y_train = [d[0] for d in train_dataset], [d[1] for d in train_dataset]
    x_test, y_test = [d[0] for d in test_dataset], [d[1] for d in test_dataset]
    model = make_pipeline(TfidfVectorizer(tokenizer=jieba.lcut), MultinomialNB())

    # Training the model
    model.fit(x_train, y_train)

    # Predicting the labels for the test set
    predicted_labels = model.predict(x_test)

    # Evaluating the model
    print("Second Step Four Naive Bayes Classification Report:\n", classification_report(y_test, predicted_labels, digits=4))

def perceptron_five():
    # two class
    import numpy as np
    train_dataset, test_dataset = torch.load(args.train_five_dataset), torch.load(args.test_five_dataset)
    x_train, y_train = [d[0] for d in train_dataset], [d[1] for d in train_dataset]
    x_test, y_test = [d[0] for d in test_dataset], [d[1] for d in test_dataset]
    model = make_pipeline(TfidfVectorizer(tokenizer=jieba.lcut), Perceptron())

    # Training the model
    model.fit(x_train, y_train)

    # Predicting the labels for the test set
    predicted_labels = model.predict(x_test)

    # Evaluating the model
    print("five Perceptron Classification Report:\n", classification_report(y_test, predicted_labels, digits=4))

def naive_bayes_five():
    # two class
    train_dataset, test_dataset = torch.load(args.train_five_dataset), torch.load(args.test_five_dataset)
    x_train, y_train = [d[0] for d in train_dataset], [d[1] for d in train_dataset]
    x_test, y_test = [d[0] for d in test_dataset], [d[1] for d in test_dataset]
    model = make_pipeline(TfidfVectorizer(tokenizer=jieba.lcut), MultinomialNB())

    # Training the model
    model.fit(x_train, y_train)

    # Predicting the labels for the test set
    predicted_labels = model.predict(x_test)

    # Evaluating the model
    print("Five Naive Bayes Classification Report:\n", classification_report(y_test, predicted_labels, digits=4))

def prepare_data_jul_14_bi():
    train_dataset, val_dataset, test_dataset = torch.load(args.train_dataset, weights_only=False), torch.load(args.val_dataset, weights_only=False), torch.load(args.test_dataset, weights_only=False)
    train_dataset_processed = [(line[1], line[0]) for line in train_dataset]
    val_dataset_processed = [(line[1], line[0]) for line in val_dataset]
    test_dataset_processed = [(line[1], line[0]) for line in test_dataset]
    return train_dataset_processed, val_dataset_processed, test_dataset_processed

def prepare_data_jul_14_tri():
    train_dataset, val_dataset, test_dataset = torch.load(args.train_tri_dataset, weights_only=False), torch.load(args.val_tri_dataset, weights_only=False), torch.load(args.test_tri_dataset, weights_only=False)
    train_dataset_processed = [(line[1], line[0]) for line in train_dataset]
    val_dataset_processed = [(line[1], line[0]) for line in val_dataset]
    test_dataset_processed = [(line[1], line[0]) for line in test_dataset]
    return train_dataset_processed, val_dataset_processed, test_dataset_processed

def prepare_data_jul_14_four_end():
    train_dataset, val_dataset, test_dataset = torch.load(args.train_four_dataset, weights_only=False), torch.load(args.val_four_dataset, weights_only=False), torch.load(args.test_four_dataset, weights_only=False)
    train_dataset_processed = [(line[1], line[0]) for line in train_dataset]
    val_dataset_processed = [(line[1], line[0]) for line in val_dataset]
    test_dataset_processed = [(line[1], line[0]) for line in test_dataset]
    return train_dataset_processed, val_dataset_processed, test_dataset_processed

def prepare_data_jul_14_four():
    train_dataset, val_dataset, test_dataset = torch.load(args.train_second_four_dataset, weights_only=False), torch.load(args.val_second_four_dataset, weights_only=False), torch.load(args.test_second_four_dataset, weights_only=False)
    train_dataset_processed = [(line[1], line[0]) for line in train_dataset]
    val_dataset_processed = [(line[1], line[0]) for line in val_dataset]
    test_dataset_processed = [(line[1], line[0]) for line in test_dataset]
    return train_dataset_processed, val_dataset_processed, test_dataset_processed

def prepare_data_jul_14_five():
    train_dataset, val_dataset, test_dataset = torch.load(args.train_five_dataset, weights_only=False), torch.load(args.val_five_dataset, weights_only=False), torch.load(args.test_five_dataset, weights_only=False)
    train_dataset_processed = [(line[1], line[0]) for line in train_dataset]
    val_dataset_processed = [(line[1], line[0]) for line in val_dataset]
    test_dataset_processed = [(line[1], line[0]) for line in test_dataset]
    return train_dataset_processed, val_dataset_processed, test_dataset_processed

def textcnn_bi():
    import os
    import shutil
    from loguru import logger
    logger.remove()

    out_dir = 'temp/textcnn_bi'
    if os.path.exists(out_dir): shutil.rmtree(out_dir)
    m = TextCNNClassifier(output_dir=out_dir)
    train, val, test = prepare_data_jul_14_bi()
    m.train(train, num_epochs=100, evaluate_during_training_steps=1)
    predict_label, predict_proba = m.predict([line[1] for line in test])
    truth = [int(line[0]) for line in test]
    print("Five TextCNN Classification Report:\n", classification_report(truth, [int(i) for i in predict_label], digits=4))

def textrnn_bi():
    import os
    import shutil
    from loguru import logger
    logger.remove()

    out_dir = 'temp/textrnn_bi'
    if os.path.exists(out_dir): shutil.rmtree(out_dir)
    m = TextCNNClassifier(output_dir=out_dir)
    train, val, test = prepare_data_jul_14_bi()
    m.train(train, num_epochs=100, evaluate_during_training_steps=1)
    predict_label, predict_proba = m.predict([line[1] for line in test])
    truth = [int(line[0]) for line in test]
    print("Five TextRNN Classification Report:\n", classification_report(truth, [int(i) for i in predict_label], digits=4))

def textcnn_tri():
    import os
    import shutil
    from loguru import logger
    logger.remove()

    out_dir = 'temp/textcnn_tri'
    if os.path.exists(out_dir): shutil.rmtree(out_dir)
    m = TextCNNClassifier(output_dir=out_dir)
    train, val, test = prepare_data_jul_14_tri()
    m.train(train, num_epochs=100, evaluate_during_training_steps=1)
    predict_label, predict_proba = m.predict([line[1] for line in test])
    truth = [int(line[0]) for line in test]
    print("Tri TextCNN Classification Report:\n", classification_report(truth, [int(i) for i in predict_label], digits=4))

def textrnn_tri():
    import os
    import shutil
    from loguru import logger
    logger.remove()

    out_dir = 'temp/textrnn_tri'
    if os.path.exists(out_dir): shutil.rmtree(out_dir)
    m = TextCNNClassifier(output_dir=out_dir)
    train, val, test = prepare_data_jul_14_tri()
    m.train(train, num_epochs=100, evaluate_during_training_steps=1)
    predict_label, predict_proba = m.predict([line[1] for line in test])
    truth = [int(line[0]) for line in test]
    print("Tri TextRNN Classification Report:\n", classification_report(truth, [int(i) for i in predict_label], digits=4))


def textcnn_four_end():
    import os
    import shutil
    from loguru import logger
    logger.remove()

    out_dir = 'temp/textcnn_four_end'
    if os.path.exists(out_dir): shutil.rmtree(out_dir)
    m = TextCNNClassifier(output_dir=out_dir)
    train, val, test = prepare_data_jul_14_four_end()
    m.train(train, num_epochs=100, evaluate_during_training_steps=1)
    predict_label, predict_proba = m.predict([line[1] for line in test])
    truth = [int(line[0]) for line in test]
    print("Four End TextCNN Classification Report:\n", classification_report(truth, [int(i) for i in predict_label], digits=4))

def textrnn_four_end():
    import os
    import shutil
    from loguru import logger
    logger.remove()

    out_dir = 'temp/textrnn_four_end'
    if os.path.exists(out_dir): shutil.rmtree(out_dir)
    m = TextCNNClassifier(output_dir=out_dir)
    train, val, test = prepare_data_jul_14_four_end()
    m.train(train, num_epochs=100, evaluate_during_training_steps=1)
    predict_label, predict_proba = m.predict([line[1] for line in test])
    truth = [int(line[0]) for line in test]
    print("Four End TextRNN Classification Report:\n", classification_report(truth, [int(i) for i in predict_label], digits=4))


def textcnn_four():
    import os
    import shutil
    from loguru import logger
    logger.remove()

    out_dir = 'temp/textcnn_bi'
    if os.path.exists(out_dir): shutil.rmtree(out_dir)
    m = TextCNNClassifier(output_dir=out_dir)
    train, val, test = prepare_data_jul_14_four()
    m.train(train, num_epochs=100, evaluate_during_training_steps=1)
    predict_label, predict_proba = m.predict([line[1] for line in test])
    truth = [int(line[0]) for line in test]
    print("Five TextCNN Classification Report:\n", classification_report(truth, [int(i) for i in predict_label], digits=4))

def textrnn_four():
    import os
    import shutil
    from loguru import logger
    logger.remove()

    out_dir = 'temp/textrnn_bi'
    if os.path.exists(out_dir): shutil.rmtree(out_dir)
    m = TextCNNClassifier(output_dir=out_dir)
    train, val, test = prepare_data_jul_14_four()
    m.train(train, num_epochs=100, evaluate_during_training_steps=1)
    predict_label, predict_proba = m.predict([line[1] for line in test])
    truth = [int(line[0]) for line in test]
    print("Five TextRNN Classification Report:\n", classification_report(truth, [int(i) for i in predict_label], digits=4))

def textcnn_five():
    import os
    import shutil
    from loguru import logger
    logger.remove()

    out_dir = 'temp/textcnn'
    if os.path.exists(out_dir): shutil.rmtree(out_dir)
    m = TextCNNClassifier(output_dir=out_dir)
    train, val, test = prepare_data_jul_14_five()
    m.train(train, num_epochs=100, evaluate_during_training_steps=1)
    predict_label, predict_proba = m.predict([line[1] for line in test])
    truth = [int(line[0]) for line in test]
    print("Five TextCNN Classification Report:\n", classification_report(truth, [int(i) for i in predict_label], digits=4))

def textrnn_five():
    import os
    import shutil
    from loguru import logger
    logger.remove()

    out_dir = 'temp/textrnn'
    if os.path.exists(out_dir): shutil.rmtree(out_dir)
    m = TextCNNClassifier(output_dir=out_dir)
    train, val, test = prepare_data_jul_14_five()
    m.train(train, num_epochs=100, evaluate_during_training_steps=1)
    predict_label, predict_proba = m.predict([line[1] for line in test])
    truth = [int(line[0]) for line in test]
    print("Five TextRNN Classification Report:\n", classification_report(truth, [int(i) for i in predict_label], digits=4))

if __name__ == '__main__':
    # naive_bayes_bi()
    # perceptron_bi()
    # xgboost_bi()
    # perceptron_onehot_bi()
    # mnb_bi()
    # naive_bayes_ling()
    # perceptron_tri()
    # naive_bayes_tri()
    # bert_tri()
    textcnn_five()
