from main import *

def evaluate(test_dataset, model):
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
    pred, target = [], []
    for batch in test_loader:
        sent, label = batch
        logits = model(sent)
        pred += torch.argmax(logits, dim=-1).detach().cpu().tolist()
        target += label.tolist()

    acc = np.sum(np.array(pred) == np.array(target)) / len(pred)
    print("Classification Report:")
    print(classification_report(target, pred, digits=4))

if __name__ == '__main__':
    '''
    test_data = torch.load(args.test_dataset, weights_only=False)
    for transformer_name in ['google-bert/bert-base-chinese', 'hfl/chinese-roberta-wwm-ext', 'Jihuai/bert-ancient-chinese', 'ethanyt/guwenbert-base']:
        save_checkpoint = os.path.join('logs', transformer_name.split('/')[-1], 'guwen-bert.pt')
        model = BertClassifier(num_classes=2, transformer_name=transformer_name)
        state_dict = torch.load(save_checkpoint, weights_only=True)
        model.load_state_dict(state_dict)
        model.to(model.device)
        model.eval()
        print(transformer_name)
        evaluate(test_data, model)

    test_data = torch.load(args.test_tri_dataset, weights_only=False)
    for transformer_name in ['google-bert/bert-base-chinese', 'hfl/chinese-roberta-wwm-ext', 'Jihuai/bert-ancient-chinese', 'ethanyt/guwenbert-base']:
        save_checkpoint = os.path.join('logs', transformer_name.split('/')[-1], 'triple-guwen-bert.pt')
        model = BertClassifier(num_classes=3, transformer_name=transformer_name)
        state_dict = torch.load(save_checkpoint, weights_only=True)
        model.load_state_dict(state_dict)
        model.to(model.device)
        model.eval()
        print(transformer_name)
        evaluate(test_data, model)

    test_data = torch.load(args.test_second_four_dataset, weights_only=False)
    for transformer_name in ['google-bert/bert-base-chinese', 'hfl/chinese-roberta-wwm-ext', 'Jihuai/bert-ancient-chinese', 'ethanyt/guwenbert-base']:
        save_checkpoint = os.path.join('logs', transformer_name.split('/')[-1], 'second_four-guwen-bert.pt')
        model = BertClassifier(num_classes=4, transformer_name=transformer_name)
        state_dict = torch.load(save_checkpoint, weights_only=True)
        model.load_state_dict(state_dict)
        model.to(model.device)
        model.eval()
        print(transformer_name)
        evaluate(test_data, model)

    test_data = torch.load(args.test_four_dataset, weights_only=False)
    for transformer_name in ['google-bert/bert-base-chinese', 'hfl/chinese-roberta-wwm-ext', 'Jihuai/bert-ancient-chinese', 'ethanyt/guwenbert-base']:
        save_checkpoint = os.path.join('logs', transformer_name.split('/')[-1], 'four-guwen-bert.pt')
        model = BertClassifier(num_classes=4, transformer_name=transformer_name)
        state_dict = torch.load(save_checkpoint, weights_only=True)
        model.load_state_dict(state_dict)
        model.to(model.device)
        model.eval()
        print(transformer_name)
        evaluate(test_data, model)

    test_data = torch.load(args.test_five_dataset, weights_only=False)
    for transformer_name in ['google-bert/bert-base-chinese', 'hfl/chinese-roberta-wwm-ext', 'Jihuai/bert-ancient-chinese', 'ethanyt/guwenbert-base']:
        save_checkpoint = os.path.join('logs', transformer_name.split('/')[-1], 'five-guwen-bert.pt')
        model = BertClassifier(num_classes=5, transformer_name=transformer_name)
        state_dict = torch.load(save_checkpoint, weights_only=True)
        model.load_state_dict(state_dict)
        model.to(model.device)
        model.eval()
        print(transformer_name)
        evaluate(test_data, model)

    '''
    # analyze the confusion matrix
    import numpy as np
    import pandas as pd
    from sklearn.metrics import confusion_matrix, classification_report
    import seaborn as sns
    import matplotlib.pyplot as plt
    from collections import defaultdict

    save_checkpoint = os.path.join('logs', 'guwenbert-base'.split('/')[-1], 'five-guwen-bert.pt')
    model = BertClassifier(num_classes=5, transformer_name='ethanyt/guwenbert-base')
    state_dict = torch.load(save_checkpoint, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(model.device)
    model.eval()

    test_dataset = torch.load(args.test_five_dataset, weights_only=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
    examples, predicted_labels, true_labels = [], [], []
    evaluate(test_dataset, model)
    for batch in test_loader:
        sent, label = batch
        logits = model(sent)
        examples += sent
        predicted_labels += torch.argmax(logits, dim=-1).detach().cpu().tolist()
        true_labels += label.tolist()
    labels = sorted(set(true_labels + predicted_labels))
    cm = confusion_matrix(true_labels, predicted_labels, labels = sorted(set(true_labels + predicted_labels)))
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_df.to_csv('temp/confusion_matrix.csv', index=False)

    # Error analysis: collect misclassified examples
    print("\nFalse Predictions:\n")
    error_dict = defaultdict(list)
    for true, pred, example in zip(true_labels, predicted_labels, examples):
        if true != pred:
            error_dict[(true, pred)].append(example)

    import csv
    with open('temp/error_analysis.csv', 'w', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['True', 'Predicted', 'Entry'])
        class2label = {0: 'Non-Riot', 1: 'Secret Society', 2: 'Militia', 3: 'Peasant', 4: 'Others'}
        for (true, pred), exs in error_dict.items():
            print(f"True: {class2label[true]} → Predicted: {class2label[pred]}")
            for ex in exs:  # Show up to 3 per error type
                print(f"{ex}")
                writer.writerow([class2label[true], class2label[pred], ex])
            print('-' * 50)

    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    # New confusion matrix data
    new_labels = ["No Unrest", "Secret Society", "Militia", "Peasant", "Others"]
    new_confusion_data = [
        [145, 3, 4, 4, 0],
        [1, 30, 4, 2, 0],
        [0, 0, 79, 2, 1],
        [1, 0, 2, 29, 0],
        [4, 1, 0, 3, 12]
    ]

    df_new_cm = pd.DataFrame(new_confusion_data, index=[f"True {label}" for label in new_labels],
                            columns=[f"Pred {label}" for label in new_labels])

    # Plotting the updated confusion matrix without headline
    plt.figure(figsize=(10, 8))
    sns.set(font_scale=1.2)
    sns.heatmap(df_new_cm, annot=True, fmt="d", cmap="Blues", cbar=True, linewidths=.5, square=True)

    plt.title("")
    plt.ylabel("Actual Class", fontsize=14)
    plt.xlabel("Predicted Class", fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('temp/confusion_matrix_updated.png', dpi=300)