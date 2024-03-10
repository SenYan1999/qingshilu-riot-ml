import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoConfig


class BertClassifier(nn.Module):
    def __init__(self, num_classes, transformer_name, device='cuda:0'):
        super(BertClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(transformer_name)
        self.hidden_dim = AutoConfig.from_pretrained(transformer_name).hidden_size
        self.tokenizer = AutoTokenizer.from_pretrained(transformer_name)
        self.classifier = nn.Linear(self.hidden_dim, num_classes)
        self.device = device

    def forward(self, sent):
        input_ids = self.prepare_inputs(sent)
        encoding = self.bert(**input_ids).pooler_output
        # encoding = nn.functional.avg_pool1d(encoding.permute(0, 2, 1), encoding.shape[1]).squeeze()
        logits = self.classifier(encoding)
        return logits

    def prepare_inputs(self, sent):
        input_ids = self.tokenizer(sent, max_length=512, padding='longest', truncation=True, return_tensors='pt').to(
            self.device)
        return input_ids
