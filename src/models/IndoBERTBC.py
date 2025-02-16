import torch.nn as nn
from transformers import BertModel

class IndoBERTBC(nn.Module):
    def __init__(self, hidden_state:int=768, dropout:bool=True):
        super(IndoBERTBC, self).__init__()
        self.bert = BertModel.from_pretrained('indobenchmark/indobert-base-p2')
        self.classifier = nn.Linear(hidden_state, 1)
        self.dropout = dropout
        if self.dropout:
            self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask):
        logits = self.bert(input_ids = input_ids, attention_mask = attention_mask, token_type_ids = None)
        logits = self.classifier(logits.pooler_output)
        if self.dropout:
            logits = self.dropout(logits)
        return logits.squeeze()