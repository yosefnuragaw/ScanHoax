import torch.nn as nn
from transformers import BertModel
from src.models.BiLSTM import BiLSTM
import torch

class IndoBERTCC(nn.Module):
    def __init__(self, hidden_state:int=768,cat_size:int=8, dropout:bool=True):
        super(IndoBERTCC, self).__init__()
        self.bert = BertModel.from_pretrained('indobenchmark/indobert-base-p2')
        self.bilstm = BiLSTM(hidden_state,cat_size,dropout)
        self.classifier = nn.Linear(hidden_state, cat_size)  
        self.dropout = dropout
        if self.dropout:
            self.dropout = nn.Dropout(0.1)


    def forward(self, input_ids, attention_mask):
        logits = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        lstm_logits = self.bilstm(logits.last_hidden_state)
        pooled_logits = self.classifier(logits.pooler_output)
        if self.dropout:
            pooled_logits = self.dropout(pooled_logits)

        concatenated_tensor = torch.cat((lstm_logits, pooled_logits), dim=1)
        logits = torch.mean(concatenated_tensor, dim=1)
        return logits.squeeze()