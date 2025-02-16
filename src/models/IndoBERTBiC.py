import torch.nn as nn
from transformers import BertModel
from src.models.BiLSTM import BiLSTM

class IndoBERTBiC(nn.Module):
    def __init__(self, hidden_state:int=768, output:int=1):
        super(IndoBERTBiC, self).__init__()
        self.bert = BertModel.from_pretrained('indobenchmark/indobert-base-p2')
        self.bilstm = BiLSTM(hidden_state,output)

    def forward(self, input_ids, attention_mask):
        logits = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        lstm_logits = self.bilstm(logits.last_hidden_state)
        return lstm_logits.squeeze()