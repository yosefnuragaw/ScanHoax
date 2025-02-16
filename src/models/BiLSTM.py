import torch.nn as nn

class BiLSTM(nn.Module):
    def __init__(self, hidden_state:int=768, output:int=1, dropout:bool=True):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_state
        self.bilstm = nn.LSTM(hidden_state, hidden_state, bidirectional=True, batch_first = True)
        self.classifier = nn.Linear(hidden_state*2,output)
        self.dropout = dropout
        if self.dropout:
            self.dropout = nn.Dropout(0.1)

    def forward(self, last_hidden_state):
        logits, _ = self.bilstm(last_hidden_state)
        logits = self.classifier(logits[:,0,:])
        if self.dropout:
            logits = self.dropout(logits)
        return logits