import datetime

from sklearn.metrics import f1_score, roc_auc_score
import torch
import torch.nn as nn

def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


def data_to_tensor(tokenizer,features,labels):
  input_ids = []
  attention_masks = []
  for feature in features:
      encoded_dict = tokenizer.encode_plus(
                          feature,                      
                          add_special_tokens = True, 
                          max_length = 512,           
                          pad_to_max_length = True,
                          return_attention_mask = True,  
                          truncation = True,
                          return_tensors = 'pt',     
                    )
      input_ids.append(encoded_dict['input_ids'])
      attention_masks.append(encoded_dict['attention_mask'])
  input_ids = torch.cat(input_ids, dim=0)
  attention_masks = torch.cat(attention_masks, dim=0)
  labels = torch.tensor(labels)
  return input_ids,attention_masks,labels


def train(model = None, train_data = None,optimizer = None, criterion = None, device=None):  
  sigmoid = nn.Sigmoid()

  total_train_loss =0
  total_train_accuracy = 0
  y_true = []
  y_pred_label = []
  model.train()

  counter = 0
  for step, batch in enumerate(train_data):
        input_ids = batch[0].to(device)
        input_mask = batch[1].to(device)
        labels = batch[2].to(device)
        counter += len(labels)

        model.zero_grad()
        out = model(input_ids = input_ids, attention_mask = input_mask)
#         return out
        loss = criterion(out,labels.float())
        total_train_loss += loss.item()
        out = out.to(device)
        labels = labels.to(device)

        binary_predictions = torch.where(sigmoid(out) >= 0.5, torch.tensor(1).to(device), torch.tensor(0).to(device))
        total_train_accuracy +=  torch.sum(binary_predictions == labels).item()

        y_true.extend(labels.flatten())
        y_pred_label.extend(binary_predictions.tolist())
        if (step > 0) and (step % 100 == 0 or step == len(train_data)):
          print(f"step {step}/{len(train_data)} [total_train_accuracy : {total_train_accuracy/counter} loss : {total_train_loss/counter}]")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

  avg_train_accuracy = total_train_accuracy/counter
  avg_train_loss = total_train_loss/counter

  return avg_train_accuracy,avg_train_loss
def validate(model=None,val_data=None,criterion = None, device=None):
    sigmoid = nn.Sigmoid()
    model.eval()
    total_eval_accuracy = 0
    total_eval_loss = 0
    y_true = []
    y_pred = []
    y_pred_label = []
    counter = 0

    for step,batch in enumerate(val_data):
        input_ids = batch[0].to(device)
        input_mask = batch[1].to(device)
        labels = batch[2].to(device)
        counter += len(labels)

        with torch.no_grad():
            out = model(input_ids = input_ids, attention_mask = input_mask)
        loss = criterion(out, labels.float())
        total_eval_loss += loss.item()
        out = out.to(device)
        labels = labels.to(device)
        
        binary_predictions = torch.where(sigmoid(out) >= 0.5, torch.tensor(1).to(device), torch.tensor(0).to(device))
        total_eval_accuracy += torch.sum(binary_predictions == labels).item()
        y_true.append(labels.flatten())
        y_pred.append(sigmoid(out).flatten())
        y_pred_label.extend(binary_predictions.tolist())



    avg_val_accuracy = total_eval_accuracy/counter
    avg_val_loss = total_eval_loss/counter
    y_true = torch.cat(y_true).tolist()
    y_pred = torch.cat(y_pred).tolist()
    roc_auc = roc_auc_score(y_true,y_pred)
    f1      = f1_score (y_true,y_pred_label)
    return avg_val_accuracy,avg_val_loss,roc_auc,f1