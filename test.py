import torch
import torch.nn as nn
import random
import numpy as np
import pandas as pd
import sys
from torch.utils.data import  DataLoader, TensorDataset, random_split, SubsetRandomSampler
from transformers import BertTokenizer, AdamW
from sklearn.model_selection import KFold
from src.utils.function import data_to_tensor, train, validate
from tqdm import tqdm
from src.models.IndoBERTBC import IndoBERTBC
from src.models.IndoBERTBiC import IndoBERTBiC
from src.models.IndoBERTCC import IndoBERTCC
import warnings
warnings.filterwarnings("ignore")

# Parse command-line arguments
if len(sys.argv) != 4:
    print("Usage: python test.py <typeModel> <epochs> <batch_size>")
    sys.exit(1)

type_model_input = sys.argv[1]
epochs = int(sys.argv[2])
batch_size = int(sys.argv[3])

# Mapping input string to model class
model_mapping = {
    "bc": "IndoBERTBC",
    "cc": "IndoBERTBiC",
    "bic": "IndoBERTCC"
}

type_model = model_mapping.get(type_model_input, None)
if type_model is None:
    print("Invalid model type. Choose from: bc, cc, bic")
    sys.exit(1)

# Set random seed for reproducibility
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using CUDA: {torch.cuda.is_available()}")

# Load dataset
df = pd.read_csv("src/dataset/hoax-news-processed.csv").dropna()
df = df[:20]
tokenizer = BertTokenizer.from_pretrained('indobenchmark/indobert-base-p2', do_lower_case=True)
features, labels = df.clean_narasi.values, df.hoax.values
input_ids, attention_masks, labels = data_to_tensor(tokenizer, features, labels)

dataset = TensorDataset(input_ids, attention_masks, labels)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(SEED))

print(f"{train_size} training samples")
print(f"{val_size} validation samples")

# K-Fold Cross Validation setup
splits = KFold(n_splits=5, shuffle=True, random_state=SEED)

fold_results = {
    'train_loss': [], 'test_loss': [],
    'train_acc': [], 'test_acc': [],
    'test_rocauc': [], 'test_f1': []
}

from tqdm import tqdm

split_indices = list(splits.split(np.arange(len(dataset))))  # Convert generator to list
pbar = tqdm(total=len(split_indices), desc="K-Fold Training")

for fold, (train_idx, val_idx) in enumerate(split_indices):
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)

    train_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    val_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)

    if type_model_input == "bc":
        model = IndoBERTBC().to(device)
    elif type_model_input == "bic":
        model = IndoBERTBiC().to(device)
    else:
        model = IndoBERTCC().to(device)

    optimizer = AdamW(model.parameters(), lr=5e-5, eps=1e-5)
    criterion = nn.BCEWithLogitsLoss()

    fold_stats = {'train_loss': 0, 'test_loss': 0, 'train_acc': 0, 'test_acc': 0, 'test_rocauc': 0, 'test_f1': 0}

    for epoch in range(epochs):
        train_acc, train_loss = train(model, train_dataloader, optimizer, criterion, device)
        val_acc, val_loss, val_rocauc, val_f1 = validate(model, val_dataloader, criterion, device)

        if val_f1 > fold_stats['test_f1']:
            fold_stats['train_loss'] = train_loss
            fold_stats['test_loss'] = val_loss
            fold_stats['train_acc'] = train_acc
            fold_stats['test_acc'] = val_acc
            fold_stats['test_rocauc'] = val_rocauc
            fold_stats['test_f1'] = val_f1

    for key in fold_results.keys():
        fold_results[key].append(fold_stats[key])

    step = fold + 1
    pbar.set_postfix({
        "Avg Train Loss": f"{sum(fold_results['train_loss'])/step:.4f}",
        "Avg Train Acc": f"{sum(fold_results['train_acc'])/step:.4f}",
        "Avg Test Loss": f"{sum(fold_results['test_loss'])/step:.4f}",
        "Avg Test Acc": f"{sum(fold_results['test_acc'])/step:.4f}",
        "Avg Test ROC-AUC": f"{sum(fold_results['test_rocauc'])/step:.4f}",
        "Avg Test F1": f"{sum(fold_results['test_f1'])/step:.4f}"
    })
    
    pbar.update(1)  # Increment progress bar

pbar.close()  # Close progress bar after training is complete
