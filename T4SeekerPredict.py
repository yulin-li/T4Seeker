import argparse
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score, precision_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

import numpy as np
from models.model import  T4Seeker


class ProteinDataset(Dataset):
    def __init__(self, file, fileml, selected_feature_indices=None, kmers=1, max_len=1000):
        df = pd.read_csv(file)
        df2 = pd.read_csv(fileml[0], dtype=np.float32).iloc[:, :-1]
        df3 = pd.read_csv(fileml[1], dtype=np.float32).iloc[:, :-1]
        if not selected_feature_indices:
            selected_feature_indices = np.load('models/selected_feature_indices.npy')
        df3 = df3.iloc[:, selected_feature_indices]
        df4 = pd.concat([df2, df3], axis=1)
        self.kmers = kmers
        self.max_len = max_len

        self.seqs = df['Sequence'].values
        self.labels = df['Label'].values
        self.vocab = self.create_vocab_by_amino_acids()
        self.features = self.standardize(df4.values)

    def standardize(self, x):
        scaler = StandardScaler()
        self.features = scaler.fit_transform(x)
        return self.features

    def create_vocab_by_amino_acids(self,amino_acids=None):
        if amino_acids is None:
            amino_acids = [
                'A', 'C', 'D', 'E', 'F',
                'G', 'H', 'I', 'K', 'L',
                'M', 'N', 'P', 'Q', 'R',
                'S', 'T', 'V', 'W', 'X',
                'Y'
            ]
        ##aa to dicts
        vocab = {}
        # vocab['[CLS]'] = len(vocab)
        # vocab['[SEP]'] = len(vocab)
        vocab['[PAD]'] = len(vocab)
        vocab['[UNK]'] = len(vocab)
        vocab.update({aa: idx + len(vocab) for idx, aa in enumerate(amino_acids)})  # Starting index from 1

        return vocab
    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq = self.seqs[idx]
        label = self.labels[idx]
        tokens = self.tokenize_sequence(seq, self.kmers, self.max_len)
        token_ids = self.sequence_to_ids(tokens, self.vocab)
        return torch.tensor(token_ids, dtype=torch.long), torch.tensor(self.features[idx], dtype=torch.float32), torch.tensor(label, dtype=torch.long)

    def tokenize_sequence(self,seq, k, max_len=1000):
        if len(seq) >= max_len:
            seq = seq[:max_len]

        tokens = [seq[i:i + k] for i in range(len(seq) - k + 1)]  # + ['[SEP]']
        if len(seq) < max_len:
            seq_pads = ['[PAD]'] * (max_len - len(seq))
            tokens += seq_pads  # + ['[SEP]']
        #  tokens += ['[SEP]']
        return tokens

    def sequence_to_ids(self,tokens, vocab):
        return [vocab[token] if token in vocab else vocab['[UNK]'] for token in tokens]

def evaluate_model(model, dataloader, device, test=False):
    model.eval()
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for seqs, features, labels in dataloader:
            seqs = seqs.to(device)
            features = features.to(device)
            labels = labels.to(device)
            outputs, _ = model(seqs, features)

            all_labels.append(labels.detach().cpu().numpy())
            all_probs.append(outputs.detach().cpu().numpy())

    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)
    accuracy = accuracy_score(all_labels, np.argmax(all_probs, axis=1))
    auc = roc_auc_score(all_labels, all_probs[:, 1])

    if test:
        f1 = f1_score(all_labels, np.argmax(all_probs, axis=1))
        recall = recall_score(all_labels, np.argmax(all_probs, axis=1))
        precision = precision_score(all_labels, np.argmax(all_probs, axis=1))

        ##calculate  Specificity
        tn, fp, fn, tp = confusion_matrix(all_labels, np.argmax(all_probs, axis=1)).ravel()
        specificity = tn / (tn + fp)

        return specificity, accuracy, auc, f1, recall, precision
    return accuracy, auc

def main(args):

    gpu = True
    if gpu:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = 'cpu'

    testdataset = ProteinDataset(args.fasta_test, [args.DR_test, args.ESM_test])
    testdataloader = DataLoader(testdataset, batch_size=32, shuffle=False)

    model = T4Seeker()
    model.to(device)

    checkpoint = torch.load('./models/model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])

    test_specificity, test_acc, test_auc, test_f1, test_recall, test_precision = evaluate_model(model, testdataloader, device, test=True)
    print(f"Test: Acc: {test_acc:.4f},  F1: {test_f1:.4f}, Recall: {test_recall:.4f}, Precision: {test_precision:.4f}, Specificity: {test_specificity:.4f}, AUC: {test_auc:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict T4se')
    parser.add_argument('--fasta_test', type=str, required=True, help='Path to the test fasta file')
    parser.add_argument('--DR_test', type=str, required=True, help='Path to the DR test features file')
    parser.add_argument('--ESM_test', type=str, required=True, help='Path to the ESM test features file')

    args = parser.parse_args()
    main(args)
