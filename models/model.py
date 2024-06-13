import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class TextCNN(nn.Module):
    """
    CNN text classification model, based on the paper.
    """

    def __init__(self, vocab_size=23,embedding_dim=512, num_filters=256, kernel_sizes=[3, 4, 5], num_classes=2):
        super(TextCNN, self).__init__()

        V = vocab_size
        E = embedding_dim
        Nf = num_filters
        Ks = kernel_sizes
        C = num_classes

        self.embedding = nn.Embedding(V, E)  # embedding layer

        # three different convolutional layers
        self.convs = nn.ModuleList([nn.Conv1d(E, Nf, k) for k in Ks])
        self.dropout = nn.Dropout()  # a dropout layer

    @staticmethod
    def conv_and_max_pool(x, conv):
        """Convolution and global max pooling layer"""
        return F.relu(conv(x).permute(0, 2, 1).max(1)[0])

    def forward(self, inputs):
        # Conv1d takes in (batch, channels, seq_len), but raw embedded is (batch, seq_len, channels)
        embedded = self.embedding(inputs).permute(0, 2, 1)
        x = [self.conv_and_max_pool(embedded, k) for k in self.convs]  # convolution and global max pooling
        x = self.dropout(torch.cat(x, 1))
        return x


class LSTM(nn.Module):
    def __init__(self, vocab_size=23,embedding_dim=512, hidden_dim=256, num_layers=2):
        super(LSTM, self).__init__()

        self.dimension = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, bidirectional=True, batch_first=True)
        self.drop = nn.Dropout(p=0.5)

    def forward(self, x):
        text_len = self.count_non_zero_elements_per_batch(x)
        x = self.embedding(x)
        output, _ = self.lstm(x)

        out_forward = output[range(len(output)), text_len - 1, :self.dimension]
        out_reverse = output[:, 0, self.dimension:]
        out_reduced = torch.cat((out_forward, out_reverse), 1)
        text_fea = self.drop(out_reduced)
        return text_fea

    def count_non_zero_elements_per_batch(self, tensor_seq_2d):
        tensor_seq_2d = tensor_seq_2d.cpu().tolist()
        lengths_non_zero = [sum(1 for x in seq if x != 0) for seq in tensor_seq_2d]
        return torch.tensor(lengths_non_zero)


class T4Seeker(nn.Module):
    def __init__(self,ML_dim = 1220 + 320, out_dim = 1024, num_classes=2):
        super(T4Seeker, self).__init__()
        self.textcnn = TextCNN()
        self.lstm = LSTM()
        combined_features_dim = self.textcnn.convs[0].out_channels * len(self.textcnn.convs) + self.lstm.dimension * 2

        ML_dim = 1024 + 1220

      #  ML_dim = 320 + 1220
       # out_dim = 1024
        self.fc_features = nn.Linear(ML_dim, out_dim)
        self.fc = nn.Linear(combined_features_dim+ML_dim-768, combined_features_dim )
        self.drop = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(combined_features_dim, num_classes)

    def forward(self, x,features):
        textcnn_features = self.textcnn(x)
        lstm_features = self.lstm(x)
       # features = self.fc_features(features)
        combined_features = torch.cat(( lstm_features,features), dim=1)
        output = self.fc(combined_features)
        output = self.drop(output)
        output = self.fc2(output)
        return output,combined_features
