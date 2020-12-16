"""
    Example for a simple model
"""

from abc import ABCMeta
from nets.fc import FCNet
from torch import nn, Tensor

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.utils.rnn import pack_padded_sequence


class Net(nn.Module):
    """ Re-implementation of ``Show, Ask, Attend, and Answer: A Strong Baseline For Visual Question Answering'' [0]

    [0]: https://arxiv.org/abs/1704.03162
    TODO use and delete https://github.com/Cyanogenoid/vqa-counting/blob/master/vqa-v2/model.py
    """

    def __init__(self, cfg, embedding_tokens):
        super(Net, self).__init__()
        question_features = cfg['question_features']
        image_features = cfg['image_features']
        glimpses = cfg['glimpses']
        dropouts = cfg['dropouts']
        self.text = TextProcessor(
            embedding_tokens=embedding_tokens,
            embedding_features=cfg['embedding_features'],
            lstm_features=question_features,
            drop=dropouts['text'],
        )
        self.attention = Attention(
            v_features=image_features,
            q_features=question_features,
            mid_features=cfg['attention_hidden_dim'],
            glimpses=glimpses,
            drop=dropouts['attention'],
        )
        self.classifier = Classifier(
            in_features=glimpses * image_features + question_features,
            mid_features=cfg['classifier_hidden_dim'],
            out_features=cfg['max_answers'],
            drop=dropouts['classifier'],
        )

        # xavier_uniform_ init for linear and conv layers
        for m in self.modules(): # TODO need?
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, v, q, q_len):
        q = self.text(q, list(q_len.data))
        v = v / (v.norm(p=2, dim=1, keepdim=True).expand_as(v) + 1e-8) # TODO ??

        attention = self.attention(v, q)
        v = apply_attention(v, attention)

        combined = torch.cat([v, q], dim=1)
        answer = self.classifier(combined)

        return answer


class Classifier(nn.Sequential):
    def __init__(self, in_features, mid_features, out_features, drop=0.0):
        super(Classifier, self).__init__()
        self.add_module('drop1', nn.Dropout(drop))
        self.add_module('lin1', nn.Linear(in_features=in_features, out_features=mid_features))
        self.add_module('relu', nn.ReLU())
        self.add_module('drop2', nn.Dropout(drop))
        self.add_module('lin2', nn.Linear(in_features=mid_features, out_features=out_features))


class TextProcessor(nn.Module):
    def __init__(self, embedding_tokens, embedding_features, lstm_features, drop=0.0):
        super(TextProcessor, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=embedding_tokens, embedding_dim=embedding_features, padding_idx=0)
        self.drop = nn.Dropout(drop)
        self.tanh = nn.Tanh()
        self.lstm = nn.LSTM(input_size=embedding_features,
                            hidden_size=lstm_features,
                            num_layers=1) # TODO dropout, num of hidden layer, b-directional
        self.features = lstm_features # TODO output lstm dimension ??

        # TODO need?
        # xavier_uniform_ init
        self._init_lstm(self.lstm.weight_ih_l0)
        self._init_lstm(self.lstm.weight_hh_l0)

        self.lstm.bias_ih_l0.data.zero_()
        self.lstm.bias_hh_l0.data.zero_()

        init.xavier_uniform_(self.embedding.weight)

    def _init_lstm(self, weight):
        for w in weight.chunk(4, 0):
            init.xavier_uniform_(w)

    def forward(self, q, q_len):
        embedded = self.embedding(q)
        embedded_drop = self.drop(embedded)
        tanhed = self.tanh(embedded_drop)
        packed = pack_padded_sequence(tanhed, q_len, batch_first=True) # TODO understand padding and packed object
        _, (_, c) = self.lstm(packed)
        return c.squeeze(0)


class Attention(nn.Module):
    def __init__(self, v_features, q_features, mid_features, glimpses, drop=0.0):
        super(Attention, self).__init__()
        self.v_conv = nn.Conv2d(in_channels=v_features, out_channels=mid_features, kernel_size=1, bias=False)  # let self.lin take care of bias
        self.q_lin = nn.Linear(in_features=q_features, out_features=mid_features)
        self.x_conv = nn.Conv2d(in_channels=mid_features, out_channels=glimpses, kernel_size=1)

        self.drop = nn.Dropout(drop)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, v, q):

        v = self.v_conv(self.drop(v)) # todo conv only on V
        q = self.q_lin(self.drop(q))
        q = tile_2d_over_nd(q, v)
        x = self.relu(v + q) # todo why + and not cat?
        x = self.x_conv(self.drop(x))
        return x


def apply_attention(input, attention):
    """ Apply any number of attention maps over the input. """
    n, c = input.size()[:2]
    glimpses = attention.size(1)

    # flatten the spatial dims into the third dim, since we don't need to care about how they are arranged
    input = input.view(n, 1, c, -1) # [n, 1, c, s]
    attention = attention.view(n, glimpses, -1)
    attention = F.softmax(attention, dim=-1).unsqueeze(2) # [n, g, 1, s]
    weighted = attention * input # [n, g, v, s]
    weighted_mean = weighted.sum(dim=-1) # [n, g, v]
    return weighted_mean.view(n, -1)


def tile_2d_over_nd(feature_vector, feature_map):
    """ Repeat the same feature vector over all spatial positions of a given feature map.
        The feature vector should have the same batch size and number of features as the feature map.
    """
    n, c = feature_vector.size()
    spatial_size = feature_map.dim() - 2
    tiled = feature_vector.view(n, c, *([1] * spatial_size)).expand_as(feature_map)
    return tiled


