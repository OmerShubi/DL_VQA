import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.utils.rnn import pack_padded_sequence


class VqaNet(nn.Module):
    """
    Based on paper - Show, Ask, Attend, and Answer: A Strong Baseline For Visual Question Answering
    """

    def __init__(self, cfg, embedding_tokens):
        super(VqaNet, self).__init__()
        text_cfg = cfg['text']
        image_cfg = cfg['image']
        attention_cfg = cfg['attention']
        classifier_cfg = cfg['classifier']

        lstm_out_features = text_cfg['question_features']
        if text_cfg['bidirectional']:
            lstm_out_features *= 2
        glimpses = attention_cfg['glimpses']
        # image_features = image_cfg['image_features']
        image_features = image_cfg['num_channels'][-1]

        self.text = questionNet(
            embedding_tokens=embedding_tokens,
            embedding_features=text_cfg['embedding_features'],
            lstm_features=text_cfg['question_features'],
            drop=text_cfg['dropout'],
            num_lstm_layers=text_cfg['num_lstm_layers'],
            bidirectional=text_cfg['bidirectional'])

        #TODO change!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        self.image = ImageNet2(image_cfg)

        self.attention = Attention(
            v_features=image_features,
            q_features=lstm_out_features,
            mid_features=attention_cfg['hidden_dim'],
            glimpses=glimpses,
            do_option=attention_cfg['do_option'],
            drop=attention_cfg['dropout'],
        )

        self.classifier = Classifier(
            in_features=glimpses * image_features + lstm_out_features,
            mid_features=classifier_cfg['hidden_dim'],
            out_features=cfg['max_answers'],
            drop=classifier_cfg['dropout'],
        )

    def forward(self, v, q, q_len):
        v = self.image(v)
        q = self.text(q, list(q_len.data))
        v = v / (v.norm(p=2, dim=1, keepdim=True).expand_as(v) + 1e-12)

        attention = self.attention(v, q)
        v = apply_attention(v, attention)

        combined = torch.cat([v, q], dim=1)
        answer = self.classifier(combined)

        return answer




class ImageNet2(nn.Sequential):
    def __init__(self, image_cng):
        super(ImageNet2, self).__init__()
        kernel_size = image_cng['kernel_size']
        num_channels = image_cng['num_channels']
        stride = image_cng['stride']

        for i in range(len(num_channels)-1):
            self.add_module(f'conv{i}', nn.Conv2d(in_channels=num_channels[i], out_channels=num_channels[i+1], kernel_size=kernel_size, stride=stride))
            self.add_module(f'relu{i}', nn.ReLU())
            self.add_module(f'maxpool{i}', nn.MaxPool2d(2, 2))

        self.add_module('drop', nn.Dropout(image_cng['dropout']))

class ImageNet(nn.Module):
    def __init__(self, image_cng):
        super(ImageNet, self).__init__()
        self.do_skip_connection = image_cng['do_skip_connection']
        kernel_size = image_cng['kernel_size']
        self.num_channels = image_cng['num_channels']
        self.stride = image_cng['stride']
        padding = kernel_size//2
        for i in range(len(self.num_channels)-1):
            if (self.do_skip_connection and i % 2 == 0) or not self.do_skip_connection:
                setattr(self, f'conv{i}', nn.Conv2d(in_channels=self.num_channels[i], out_channels=self.num_channels[i + 1],
                                                    kernel_size=kernel_size, padding=padding, stride=self.stride))
            else:
                setattr(self, f'conv{i}', nn.Conv2d(in_channels=self.num_channels[i], out_channels=self.num_channels[i+1],
                                                    kernel_size=kernel_size, padding=padding))
            setattr(self, f'relu{i}', nn.ReLU())

            if self.do_skip_connection:
                if (i + 1) % 2 == 0:
                    setattr(self, f'conv_skip{i}', nn.Conv2d(in_channels=self.num_channels[i-1], out_channels=self.num_channels[i+1],
                                                             kernel_size=1, stride=self.stride, bias=False))


        self.maxpool = nn.MaxPool2d(2, 2)

        self.dropout = nn.Dropout(image_cng['dropout'])

    def forward(self, x):

        for i in range(len(self.num_channels) - 1):
            if i % 2 == 0:
                x_orig = x
            x = getattr(self, f'conv{i}')(x)
            x = getattr(self, f'relu{i}')(x)
            if self.do_skip_connection:
                if (i+1) % 2 == 0:
                    x_orig = getattr(self, f'conv_skip{i}')(x_orig)
                    x += x_orig
                    if self.stride == 1:
                        x = self.maxpool(x)
            else:
                if self.stride == 1:
                    x = self.maxpool(x)

        x = self.dropout(x)

        return x


class questionNet(nn.Module):
    def __init__(self, embedding_tokens, embedding_features, lstm_features, num_lstm_layers, drop, bidirectional):
        super(questionNet, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=embedding_tokens, embedding_dim=embedding_features, padding_idx=0)
        self.drop = nn.Dropout(drop)
        self.tanh = nn.Tanh()
        self.lstm = nn.LSTM(input_size=embedding_features,
                            hidden_size=lstm_features,
                            num_layers=num_lstm_layers, dropout=drop, bidirectional=bidirectional)

    def forward(self, q, q_len):
        embedded = self.embedding(q)
        embedded_drop = self.drop(embedded)
        tanhed = self.tanh(embedded_drop)
        packed = pack_padded_sequence(tanhed, q_len, batch_first=True, enforce_sorted=False)
        _, (_, c) = self.lstm(packed)
        return c.transpose(0, 1).flatten(1)


class Attention(nn.Module):
    def __init__(self, v_features, q_features, mid_features, glimpses, do_option, drop=0.0):
        super(Attention, self).__init__()
        self.do_option = do_option
        self.v_conv = nn.Conv2d(in_channels=v_features, out_channels=mid_features, kernel_size=1, bias=False)  # let self.lin take care of bias
        self.q_lin = nn.Linear(in_features=q_features, out_features=mid_features)
        if self.do_option == '|':
            self.x_conv = nn.Conv2d(in_channels=2*mid_features, out_channels=glimpses, kernel_size=1)
        else:
            self.x_conv = nn.Conv2d(in_channels=mid_features, out_channels=glimpses, kernel_size=1)

        self.drop = nn.Dropout(drop)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, v, q):

        v = self.v_conv(self.drop(v))
        q = self.q_lin(self.drop(q))
        q = tile_2d_over_nd(q, v)
        if self.do_option == '*':
            x = self.relu(v * q)
        elif self.do_option == '+':
            x = self.relu(v + q)
        elif self.do_option == '|':
            x = self.relu(torch.cat([v,q], dim=1))
        x = self.x_conv(self.drop(x))
        return x


class Classifier(nn.Sequential):
    def __init__(self, in_features, mid_features, out_features, drop=0.0):
        super(Classifier, self).__init__()
        self.add_module('drop1', nn.Dropout(drop))
        self.add_module('lin1', nn.Linear(in_features=in_features, out_features=mid_features))
        self.add_module('relu', nn.ReLU())
        self.add_module('drop2', nn.Dropout(drop))
        self.add_module('lin2', nn.Linear(in_features=mid_features, out_features=out_features))


def apply_attention(input_, attention):
    """ Apply any number of attention maps over the input. """
    n, c = input_.size()[:2]
    glimpses = attention.size(1)

    # flatten the spatial dims into the third dim, since we don't need to care about how they are arranged
    input_ = input_.view(n, 1, c, -1)  # [n, 1, c, s]
    attention = attention.view(n, glimpses, -1)
    attention = F.softmax(attention, dim=-1).unsqueeze(2)  # [n, g, 1, s]
    weighted = attention * input_  # [n, g, v, s]
    weighted_mean = weighted.sum(dim=-1)  # [n, g, v]
    return weighted_mean.view(n, -1)


def tile_2d_over_nd(feature_vector, feature_map):
    """ Repeat the same feature vector over all spatial positions of a given feature map.
        The feature vector should have the same batch size and number of features as the feature map.
    """
    n, c = feature_vector.size()
    spatial_size = feature_map.dim() - 2
    tiled = feature_vector.view(n, c, *([1] * spatial_size)).expand_as(feature_map)
    return tiled

# TODO make look like own
