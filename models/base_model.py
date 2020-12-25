import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.utils.rnn import pack_padded_sequence

# TODO action item - speed
"""
batch size - increase till memory crash

"""


class Net(nn.Module):
    """ Re-implementation of ``Show, Ask, Attend, and Answer: A Strong Baseline For Visual Question Answering'' [0]

    [0]: https://arxiv.org/abs/1704.03162
    TODO use and delete https://github.com/Cyanogenoid/vqa-counting/blob/master/vqa-v2/model.py
    """

    def __init__(self, cfg, embedding_tokens):
        super(Net, self).__init__()
        text_cfg = cfg['text']
        image_cfg = cfg['image']
        attention_cfg = cfg['attention']
        classifier_cfg = cfg['classifier']

        question_features = text_cfg['question_features']
        image_features = image_cfg['image_features']
        glimpses = attention_cfg['glimpses']

        self.text = TextProcessor(
            embedding_tokens=embedding_tokens,
            embedding_features=text_cfg['embedding_features'],
            lstm_features=question_features,
            drop=text_cfg['dropout'],
        )
        #TODO change
        # self.image = GoogLeNet()
        self.image = ImageConv()

        self.attention = Attention(
            v_features=image_features,
            q_features=question_features,
            mid_features=attention_cfg['hidden_dim'],
            glimpses=glimpses,
            drop=attention_cfg['dropout'],
        )

        self.classifier = Classifier(
            in_features=glimpses * image_features + question_features,
            mid_features=classifier_cfg['hidden_dim'],
            out_features=cfg['max_answers'],
            drop=classifier_cfg['dropout'],
        )

        # xavier_uniform_ init for linear and conv layers
        # for m in self.modules(): # TODO need?
        #     if isinstance(m, nn.xavier_uniform_) or isinstance(m, nn.Conv2d):
        #         init.xavier_uniform_(m.weight)
        #         if m.bias is not None:
        #             m.bias.data.zero_()

    def forward(self, v, q, q_len):
        v = self.image(v)
        q = self.text(q, list(q_len.data))
        v = v / (v.norm(p=2, dim=1, keepdim=True).expand_as(v) + 1e-8)

        attention = self.attention(v, q)
        v = apply_attention(v, attention)

        combined = torch.cat([v, q], dim=1)
        answer = self.classifier(combined)

        return answer


class ImageConv(nn.Module):
    def __init__(self):
        super(ImageConv, self).__init__()
        kernel1_size = 3
        kernel2_size = 3
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=kernel1_size)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=kernel2_size)
        # self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=kernel2_size)
        # self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=kernel2_size)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        # x = self.pool(F.relu(self.conv3(x)))
        # x = self.pool(F.relu(self.conv4(x))) # + x_orig
        x = self.dropout(x)
        return x


class inception(nn.Module):
    def __init__(self, num_of_planes, nof1x1, nof3x3_1, nof3x3_out, nof5x5_1, nof5x5_out, pool_planes):
        super(inception, self).__init__()
        # 1x1 conv branch
        self.b1x1 = nn.Sequential(
            nn.Conv2d(num_of_planes, nof1x1, kernel_size=1),
            nn.BatchNorm2d(nof1x1),
            nn.ReLU(True),
        )
        # 1x1 conv -> 3x3 conv branch
        self.b1x3 = nn.Sequential(
            nn.Conv2d(in_channels=num_of_planes, out_channels=nof3x3_1, kernel_size=1),
            nn.BatchNorm2d(nof3x3_1),
            nn.ReLU(True),
            nn.Conv2d(nof3x3_1, nof3x3_out, kernel_size=3, padding=1),
            nn.BatchNorm2d(nof3x3_out),
            nn.ReLU(True),
        )

        # 1x1 conv -> 5x5 conv branch
        self.b1x5 = nn.Sequential(
            nn.Conv2d(num_of_planes, nof5x5_1, kernel_size=1),
            nn.BatchNorm2d(nof5x5_1),
            nn.ReLU(True),
            nn.Conv2d(nof5x5_1, nof5x5_out, kernel_size=5, padding=2),
            nn.BatchNorm2d(nof5x5_out),
            nn.ReLU(True),
        )

        # 3x3 pool -> 1x1 conv branch
        self.b3x1 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(num_of_planes, pool_planes, kernel_size=1),
            nn.BatchNorm2d(pool_planes),
            nn.ReLU(True),
        )

    def forward(self, a):
        b1 = self.b1x1(a)
        b2 = self.b1x3(a)
        b3 = self.b1x5(a)
        b4 = self.b3x1(a)
        return torch.cat([b1, b2, b3, b4], 1)           # concatenating the convolutions' branches


# the convolutional level
class GoogLeNet(nn.Module):
    def __init__(self):
        super(GoogLeNet, self).__init__()
        self.first_layer = nn.Sequential(
            nn.Conv2d(3, 30, kernel_size=3, padding=1),     # using 30 filters 3x3
            nn.BatchNorm2d(30),
            nn.ReLU(True),
        )

        self.a3 = inception(num_of_planes=30,
                            nof1x1=10,
                            nof3x3_1=4,
                            nof3x3_out=12,
                            nof5x5_1=4,
                            nof5x5_out=8,
                            pool_planes=8)
        # input num channels = num_of_planes
        # dim 1 (num channels)  = nof1x1+nof3x3_out+nof5x5_out+pool_planes

        self.b3 = inception(38, 14,  6, 16, 4, 10, 10)

        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.a4 = inception(50, 20,  8, 20, 4, 12, 12)
        self.b4 = inception(64, 22,  9, 24, 4, 14, 16)

        self.a5 = inception(76, 26,  12, 28, 4, 18, 18)
        self.b5 = inception(90, 34,  16, 36, 6, 20, 20)
        self.dropout = nn.Dropout(p=0.3)


    def forward(self, x):
        # x dim = [batch_size, 3, image_size, image_size]
        out = self.first_layer(x)  # [batch_size, 30, image_size, image_size]
        out = self.a3(out)  # [batch_size, 38, image_size, image_size]
        out = self.b3(out)  # [batch_size, 50, image_size, image_size]
        out = self.maxpool(out)  # [batch_size, 50, image_size/2, image_size/2]
        out = self.a4(out)  # [batch_size, 64, image_size/2, image_size/2]
        out = self.b4(out)  # [batch_size, 76, image_size/2, image_size/2]
        out = self.maxpool(out)
        out = self.a5(out)
        out = self.b5(out)
        # out = out+x after conv to change dims - resnet
        out = self.dropout(out)

        return out


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
    #     self._init_lstm(self.lstm.weight_ih_l0)
    #     self._init_lstm(self.lstm.weight_hh_l0)
    #
    #     self.lstm.bias_ih_l0.data.zero_()
    #     self.lstm.bias_hh_l0.data.zero_()
    #
    #     init.xavier_uniform_(self.embedding.weight)
    #
    # def _init_lstm(self, weight):
    #     for w in weight.chunk(4, 0):
    #         init.xavier_uniform_(w)

    def forward(self, q, q_len):
        embedded = self.embedding(q)
        embedded_drop = self.drop(embedded)
        tanhed = self.tanh(embedded_drop)
        packed = pack_padded_sequence(tanhed, q_len, batch_first=True, enforce_sorted=False) # TODO understand padding and packed object
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


def apply_attention(input_, attention):
    """ Apply any number of attention maps over the input. """
    n, c = input_.size()[:2]
    glimpses = attention.size(1)

    # flatten the spatial dims into the third dim, since we don't need to care about how they are arranged
    input_ = input_.view(n, 1, c, -1) # [n, 1, c, s]
    attention = attention.view(n, glimpses, -1)
    attention = F.softmax(attention, dim=-1).unsqueeze(2) # [n, g, 1, s]
    weighted = attention * input_ # [n, g, v, s]
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


