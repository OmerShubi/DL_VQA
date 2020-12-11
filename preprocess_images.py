import os

import h5py
from torch.autograd import Variable
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.models as models
from tqdm import tqdm

from main_preprocess import init_coco_loader
from resnet import resnet as caffe_resnet


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = caffe_resnet.resnet152(pretrained=True)

        def save_output(module, input, output):
            self.buffer = output
        self.model.layer4.register_forward_hook(save_output)

    def forward(self, x):
        self.model(x)
        return self.buffer




def main():

    base_path = '/datashare'
    train_path = os.path.join(base_path, 'train2014')  # directory of training images
    val_path = os.path.join(base_path, 'val2014')  # directory of validation images


    cudnn.benchmark = True

    net = Net().cuda()
    net.eval()
    image_size = 320  # scale shorter end of image to this size and centre crop
    output_size = image_size // 32  # size of the feature maps after processing through a network
    output_features = 2048  # number of feature maps thereof

    loader = init_coco_loader(train_path, val_path)
    features_shape = (
        len(loader.dataset),
        output_features,
        output_size,
        output_size
    )
    preprocessed_path = './resnet-14x14.h5'  # path where preprocessed features are saved to and loaded from

    with h5py.File(preprocessed_path, libver='latest') as fd:
        features = fd.create_dataset('features', shape=features_shape, dtype='float16')
        coco_ids = fd.create_dataset('ids', shape=(len(loader.dataset),), dtype='int32')

        i = j = 0
        with torch.no_grad():
            for ids, imgs in tqdm(loader):
                imgs = Variable(imgs.cuda(non_blocking=True))
                out = net(imgs)

                j = i + imgs.size(0)
                features[i:j, :, :] = out.data.cpu().numpy().astype('float16')
                coco_ids[i:j] = ids.numpy().astype('int32')
                i = j


if __name__ == '__main__':
    main()
