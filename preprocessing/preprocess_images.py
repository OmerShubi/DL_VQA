import os

import h5py
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.models as models
from tqdm import tqdm

from preprocessing import data_preprocessing
from resnet import resnet as caffe_resnet
import torchvision.transforms as transforms


def get_transformations(target_size, central_fraction=1.0):
    return transforms.Compose([
        transforms.Scale(int(target_size / central_fraction)),
        transforms.CenterCrop(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

def init_coco_loader(*paths):
    """

    :param paths: iterable, paths to raw images directories
    :return: CocoImages combined dataloader
    """
    # preprocess config
    preprocess_batch_size = 64  # TODO param
    image_size = 320  # scale shorter end of image to this size and centre crop  # TODO param
    central_fraction = 0.875  # only take this much of the centre when scaling and centre cropping  # TODO param
    data_workers = 8  # TODO param

    transformations = get_transformations(image_size, central_fraction)
    datasets = [data_preprocessing.CocoImages(path, transform=transformations) for path in paths]
    img_dataset = data_preprocessing.Composite(*datasets)

    img_data_loader = torch.utils.data.DataLoader(
        img_dataset,
        batch_size=preprocess_batch_size,
        num_workers=data_workers,
        shuffle=False,
        pin_memory=True,
    )
    return img_data_loader

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


def create_processed_images(data_base_path, train_imgs_path, val_imgs_path, save_path):

    train_path = os.path.join(data_base_path, train_imgs_path)  # directory of training images
    val_path = os.path.join(data_base_path, val_imgs_path)  # directory of validation images


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
    preprocessed_path = save_path  # path where preprocessed features are saved to and loaded from

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
    create_processed_images()
