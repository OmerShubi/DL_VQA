import os

import h5py
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.models as models
from tqdm import tqdm
from PIL import Image

from preprocessing import data_preprocessing
import torchvision.transforms as transforms


def get_transformations(target_size, central_fraction=1.0):
    return transforms.Compose([
        transforms.Resize(size=int(target_size / central_fraction)),
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
    datasets = [CocoImages(path, transform=transformations) for path in paths]
    img_dataset = Composite(*datasets)

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
        self.model = models.resnet152(pretrained=True)

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



class CocoImages(torch.utils.data.Dataset):
    def __init__(self, path, transform=None):
        """
        Dataset for MSCOCO images located in a folder on the filesystem

        :param path: path to image dir
        :param transform: transforms.Compose, transformations to apply
        """
        super(CocoImages, self).__init__()
        self.path = path
        self.id_to_filename = self._find_images()
        self.sorted_ids = sorted(self.id_to_filename.keys())  # used for deterministic iteration order
        print('found {} images in {}'.format(len(self), self.path))
        self.transform = transform

    def _find_images(self):
        id_to_filename = {}
        for filename in os.listdir(self.path):
            if not filename.endswith('.jpg'):
                continue
            id_and_extension = filename.split('_')[-1]
            id = int(id_and_extension.split('.')[0])
            id_to_filename[id] = filename
        return id_to_filename

    def __getitem__(self, item):
        id = self.sorted_ids[item]
        path = os.path.join(self.path, self.id_to_filename[id])
        img = Image.open(path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        return id, img

    def __len__(self):
        return len(self.sorted_ids)


class Composite(torch.utils.data.Dataset):
    """ Dataset that is a composite of several Dataset objects. Useful for combining splits of a dataset. """

    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, item):
        current = self.datasets[0]
        for d in self.datasets:
            if item < len(d):
                return d[item]
            item -= len(d)
        else:
            raise IndexError('Index too large for composite dataset')

    def __len__(self):
        return sum(map(len, self.datasets))



if __name__ == '__main__':
    create_processed_images('/datashare','train2014','val2014','./resnet18_13.h5')
