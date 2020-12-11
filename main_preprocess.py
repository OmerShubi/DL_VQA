import os

import torch
import data_preprocessing
from utils import get_transformations
import preprocess_images
import preprocess_vocab


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


if __name__ == '__main__':
    base_path = '/datashare'  # TODO param
    train_path = os.path.join(base_path, 'train2014')  # directory of training images  # TODO param
    val_path = os.path.join(base_path, 'val2014')  # directory of validation images  # TODO param

    img_loader = init_coco_loader(train_path, val_path)

    images_to_h5 = False
    if images_to_h5:
        # todo replace resnet?
        preprocess_images.main()

    q_and_a_to_vocab = False
    if q_and_a_to_vocab:
        preprocess_vocab.main()

    # TODO check if there is difference between v1 and v2, if yes check parsing of answers also
    train_loader = data_preprocessing.get_loader(train=True)
    valid_loader = data_preprocessing.get_loader(val=True)
