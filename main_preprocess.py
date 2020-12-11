import os

import torch
import torchvision.transforms as transforms
import data_preprocessing


def get_transform(target_size, central_fraction=1.0):
    return transforms.Compose([
        transforms.Scale(int(target_size / central_fraction)),
        transforms.CenterCrop(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def init_coco_loader(*paths):
    # preprocess config
    preprocess_batch_size = 64
    image_size = 320  # scale shorter end of image to this size and centre crop
    central_fraction = 0.875  # only take this much of the centre when scaling and centre cropping
    data_workers = 8

    transform = get_transform(image_size, central_fraction)
    datasets = [data_preprocessing.CocoImages(path, transform=transform) for path in paths]
    dataset = data_preprocessing.Composite(*datasets)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=preprocess_batch_size,
        num_workers=data_workers,
        shuffle=False,
        pin_memory=True,
    )
    return data_loader


if __name__ == '__main__':
    base_path = '/datashare'
    train_path = os.path.join(base_path, 'train2014')  # directory of training images
    val_path = os.path.join(base_path, 'val2014')  # directory of validation images

    img_loader = init_coco_loader(train_path, val_path)  # TODO is it okay both train and val?
    train_loader = data_preprocessing.get_loader(train=True)
    valid_loader = data_preprocessing.get_loader(val=True)
