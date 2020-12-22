import os

import h5py
from PIL import Image
import torchvision.transforms as transforms


def _get_transformations(target_size, central_fraction=1.0):
    return transforms.Compose([
        transforms.Resize(size=int(target_size / central_fraction)),
        transforms.CenterCrop(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def preprocess_images(other_paths, data_paths, image_size, central_fraction, processed_path):


    base_path = other_paths['base_path']
    image_path = os.path.join(base_path, data_paths['imgs'])
    num_imgs = 0
    file_names = []
    for filename in os.listdir(image_path):
        if not filename.endswith('.jpg'):
            print(f"{filename} is not jpg, skipping")
            continue
        num_imgs += 1
        file_names.append(filename)

    # save to file
    features_shape = (
        num_imgs,
        3,
        image_size,
        image_size)
    #
    with h5py.File(processed_path, libver='latest') as fd:
        features = fd.create_dataset('features', shape=features_shape, dtype='float16')
        coco_ids = fd.create_dataset('ids', shape=(num_imgs,), dtype='int32')
        id_to_features = {}
        transformations = _get_transformations(image_size, central_fraction)

        for indx, filename in enumerate(file_names):
            id_and_extension = filename.split('_')[-1]
            id = int(id_and_extension.split('.')[0])
            path = os.path.join(image_path, filename)
            img = Image.open(path).convert('RGB')
            transformed_img = transformations(img)
            features[indx, :, :, :] = transformed_img.numpy().astype('float16')
            coco_ids[indx] = id
