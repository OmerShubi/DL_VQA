"""
Main file
We will run the whole program from here
"""
import os

import torch
import hydra

from preprocessing import preprocess_vocab
from preprocessing.data_preprocessing import VQA_dataset
from preprocessing.preprocess_images import preprocess_images
from train import train
from models.base_model import Net
from torch.utils.data import DataLoader
from utils import main_utils, train_utils
from utils.train_logger import TrainLogger
from omegaconf import DictConfig, OmegaConf

torch.backends.cudnn.benchmark = True


@hydra.main(config_path="config", config_name='config')
def main(cfg: DictConfig) -> None:
    """
    Run the code following a given configuration
    :param cfg: configuration file retrieved from hydra framework
    """
    main_utils.init(cfg)
    logger = TrainLogger(exp_name_prefix=cfg['main']['experiment_name_prefix'], logs_dir=cfg['main']['paths']['logs'])
    logger.write(OmegaConf.to_yaml(cfg))

    # Set seed for results reproduction
    main_utils.set_seed(cfg['main']['seed'])  # TODO make sure

    # todo replace resnet - https://github.com/shilrley6/Faster-R-CNN-with-model-pretrained-on-Visual-Genome

    vocab_path = cfg['main']['paths']['vocab_path']
    if not os.path.exists(vocab_path):
        logger.write("Creating Vocab")
        preprocess_vocab.create_vocab(data_base_path=cfg['main']['paths']['base_path'],
                                      data_paths=cfg['main']['train_paths'],
                                      vocab_path=vocab_path,
                                      max_answers=cfg['train']['max_answers'])

    train_imgs = cfg['main']['train_paths']['processed_imgs']
    if not os.path.exists(train_imgs):
        logger.write("Processing train images")
        preprocess_images(other_paths=cfg['main']['paths'],
                          data_paths=cfg['main']['train_paths'],
                          image_size=cfg['train']['image_size'],
                          central_fraction=cfg['train']['central_fraction'],
                          processed_path=train_imgs)

    val_imgs = cfg['main']['val_paths']['processed_imgs']
    if not os.path.exists(val_imgs):
        logger.write("Processing validation images")
        preprocess_images(other_paths=cfg['main']['paths'],
                          data_paths=cfg['main']['val_paths'],
                          image_size=cfg['train']['image_size'],
                          central_fraction=cfg['train']['central_fraction'],
                          processed_path=val_imgs)

    # Load dataset
    logger.write("Creating train dataset")
    train_dataset = VQA_dataset(data_paths=cfg['main']['train_paths'],
                                other_paths=cfg['main']['paths'],
                                logger=logger,
                                answerable_only=True)

    logger.write("Creating validation dataset")
    val_dataset = VQA_dataset(data_paths=cfg['main']['val_paths'],
                              other_paths=cfg['main']['paths'],
                              logger=logger,
                              answerable_only=False)

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=cfg['train']['batch_size'],
                              shuffle=True,
                              num_workers=cfg['main']['num_workers'],
                              pin_memory=True)

    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=cfg['train']['batch_size'],
                            shuffle=False,
                            num_workers=cfg['main']['num_workers'],
                            pin_memory=True)

    # Init model
    model = Net(cfg['train'], embedding_tokens=train_loader.dataset.num_tokens)

    if torch.cuda.is_available():
        model = model.cuda()

    logger.write(main_utils.get_model_string(model))

    # Run model
    train_params = train_utils.get_train_params(cfg)

    # Report metrics and hyper parameters to tensorboard
    metrics = train(model, train_loader, val_loader, train_params, logger)
    hyper_parameters = main_utils.get_flatten_dict(cfg['train'])

    logger.report_metrics_hyper_params(hyper_parameters, metrics)


if __name__ == '__main__':
    main()
