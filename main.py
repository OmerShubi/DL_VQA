"""
Main file
We will run the whole program from here
"""
import os
import pickle

import torch
import hydra

from preprocessing import preprocess_vocab
from preprocessing.data_preprocessing import VQA_dataset
from preprocessing.preprocess_images import preprocess_images
from train import train, get_metrics
from models.model import VqaNet
from torch.utils.data import DataLoader
from utils import main_utils, train_utils
from utils.train_logger import TrainLogger
from omegaconf import DictConfig, OmegaConf

torch.backends.cudnn.benchmark = True
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


@hydra.main(config_path="config", config_name='config')
def main(cfg: DictConfig) -> float:
    """
    Run the code following a given configuration
    :param cfg: configuration file retrieved from hydra framework
    """

    main_utils.init(cfg)

    if cfg['main']['use_full']:
        full_flag = 'full'
    else:
        full_flag = 'small'

    logger = TrainLogger(exp_name_prefix=cfg['main']['experiment_name_prefix'], logs_dir=cfg['main'][full_flag]['paths']['logs'])
    logger.write(f"Num gpus: {torch.cuda.device_count()}")  # print 1
    logger.write(f"gpu ID:{os.environ['CUDA_VISIBLE_DEVICES']}")  # print 0
    logger.write(OmegaConf.to_yaml(cfg))

    # Set seed for results reproduction
    main_utils.set_seed(cfg['main']['seed'])

    # Load or create Train Vocab
    vocab_path = cfg['main'][full_flag]['paths']['vocab_path']
    if not os.path.exists(vocab_path):
        logger.write("Creating Vocab")
        preprocess_vocab.create_vocab(data_base_path=cfg['main'][full_flag]['paths']['base_path'],
                                      data_paths=cfg['main'][full_flag]['train_paths'],
                                      vocab_path=vocab_path,
                                      max_answers=cfg['train']['max_answers'])

    # Load or create preprocessed train images
    train_imgs = cfg['main'][full_flag]['train_paths']['processed_imgs']
    if not os.path.exists(train_imgs):
        logger.write(f"Processing train images, saving at {train_imgs}")
        preprocess_images(other_paths=cfg['main'][full_flag]['paths'],
                          data_paths=cfg['main'][full_flag]['train_paths'],
                          image_size=cfg['train']['image_size'],
                          central_fraction=cfg['train']['central_fraction'],
                          processed_path=train_imgs)

    # Load or create preprocessed validation images
    val_imgs = cfg['main'][full_flag]['val_paths']['processed_imgs']
    if not os.path.exists(val_imgs):
        logger.write(f"Processing validation images, saving at {val_imgs}")
        preprocess_images(other_paths=cfg['main'][full_flag]['paths'],
                          data_paths=cfg['main'][full_flag]['val_paths'],
                          image_size=cfg['train']['image_size'],
                          central_fraction=cfg['train']['central_fraction'],
                          processed_path=val_imgs)

    # Load train dataset
    vqa_path_train = cfg['main'][full_flag]['train_paths']['vqaDataset']
    if os.path.exists(vqa_path_train):
        logger.write(f"Loading VQA train dataset from {vqa_path_train}")
        train_dataset = pickle.load(open(vqa_path_train, 'rb'))
    else:
        logger.write("Creating train dataset")
        train_dataset = VQA_dataset(data_paths=cfg['main'][full_flag]['train_paths'],
                                    other_paths=cfg['main'][full_flag]['paths'],
                                    logger=logger,
                                    answerable_only=True)
        pickle.dump(train_dataset, open(vqa_path_train, 'wb'))

    # Load test dataset
    vqa_path_val = cfg['main'][full_flag]['val_paths']['vqaDataset']
    if os.path.exists(vqa_path_val):
        logger.write(f"Loading VQA val dataset from {vqa_path_val}")
        val_dataset = pickle.load(open(vqa_path_val, 'rb'))
    else:
        logger.write("Creating val dataset")
        val_dataset = VQA_dataset(data_paths=cfg['main'][full_flag]['val_paths'],
                                  other_paths=cfg['main'][full_flag]['paths'],
                                  logger=logger,
                                  answerable_only=False)
        pickle.dump(val_dataset, open(vqa_path_val, 'wb'))

    # Init model
    model = VqaNet(cfg['train'], embedding_tokens=train_dataset.num_tokens)
    optimizer_stuff = None

    # Start from pretrained if exist
    if cfg['main']['start_from_pretrained_model']:
        model_load_path = cfg['main'][full_flag]['paths']['pretrained_model_path']
        model_stuff = torch.load(model_load_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(model_stuff['model_state'])
        optimizer_stuff = model_stuff['optimizer_state']
        logger.write(f"Loaded model and optimizer, epoch: {model_stuff['epoch']}")

    if torch.cuda.is_available():
        model = model.cuda()

    model_string, _ = main_utils.get_model_string(model)

    logger.write(model_string)

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

    train_params = train_utils.get_train_params(cfg)

    # Run model
    metrics = train(model, train_loader, val_loader, train_params, logger, optimizer_stuff)

    # Report metrics and hyper parameters to tensorboard
    hyper_parameters = main_utils.get_flatten_dict(cfg['train'])

    logger.report_metrics_hyper_params(hyper_parameters, metrics)
    if isinstance(metrics['Metrics/BestAccuracy'], float):
        return metrics['Metrics/BestAccuracy']
    else:
        return metrics['Metrics/BestAccuracy'].item()


if __name__ == '__main__':
    main()
