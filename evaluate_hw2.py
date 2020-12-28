import os
import pickle

import torch
import hydra

from preprocessing import preprocess_vocab
from preprocessing.data_preprocessing import VQA_dataset
from preprocessing.preprocess_images import preprocess_images
from train import train, get_metrics, evaluate
from models.model import VqaNet
from torch.utils.data import DataLoader
from utils import main_utils, train_utils
from utils.train_logger import TrainLogger
from omegaconf import DictConfig, OmegaConf

torch.backends.cudnn.benchmark = True
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


@hydra.main(config_path="config", config_name='config')
def evaluate_hw2(cfg: DictConfig):
    """
        Run the code following a given configuration
        :param cfg: configuration file retrieved from hydra framework
    """

    main_utils.init(cfg)
    logger = TrainLogger(exp_name_prefix=cfg['main']['experiment_name_prefix'], logs_dir=cfg['main']['full']['paths']['logs'])

    # Set seed for results reproduction
    main_utils.set_seed(cfg['main']['seed'])

    vocab_path = cfg['main']['full']['paths']['vocab_path']
    if not os.path.exists(vocab_path):
        preprocess_vocab.create_vocab(data_base_path=cfg['main']['full']['paths']['base_path'],
                                      data_paths=cfg['main']['full']['train_paths'],
                                      vocab_path=vocab_path,
                                      max_answers=cfg['train']['max_answers'])

    val_imgs = cfg['main']['full']['val_paths']['processed_imgs']
    if not os.path.exists(val_imgs):
        preprocess_images(other_paths=cfg['main']['full']['paths'],
                          data_paths=cfg['main']['full']['val_paths'],
                          image_size=cfg['train']['image_size'],
                          central_fraction=cfg['train']['central_fraction'],
                          processed_path=val_imgs)

    # Load dataset
    vqa_path_val = cfg['main']['full']['val_paths']['vqaDataset']
    if os.path.exists(vqa_path_val):
        val_dataset = pickle.load(open(vqa_path_val, 'rb'))
    else:
        val_dataset = VQA_dataset(data_paths=cfg['main']['full']['val_paths'],
                                  other_paths=cfg['main']['full']['paths'],
                                  logger=logger,
                                  answerable_only=False)
        pickle.dump(val_dataset, open(vqa_path_val, 'wb'))

    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=cfg['train']['batch_size'],
                            shuffle=False,
                            num_workers=cfg['main']['num_workers'],
                            pin_memory=True)

    # Init model
    model = VqaNet(cfg['train'], embedding_tokens=val_loader.dataset.num_tokens)
    model_load_path = cfg['main']['full']['paths']['pretrained_model_path']
    model_stuff = torch.load(model_load_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(model_stuff['model_state'])

    if torch.cuda.is_available():
        model = model.cuda()

    # Run model
    score, _ = evaluate(model, val_loader, cfg['train']['max_answers'])

    return score


if __name__ == '__main__':
    score = evaluate_hw2()
    print(score)