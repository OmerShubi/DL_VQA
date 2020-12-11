"""
Main file
We will run the whole program from here
"""

import torch
import hydra

from preprocessing import preprocess_images, preprocess_vocab
from preprocessing.data_preprocessing import VQA_dataset
from train import train
from models.base_model import MyModel
from torch.utils.data import DataLoader
from utils import main_utils, train_utils
from utils.main_utils import collate_fn
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
    main_utils.set_seed(cfg['main']['seed'])


    images_to_h5 = False
    if images_to_h5:
        # todo replace resnet?
        preprocess_images.main()

    q_and_a_to_vocab = False
    if q_and_a_to_vocab:
        preprocess_vocab.main()

    # TODO check if there is difference between v1 and v2, if yes check parsing of answers also
    # Load dataset
    train_dataset = VQA_dataset(data_paths=cfg['main']['train_paths'],
                                other_paths=cfg['main']['paths'],
                                answerable_only=True)  # TODO answerable_only?

    val_dataset = VQA_dataset(data_paths=cfg['main']['val_paths'],
                              other_paths=cfg['main']['paths'],
                              answerable_only=False)

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=cfg['train']['batch_size'],
                              shuffle=True,
                              num_workers=cfg['main']['num_workers'],
                              pin_memory=True,
                              collate_fn=collate_fn)  # TODO collate_fn??

    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=cfg['train']['batch_size'],
                            shuffle=False,
                            num_workers=cfg['main']['num_workers'],
                            pin_memory=True,
                            collate_fn=collate_fn)

    # Init model
    model = MyModel(num_hid=cfg['train']['num_hid'], dropout=cfg['train']['dropout'])

    # TODO: Add gpus_to_use
    if cfg['main']['parallel']:
        model = torch.nn.DataParallel(model)

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
