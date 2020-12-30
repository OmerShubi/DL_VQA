"""
Includes all utils related to training
"""

import torch

from typing import Dict
from torch import Tensor
from omegaconf import DictConfig


def batch_accuracy(predicted, true):
    """ Compute the accuracies for a batch of predictions and answers """
    #  predicted_index is tensor of batch X index of highest predicted value
    _, predicted_index = predicted.max(dim=1, keepdim=True)
    # agreeing is tensor of number of people that said each predicted_index answer
    indices, values, size = true
    indices_nonzero = indices.nonzero().t()
    indices_for_slicing = indices_nonzero.cpu().numpy()
    relevant_values = values[indices_for_slicing]
    indices_nonzero[1] = indices[indices_for_slicing]-1
    true_sparse = torch.sparse_coo_tensor(indices_nonzero.cuda(), relevant_values, size)
    agreeing = torch.tensor([true_sparse[batch_index, index.item()] for batch_index, index in enumerate(predicted_index)])

    return torch.sum((agreeing * 0.3).clamp(max=1))


def compute_score_with_logits(logits: Tensor, labels: Tensor) -> Tensor:
    """
    Calculate multiclass accuracy with logits (one class also works)
    :param logits: tensor with logits from the model
    :param labels: tensor holds all the labels
    :return: score for each sample
    """
    logits = torch.max(logits, 1)[1].data  # argmax

    logits_one_hots = torch.zeros(*labels.size())
    if torch.cuda.is_available():
        logits_one_hots = logits_one_hots.cuda()
    logits_one_hots.scatter_(1, logits.view(-1, 1), 1)

    scores = (logits_one_hots * labels)

    return scores


def get_zeroed_metrics_dict() -> Dict:
    """
    :return: dictionary to store all relevant metrics for training
    """
    return {'train_loss': 0, 'train_score': 0, 'total_norm': 0, 'count_norm': 0}


class TrainParams:
    """
    This class holds all train parameters.
    Add here variable in case configuration file is modified.
    """
    n_epochs_stop: int
    num_epochs: int
    lr: float
    lr_decay: float
    lr_gamma: float
    lr_step_size: int
    save_model: bool
    max_answers: int

    def __init__(self, **kwargs):
        """
        :param kwargs: configuration file
        """
        self.n_epochs_stop = kwargs['n_epochs_stop']
        self.num_epochs = kwargs['num_epochs']

        self.lr = kwargs['lr']['lr_value']
        self.lr_decay = kwargs['lr']['lr_decay']
        self.lr_gamma = kwargs['lr']['lr_gamma']
        self.lr_step_size = kwargs['lr']['lr_step_size']

        self.save_model = kwargs['save_model']
        self.max_answers = kwargs['max_answers']


def get_train_params(cfg: DictConfig) -> TrainParams:
    """
    Return a TrainParams instance for a given configuration file
    :param cfg: configuration file
    :return:
    """
    return TrainParams(**cfg['train'])
