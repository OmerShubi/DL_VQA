"""
Here, we will run everything that is related to the training procedure.
"""

import time
import torch
import torch.nn as nn
import numpy as np

from tqdm import tqdm
from utils import train_utils
from torch.utils.data import DataLoader
from utils.types import Scores, Metrics
from utils.train_utils import TrainParams, batch_accuracy
from utils.train_logger import TrainLogger


def get_metrics(best_eval_score: float, eval_score: float, train_loss: float) -> Metrics:
    """
    Example of metrics dictionary to be reported to tensorboard. Change it to your metrics
    :param best_eval_score: best_eval_score
    :param eval_score: eval_score
    :param train_loss: train_loss
    :return:
    """
    return {'Metrics/BestAccuracy': best_eval_score,
            'Metrics/LastAccuracy': eval_score,
            'Metrics/LastLoss': train_loss}


def update_learning_rate(optimizer, iteration, initial_lr):
    lr_halflife = 50000
    lr = initial_lr * 0.5 ** (float(iteration) / lr_halflife)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(model: nn.Module, train_loader: DataLoader, eval_loader: DataLoader, train_params: TrainParams,
          logger: TrainLogger, optimizer_stuff: dict) -> Metrics:
    """
    Training procedure

    :param model:
    :param train_loader:
    :param eval_loader:
    :param train_params:
    :param logger:
    :return:
    """
    total_iterations = 0
    metrics = train_utils.get_zeroed_metrics_dict()
    best_eval_score = torch.tensor(0.0)
    epochs_no_improve = 0
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=train_params.lr)
    if optimizer_stuff:
        optimizer.load_state_dict(optimizer_stuff)
    # Create learning rate scheduler
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
    #                                             step_size=train_params.lr_step_size,
    #                                             gamma=train_params.lr_gamma)

    log_softmax = nn.LogSoftmax(dim=1).cuda()

    for epoch in range(train_params.num_epochs):
        t = time.time()
        metrics = train_utils.get_zeroed_metrics_dict()

        for batch_data in tqdm(train_loader):
            batch_loss, batch_score = run_batch(model,
                                                log_softmax,
                                                batch_data,
                                                train_params.max_answers)

            # Optimization step
            optimizer.zero_grad()
            # Learning rate scheduler step
            update_learning_rate(optimizer=optimizer, iteration=total_iterations, initial_lr=train_params.lr)
            batch_loss.backward()
            optimizer.step()
            total_iterations += 1

            # Calculate metrics
            # metrics['total_norm'] += nn.utils.clip_grad_norm_(model.parameters(), train_params.grad_clip)
            # metrics['count_norm'] += 1

            metrics['train_score'] += batch_score

            metrics['train_loss'] += batch_loss

            # Report model to tensorboard
            # if epoch == 0 and i == 0:
            # logger.report_graph(model, [v, q, q_len])

        # scheduler.step()

        # Calculate metrics
        metrics['train_loss'] /= len(train_loader)

        metrics['train_score'] /= len(train_loader.dataset)
        metrics['train_score'] *= 100

        model.train(False)
        metrics['eval_score'], metrics['eval_loss'] = evaluate(model, eval_loader, train_params.max_answers)
        model.train(True)

        epoch_time = time.time() - t
        logger.write_epoch_statistics(epoch=epoch,
                                      epoch_time=epoch_time,
                                      train_loss=metrics['train_loss'],
                                      norm=0,
                                      train_score=metrics['train_score'],
                                      eval_score=metrics['eval_score'])

        scalars = {'Accuracy/Train': metrics['train_score'],
                   'Accuracy/Validation': metrics['eval_score'],
                   'Loss/Train': metrics['train_loss'],
                   'Loss/Validation': metrics['eval_loss']}

        logger.report_scalars(scalars, epoch)

        scalars_2perplot = {'Accuracy': {'Train': metrics['train_score'],
                                         'Validation': metrics['eval_score']},
                            'Loss': {'Train': metrics['train_loss'],
                                     'Validation': metrics['eval_loss']}}

        logger.report_scalars_same_plot(scalars_2perplot, epoch)

        if metrics['eval_score'] > best_eval_score:
            epochs_no_improve = 0
            best_eval_score = metrics['eval_score']
            if train_params.save_model:
                logger.save_model(model, epoch, optimizer)
        else:
            epochs_no_improve += 1

        if epoch > 3 and epochs_no_improve == train_params.n_epochs_stop:
            logger.write('Early stopping!')
            return get_metrics(best_eval_score, metrics['eval_score'], metrics['train_loss'])

    return get_metrics(best_eval_score, metrics['eval_score'], metrics['train_loss'])


@torch.no_grad()
def evaluate(model: nn.Module, dataloader: DataLoader, max_answers):
    """
    Evaluate a model without gradient calculation

    :param model: instance of a model
    :param dataloader: dataloader to evaluate the model on
    :return: tuple of (accuracy, loss) values
    """
    score = torch.tensor(0.0)
    loss = 0

    log_softmax = nn.LogSoftmax(dim=1).cuda()

    for batch_data in tqdm(dataloader):
        batch_loss, batch_score = run_batch(model,
                                            log_softmax,
                                            batch_data,
                                            max_answers)
        loss += batch_loss
        score += batch_score
    loss /= len(dataloader)
    score /= len(dataloader.dataset)
    score *= 100

    return score, loss


def run_batch(model, log_softmax, batch_data, max_answers):
    """
    Run a single batch through the model
    :param model:
    :param log_softmax:
    :param batch_data:
    :param max_answers:
    :param type_:
    :return:
    """
    v, q, a_indices, a_values, a_length, idx, q_len = batch_data
    if torch.cuda.is_available():
        v = v.cuda()
        q = q.cuda()
        a_values = a_values.cuda()
        q_len = q_len.cuda()

    y_hat = model(v, q, q_len)
    nll = -log_softmax(y_hat)
    # (nll * a / 10) is loss in entries of correct answers, multiplied by proportion of # correct out of 10
    # Sum is over all correct answers
    # mean is over batch
    batch_size = y_hat.shape[0]
    batch_indices = np.repeat(range(batch_size), a_length)

    a_indices_flat = a_indices.flatten()
    # remove indices that where part of padding, and adjust indices to match nll (start from 0 instead of 1)
    a_indices_flat_nonzero_adjusted = a_indices_flat[a_indices_flat.nonzero()].flatten().cpu().numpy() - 1
    nll_relevant = nll[[batch_indices, a_indices_flat_nonzero_adjusted]]
    a_values_flat = a_values.flatten()

    # remove values that where part of padding (0) and divide by 10 to get probabilities
    a_values_flat_nonzero = a_values_flat[a_values_flat.nonzero()].flatten() / 10.0

    batch_loss = (nll_relevant * a_values_flat_nonzero).sum() / batch_size
    batch_score = batch_accuracy(y_hat.data, (a_indices, a_values, (batch_size, max_answers)))
    return batch_loss, batch_score
