"""
Here, we will run everything that is related to the training procedure.
"""

import time
import torch
import torch.nn as nn

from tqdm import tqdm
from utils import train_utils
from torch.utils.data import DataLoader
from utils.types import Scores, Metrics
from utils.train_utils import TrainParams,batch_accuracy
from utils.train_logger import TrainLogger


def get_metrics(best_eval_score: float, eval_score: float, train_loss: float) -> Metrics:
    """
    Example of metrics dictionary to be reported to tensorboard. Change it to your metrics
    :param best_eval_score:
    :param eval_score:
    :param train_loss:
    :return:
    """
    return {'Metrics/BestAccuracy': best_eval_score,
            'Metrics/LastAccuracy': eval_score,
            'Metrics/LastLoss': train_loss}


def train(model: nn.Module, train_loader: DataLoader, eval_loader: DataLoader, train_params: TrainParams,
          logger: TrainLogger) -> Metrics:
    """
    Training procedure. Change each part if needed (optimizer, loss, etc.)
    :param model:
    :param train_loader:
    :param eval_loader:
    :param train_params:
    :param logger:
    :return:
    """
    metrics = train_utils.get_zeroed_metrics_dict()
    best_eval_score = 0

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=train_params.lr)

    # Create learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=train_params.lr_step_size,
                                                gamma=train_params.lr_gamma)

    log_softmax = nn.LogSoftmax(dim=1).cuda() # TODO change

    for epoch in range(train_params.num_epochs):
        t = time.time()
        metrics = train_utils.get_zeroed_metrics_dict()

        for (v, q, a_indices, a_values, a_length, idx, q_len) in tqdm(train_loader):
            if torch.cuda.is_available():
                v = v.cuda()
                q = q.cuda()
                a_values = a_values.cuda()
                q_len = q_len.cuda()
                # todo make sure requires_grad is correct

            y_hat = model(v, q, q_len)
            nll = -log_softmax(y_hat)
            # (nll * a / 10) is loss in entries of correct answers, multiplied by proportion of # correct out of 10
            # Sum is over all correct answers
            # mean is over batch
            batch_size = y_hat.shape[0]
            batch_indices = [] # TODO cleaner way
            for i in range(batch_size):
                batch_indices.extend([i] * 10)
                # a_indices = a.coalesce().indices().cpu().numpy()
            # a_values = a.coalesce().values()
            loss = (nll[[batch_indices, a_indices.flatten().cpu().numpy()]] * (a_values.flatten() / 10.0)).sum() / batch_size
            # loss = (nll * a.to_dense() / 10).sum(dim=1).mean()
            # nll[a]
            # Optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calculate metrics
            metrics['total_norm'] += nn.utils.clip_grad_norm_(model.parameters(), train_params.grad_clip)
            metrics['count_norm'] += 1

            batch_score = batch_accuracy(y_hat.data, (a_indices, a_values, (batch_size, train_params.max_answers)))  # TODO make sure calculation is correct according to Itai.
            metrics['train_score'] += torch.sum(batch_score).cpu().item()

            metrics['train_loss'] += loss.item()  #* x.size(0)

            # Report model to tensorboard
            # if epoch == 0 and i == 0:
            # logger.report_graph(model, [v, q, q_len])

        # Learning rate scheduler step
        scheduler.step()

        # Calculate metrics
        metrics['train_loss'] /= len(train_loader.dataset)

        metrics['train_score'] /= len(train_loader.dataset)
        metrics['train_score'] *= 100

        norm = metrics['total_norm'] / metrics['count_norm']

        model.train(False)
        metrics['eval_score'], metrics['eval_loss'] = evaluate(model, eval_loader, train_params.max_answers)
        model.train(True)

        epoch_time = time.time() - t
        logger.write_epoch_statistics(epoch, epoch_time, metrics['train_loss'], norm,
                                      metrics['train_score'], metrics['eval_score'])

        scalars = {'Accuracy/Train': metrics['train_score'],
                   'Accuracy/Validation': metrics['train_loss'],
                   'Loss/Train': metrics['eval_score'],
                   'Loss/Validation': metrics['eval_loss']}

        logger.report_scalars(scalars, epoch)

        if metrics['eval_score'] > best_eval_score:
            best_eval_score = metrics['eval_score']
            if train_params.save_model:
                logger.save_model(model, epoch, optimizer)

    return get_metrics(best_eval_score, metrics['eval_score'], metrics['train_loss'])


@torch.no_grad()
def evaluate(model: nn.Module, dataloader: DataLoader, max_answers) -> Scores:
    """
    Evaluate a model without gradient calculation
    :param model: instance of a model
    :param dataloader: dataloader to evaluate the model on
    :return: tuple of (accuracy, loss) values
    """
    score = 0
    loss = 0

    log_softmax = nn.LogSoftmax(dim=1).cuda()

    for (v, q, a_indices, a_values, a_length, idx, q_len) in tqdm(dataloader):
        if torch.cuda.is_available():
            v = v.cuda()
            q = q.cuda()
            a_values = a_values.cuda()
            q_len = q_len.cuda()

        y_hat = model(v, q, q_len)

        nll = -log_softmax(y_hat)
        # a_indices = a.coalesce().indices().cpu().numpy()
        # a_values = a.coalesce().values()
        batch_size = y_hat.shape[0]
        batch_indices = []  # TODO cleaner way
        for i in range(batch_size):
            batch_indices.extend([i] * 10)
            # a_indices = a.coalesce().indices().cpu().numpy()
        # a_values = a.coalesce().values()

        loss = (nll[a_indices.flatten().cpu().numpy()] * (a_values.flatten() / 10.0)).sum() / batch_size
        # loss += (nll * a.to_dense() / 10).sum(dim=1).mean()

        score += torch.sum(batch_accuracy(y_hat.data, (a_indices, a_values, (batch_size, max_answers))).cpu())

    loss /= len(dataloader.dataset)
    score /= len(dataloader.dataset)
    score *= 100

    return score, loss

# TODO merge duplicate code from train and evaluate
# def run_batch(model, v, q, a_indices, a_values, a_length, idx, q_len):
#     if torch.cuda.is_available():
#         v = v.cuda()
#         q = q.cuda()
#         a_values = a_values.cuda()
#         q_len = q_len.cuda()
#
#     y_hat = model(v, q, q_len)
#
#     nll = -log_softmax(y_hat)
#     # a_indices = a.coalesce().indices().cpu().numpy()
#     # a_values = a.coalesce().values()
#     batch_size = y_hat.shape[0]
#     batch_indices = []  # TODO cleaner way
#     for i in range(batch_size):
#         batch_indices.extend([i] * 10)
#         # a_indices = a.coalesce().indices().cpu().numpy()
#     # a_values = a.coalesce().values()
#
#     loss = (nll[a_indices.flatten().cpu().numpy()] * (a_values.flatten() / 10.0)).sum() / batch_size
#     # loss += (nll * a.to_dense() / 10).sum(dim=1).mean()
#
#     score = torch.sum(batch_accuracy(y_hat.data, (a_indices, a_values, (batch_size, max_answers))).cpu())
#     return loss, score