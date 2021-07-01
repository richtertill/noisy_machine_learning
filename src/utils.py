import json
import logging
import time
from datetime import datetime

import torch

TRAINING_INFO_LEVEL_NUM = 15


class Params:
    # Source code from <https://github.com/cs230-stanford/cs230-code-examples>
    """Class that loads training params and model hyperparameters from a json file.

    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__


def save_checkpoint(model, optimizer, best_loss, epoch, checkpoint_file):
    """
    Saves checkpoint of torchvision model during training.
    :param model: torchvision model to be saved
    :param best_loss: best val loss achieved so far in training
    :param epoch: current epoch of training
    :param optimizer: optimizer object needed for saving its state for later training resuming
    :param checkpoint_file: file path where save the checkpoint
    :return: None
    """

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': best_loss,
        'rng_state': torch.get_rng_state()
    }, checkpoint_file)


def load_checkpoint(model, checkpoint_file, optimizer=None):
    """
    Loads model's state (i.e. weights), optimizer's state from a checkpoint file.
    Restores the random state of PyTorch.
    It also returns the corresponding loss and epoch needed to continue the training.
    :param model: model object for which the parameters are loaded
    :param checkpoint_file: file path where the checkpoint is stored
    :param optimizer: (optional) optimizer object which the state is loaded
    :return: loss and epoch
    """
    if torch.cuda.is_available():
        checkpoint = torch.load(checkpoint_file, map_location=torch.device('cuda'))
    else:
        checkpoint = torch.load(checkpoint_file, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    torch.random.set_rng_state(checkpoint['rng_state'].cpu())
    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    return checkpoint['loss'], checkpoint['epoch']


def now_as_ts():
    return time.time()


def now_as_str():
    return "{:%Y_%m_%d---%H_%M}".format(datetime.now())


def now_as_str_f():
    return "{:%Y_%m_%d---%H_%M_%f}".format(datetime.now())


def ts_to_str(ts):
    return "{:%Y_%m_%d---%H_%M}".format(datetime.fromtimestamp(ts))


def ts_to_str_f(ts):
    return "{:%Y_%m_%d---%H_%M_%f}".format(datetime.fromtimestamp(ts))


def get_logger(log_path=None, log_level=logging.DEBUG):
    logging.getLogger().setLevel(logging.WARNING)
    logger = logging.getLogger('uncertainty_estimation_in_dl')


    '''
    if log_path and not logger.handlers:
        print('log successful')
        # set the level
        logger.setLevel(log_level)
        
        # Logging to a file
        f = '[%(asctime)s][%(levelname)s][%(message)s]'
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter(fmt=f, datefmt='%d/%m-%H:%M:%S'))
        logger.addHandler(file_handler)
        
        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter(fmt=f, datefmt='%d/%m-%H:%M:%S'))
        logger.addHandler(stream_handler)
        
        # Add a new level between debug and info for printing logs while training
        logging.addLevelName(TRAINING_INFO_LEVEL_NUM, 'TINFO')
        setattr(logger, 'tinfo', lambda *args: logger.log(TRAINING_INFO_LEVEL_NUM, *args))
    '''
    return logger
"""utils.py - Helper functions for building the model and for loading model parameters.
   These helper functions are built to mirror those in the official TensorFlow implementation.
"""
