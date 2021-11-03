import os
import yaml
import json
from easydict import EasyDict

def save_yaml(config):
    """
    Saves the parameter in the form of YAML file in the directory where the model is saved
    :param config: (dictionary) contains the parameters
    :return: None
    """
    path = config.SETTINGS.log_path
    if not os.path.exists(path):
        os.makedirs(path)
    with open(os.path.join(path, 'config.yaml'), 'w') as file:
        yaml.dump(config, file)


def load_yaml(path):
    """
    loads a YAML file
    :param path: (string) path to the configuration.yaml file to load
    :return: config file processed into a dictionary by EasyDict
    """
    file = yaml.load(open(path), Loader=yaml.FullLoader)
    config = EasyDict(file)

    return config

