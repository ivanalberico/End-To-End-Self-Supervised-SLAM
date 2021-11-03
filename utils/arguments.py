import argparse


def arguments():

    arg_parse = argparse.ArgumentParser()
    arg_parse.add_argument('--config_path', required=True, default= "", help="path to config")
    arg_parse.add_argument('--name', required= False, help="Description of training")
    args = arg_parse.parse_args()

    return vars(args)