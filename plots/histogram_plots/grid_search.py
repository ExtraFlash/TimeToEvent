from matplotlib import pyplot as plt
import utils
import os
import json


def grid_search(config_file):
    param_space = []
    num = 0.0
    while num < 50:
        param_space.append(num)
        num += 0.3
    identifiers = [str(i) for i in param_space]
    for coef, identifier in zip(param_space, identifiers):
        coefs = [1.0, coef, 0.5]
        utils.run_config(config_file, identifier, coefs)
        plt.clf()
        plt.close()


def find_best_params(config_file):
    with open(config_file, encoding='utf-8') as f:
        config = json.load(f)
    # load the configuration
    dataset_name = config['datasets_names'][0]
    model_name = config['models_names'][0]
    loss_name = config['losses_names'][0]
    prefix = f'{dataset_name}_{model_name}_{loss_name}'

    folder = os.path.dirname(__file__)
    folder = os.path.abspath(folder)
    files = os.listdir(folder)
    max_result = 0.0
    suffix_max = ''
    for file in files:
        if file.startswith(prefix):
            suffix = file[len(prefix):]
            with open(f'{folder}/{file}/results.txt', 'r') as f:
                words = f.readline().strip()
                result = float(words.split()[-1])
                if result > max_result:
                    max_result = result
                    suffix_max = suffix

    print(f'The best result is {max_result} with suffix {suffix_max}')


if __name__ == '__main__':
    find_best_params('config.json')
    # find_best_params('config.json')