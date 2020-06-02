import argparse
import os
import time

import numpy as np
import tensorboard_logger as tl


def get_args():
    parser = argparse.ArgumentParser(description='sdca')

    parser.add_argument('--dataset', type=str, default='conll',
                        help='which dataset to use')
    parser.add_argument('--train-size', type=int, default=None,
                        help='set to None if you want the full data set.')
    parser.add_argument('--test-size', type=int, default=None,
                        help='set to None if you want the full data set.')
    parser.add_argument('--regularization', type=float, default=None,
                        help='value of the l2 regularization parameter. '
                             'if None, will be set to 1/n.')
    parser.add_argument('--npass', type=int, default=100,
                        help='maximum number of pass over the trainset duality gaps used in the '
                             'non-uniform sampling and to get a convergence criterion.')
    parser.add_argument('--sampling-scheme', type=str, default='gap',
                        help='Type of sampling.',
                        choices=["uniform", "importance", "gap", "gap+", "max", "safemax"])
    parser.add_argument('--non-uniformity', type=float, default=0.8,
                        help='between 0 and 1. probability of sampling non-uniformly.')
    parser.add_argument('--sampler-period', type=int, default=None,
                        help='if not None, period to do a full batch update of the duality gaps, '
                             'for the non-uniform sampling. Expressed as a number of epochs. '
                             'This whole epoch will be counted in the number of pass used by sdca')
    parser.add_argument('--precision', type=float, default=1e-7,
                        help='Precision wanted on the duality gap.')
    parser.add_argument('--fixed-step-size', type=float, default=None,
                        help='if None, SDCA will use a line search. Otherwise should be a '
                             'positive float to be used as the constant step size')
    parser.add_argument('--warm-start', type=np.array, default=None,
                        help='if numpy array, used as marginals to start from.')
    parser.add_argument('--line-search', type=str, choices=['golden', 'newton'], default='newton',
                        help='Use scipy.optimize.minimize_scalar bounded golden section search, '
                             'or a custom safe bounded Newton-Raphson line search on the '
                             'derivative.')
    parser.add_argument('--subprecision', type=float, default=1e-3,
                        help='Precision of the line search on the step-size value.')
    parser.add_argument('--init-previous-step-size', type=bool, default=False,
                        help='Use the previous step size taken for a given sample to initialize '
                             'the line search?')
    parser.add_argument('--skip-line-search', type=bool, default=False,
                        help='Use the previous step size taken for a given sample if it '
                             'increases the dual objective.')
    parser.add_argument('--save', type=str, choices=['results', 'all'], default='results',
                        help='Use "all" if you want to also save the step-sizes and the optimum.')

    args = parser.parse_args()

    if args.dataset == 'ocr':
        args.data_train_path = 'SDCA/data/ocr_train.mat'
        args.data_test_path = 'SDCA/data/ocr_test.mat'
        #elif args.dataset == 'ocr_yy':
        args.data_train_path = "SDCA/data/ocr_train_yy.mat"
        args.data_test_path = "SDCA/data/ocr_test_yy.mat"   
    elif args.dataset == 'conll':
        args.data_train_path = 'data/coNLL_train.mat'
        args.data_test_path = 'data/coNLL_test.mat'
    elif args.dataset == 'ner':
        args.data_train_path = 'data/NER_train.mat'
        args.data_test_path = 'data/NER_test.mat'
    elif args.dataset == 'pos':
        args.data_train_path = 'data/POS_train.mat'
        args.data_test_path = 'data/POS_test.mat'
    else:
        raise ValueError(f'the dataset {args.dataset} is not defined')

    args.is_dense = ((args.dataset == 'ocr') or (args.dataset == 'ocr_yy'))

    if args.line_search == 'golden':
        args.use_scipy_optimize = True
    elif args.line_search == 'newton':
        args.use_scipy_optimize = False

    args.time_stamp = time.strftime("%Y%m%d_%H%M%S")

    return args


def init_logdir(args, infostring):
    args.logdir = get_logdir(args)
    os.makedirs(args.logdir)
    print(f"Logging in {args.logdir}")

    # write important informations in the log directory
    with open(args.logdir + '/parameters.txt', 'w') as file:
        file.write(infostring)
        file.write('\n')
        for arg in vars(args):
            file.write("{}:{}\n".format(arg, getattr(args, arg)))

    # initialize tensorboard logger
    args.use_tensorboard = initialize_tensorboard(args.logdir)


def get_logdir(args):
    if args.sampling_scheme == 'uniform' or args.non_uniformity <= 0:
        sampling_string = 'uniform'
    else:
        sampling_string = args.sampling_scheme
        sampling_string += str(args.non_uniformity)
        if args.sampler_period is not None:
            sampling_string += '_' + args.sampler_period

    if args.fixed_step_size is not None:
        line_search_string = 'step_size' + args.fixed_step_size
    else:
        line_search_string = f'line_search_{args.line_search}{args.subprecision}'
        if args.init_previous_step_size:
            line_search_string += "_useprevious"
        if args.skip_line_search:
            line_search_string += "_skip"

    return "logs/{}_{}/{}_{}_{}".format(
        args.dataset,
        'n' + str(args.train_size),
        args.time_stamp,
        sampling_string,
        line_search_string
    )


def initialize_tensorboard(logdir):
    try:
        tl.configure(logdir=logdir, flush_secs=15)
        return True
    except Exception as e:
        print("Not using tensorboard because of ", e)
        return False


def get_information_string(args, train_data, test_data):
    return (
        f"Time stamp: {args.time_stamp} \n"
        f"Data set: {args.dataset} \n"
        f"Size of training set: sequences {args.train_size} (nodes: {train_data.nb_points}) \n"
        f"Size of test set: sequences {args.test_size} (nodes: {test_data.nb_points}) \n"
        f"Number of labels: {train_data.nb_labels} \n"
        f"Number of attributes: {train_data.nb_features} \n \n"
    )
