from scipy.io import loadmat

from .labeled_data import LabeledSequenceData, SparseLabeledSequenceData


def read_mat(path):
    data = loadmat(path, squeeze_me=True)
    return data['X'], data['y']


def get_datasets(args, xtrain=None, ytrain=None, xtest=None, ytest=None):
    if (xtrain is None) or (ytrain is None) or (xtest is None) or (ytest is None):
        xtrain, ytrain = read_mat(args.data_train_path)
        xtest, ytest = read_mat(args.data_test_path)

    if args.is_dense:
        trainset = LabeledSequenceData(xtrain, ytrain, args.train_size)
        testset = LabeledSequenceData(xtest, ytest, args.test_size)
    else:
        trainset = SparseLabeledSequenceData(xtrain, ytrain, args.train_size)
        testset = SparseLabeledSequenceData(xtest, ytest, args.test_size,
                                            trainset.vocabulary_sizes)

    args.train_size = len(trainset)
    args.test_size = len(testset)

    return trainset, testset


def get_dataset(args, train=False, test=False, xtrain=None, ytrain=None, xtest=None, ytest=None):
    if train and (xtrain is None or ytrain is None):
        xtrain, ytrain = read_mat(args.data_train_path)
    if test and (xtest is None or ytest is None):
        xtest, ytest = read_mat(args.data_test_path)

    if args.is_dense:
        if train:
            trainset = LabeledSequenceData(xtrain, ytrain, args.train_size)
        if test:
            testset = LabeledSequenceData(xtest, ytest, args.test_size)
    else:
        if train:
            trainset = SparseLabeledSequenceData(
                xtrain, ytrain, args.train_size)
        if test:
            testset = SparseLabeledSequenceData(xtest, ytest, args.test_size,
                                                trainset.vocabulary_sizes)

    if train:
        args.train_size = len(trainset)
    if test:
        args.test_size = len(testset)
    if train:
        if test:
            return trainset, testset
        else:
            return trainset
    else:
        if test:
            return testset
        else:
            return
