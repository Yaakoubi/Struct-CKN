<<<<<<< HEAD
import random
import sys
import time

import numpy as np
import torch
from pystruct.inference.maxprod import inference_max_product

from SDCA.sdca4crf.arguments import get_information_string, init_logdir
from SDCA.sdca4crf.get_datasets import *
from SDCA.sdca4crf.sdca import sdca
from SDCA.sdca4crf.utils import infer_probas
from inference_methods_local import inference_ad3_local

global current_gpu
global list_gpus


def select_word(x, y, length, beginning, index):
	begin = beginning[index]
	end = beginning[index] + length[index] - 1
	word_x = x[begin:end + 1]
	word_y = y[begin:end + 1]
	return word_x, word_y


def select_batch_words(x, y, length, beginning, indices):
	words_x = []
	words_y = []
	for i in indices:
		if i < len(beginning):
			word_x, word_y = select_word(x, y, length, beginning, i)
			if torch.is_tensor(word_x):
				words_x.append(word_x.cpu().numpy())
			else:
				words_x.append(word_x)
			words_y.append(word_y)
	return np.array(words_x), np.array(words_y)


def select_probs(probabilities, length, beginning, indices):
	probs = []
	for i in indices:
		begin = beginning[i]
		end = beginning[i] + length[i] - 1
		prob = probabilities[begin:end + 1]
		probs.append(prob)
	return np.concatenate(probs, 0)


def sentences_to_words(x, y):
	letters_x = []
	letters_y = []
	lengths = []
	for i in range(x.shape[0]):
		lengths.append(x[i].shape[0])
		for j in range(x[i].shape[0]):
			try:
				letters_x.append(x[i][j].toarray().reshape(-1))
			except AttributeError:
				letters_x.append(x[i][j])
			letters_y.append(y[i][j])
	return np.array(letters_x), np.array(letters_y), lengths


def load_data(x_train_flat, y_train_flat, len_sentences_train, begin_sent_indices_train, sentences_y_train, batch,
			  use_cuda, dim1, dim2, komninos, return_test, verbose=True):
	batch_indices = batch
	shuffled = list(range(len(sentences_y_train)))
	num_labels = len(np.unique(y_train_flat, return_counts=False))
	random.shuffle(shuffled)
	for i in range(num_labels):
		for j in shuffled:
			if i in sentences_y_train[j]:
				if j not in batch_indices:
					batch_indices.append(j)
				break
	if verbose:
		print("Batch :", batch_indices)

	sentences_x_train2, sentences_y_train2 = select_batch_words(
		x_train_flat, y_train_flat, len_sentences_train, begin_sent_indices_train, batch_indices)
	x_train_flat2, y_train_flat2, len_sentences_train2 = sentences_to_words(
		sentences_x_train2, sentences_y_train2)
	data = torch.LongTensor(x_train_flat2.astype("float32"))
	if use_cuda:
		data = data.cuda()
	n_d, dim11, dim21 = data.size()
	assert dim11 == dim1 and dim21 == dim2
	data = data.view(n_d, dim1, dim2)
	del x_train_flat2
	if komninos:
		data3 = np.load("conll_data.npz")
		sentences_x_test = data3["sentences_x_test"]
		sentences_y_test = data3["sentences_y_test"]
		del data3
	else:
		test = np.load("elmo_conll_data_test.npz")
		sentences_x_test = test["sentences_x_test"]
		sentences_y_test = test["sentences_y_test"]
		del test

	x_test_flat, y_test_flat, len_sentences_test = sentences_to_words(
		sentences_x_test, sentences_y_test)
	begin_sent_indices_test = np.concatenate([[0], [int(np.sum(
		len_sentences_test[:j])) for j in range(1, len(len_sentences_test))]]).reshape(-1)

	if return_test:
		return data, y_train_flat2, sentences_x_train2, sentences_y_train2, len_sentences_train2, x_test_flat, y_test_flat, sentences_x_test, sentences_y_test, len_sentences_test, begin_sent_indices_test
	else:
		return data, y_train_flat2, sentences_x_train2, sentences_y_train2, len_sentences_train2


def chunks(l, n):
	for i in range(0, len(l), n):
		yield l[i:i + n]


def check_nan(x):
	for array in x:
		if np.isnan(array).any():
			return True
	return False


def remove_nans(x, y, return_indices=False):
	rows = []
	for index in range(x.shape[0]):
		if np.isnan(x[index].astype(np.float64)).any():
			rows.append(index)
	print("Removing NaNs")
	print("Shape of indices : ", len(rows))
	if return_indices:
		return np.delete(x, rows, 0), np.delete(y, rows, 0), rows
	else:
		return np.delete(x, rows, 0), np.delete(y, rows, 0)


def turn_dense_to_one_matrix(a, m):
	n = len(a)
	b = np.ones((n, m)) * 1e-4
	b[np.arange(n), a] = 1
	b = b / b.sum(axis=1, keepdims=1)
	return b


def batch_marginals(X, Y, w, crf, num_classes=None):
	ys = []
	marginalss = []
	for x in X:
		y = inference_max_product(crf._get_unary_potentials(x, w),
								  crf._get_pairwise_potentials(x, w),
								  crf._get_edges(x),
								  relaxed=True)
		marginals = turn_dense_to_one_matrix(y, num_classes)

		ys.append(y)
		marginalss.append(marginals)
	return np.array(ys), np.array(marginalss)


def free_memory():
	print('Allocated:', round(torch.cuda.memory_allocated() / 1024. ** 3., 1), 'GB')
	print('Cached:   ', round(torch.cuda.memory_cached() / 1024. ** 3., 1), 'GB')
	for i in list_gpus:
		torch.cuda.set_device(i)
		try:
			torch.cuda.empty_cache()
		except RuntimeError:
			pass
	torch.cuda.set_device(current_gpu)
	print('Allocated:', round(torch.cuda.memory_allocated() / 1024. ** 3., 1), 'GB')
	print('Cached:   ', round(torch.cuda.memory_cached() / 1024. ** 3., 1), 'GB')


class Unbuffered(object):
	def __init__(self, stream):
		self.stream = stream

	def write(self, data):
		self.stream.write(data)
		self.stream.flush()

	def writelines(self, datas):
		self.stream.writelines(datas)
		self.stream.flush()

	def __getattr__(self, attr):
		return getattr(self.stream, attr)


def redirect_buffer_in(filename):
	from datetime import datetime
	now = datetime.now()
	dt_string = now.strftime("%Y_%m_%d_%H_%M_%S")
	orig_stdout = sys.stdout
	log_filename = filename + "_" + dt_string + ".log"
	f = open(log_filename, 'w')
	return f, orig_stdout


def fit_sdca(include_eos=False, xtrain=None, ytrain=None, xtest=None, ytest=None, marginals=None,
			 args=None):
	args.dataset = 'ocr'
	if args.regParSDCA != -1:
		args.regularization = args.regParSDCA
	else:
		args.regularization = None
	args.save = "all"
	args.skip_line_search = False

	args.train_size = None
	args.test_size = None
	args.precision = 1e-7
	args.fixed_step_size = None
	if args.use_warm_start != 0:
		args.warm_start = marginals
	else:
		args.warm_start = None
	args.logdir = "SDCA/logs/"

	args.is_dense = (args.dataset == 'ocr')

	if args.line_search == 'golden':
		args.use_scipy_optimize = True
	elif args.line_search == 'newton':
		args.use_scipy_optimize = False

	args.time_stamp = time.strftime("%Y%m%d_%H%M%S")
	args.subprecision = 1e-3
	
	if (not (xtrain is None)) and (not (ytrain is None)) and (not (xtest is None)) and (
			not (ytest is None)):
		train_data, test_data = get_datasets(
			args, xtrain, ytrain, xtest, ytest)
	else:
		args.data_train_path = "SDCA/data/ocr_train_struct_ckn.mat"
		args.data_test_path = "SDCA/data/ocr_test_struct_ckn.mat"
		train_data, test_data = get_datasets(args)
	if args.regularization is None:
		args.regularization = 1 / args.train_size
	infostring = get_information_string(args, train_data, test_data)
	init_logdir(args, infostring)
	optweights, optmargs = sdca(
		trainset=train_data, testset=test_data, args=args)
	predicted_labels_train = infer_probas(
		train_data, optweights, include_eos=include_eos)
	if args.save == 'all':
		np.save(args.logdir + '/opttransition.npy', optweights.transition)
		np.save(args.logdir + '/optbias.npy', optweights.bias)
		np.save(args.logdir + '/optemission.npy', optweights.emission)
		marginals_dic = {'marginal' + str(i): margs.binary for i, margs in enumerate(optmargs)}
		np.savez_compressed(args.logdir + '/optmarginals.npy', **marginals_dic)
	return predicted_labels_train, optweights, optmargs


def predict_sdca(weights, train=True, test=False, include_eos=False, xtrain=None, ytrain=None,
				 xtest=None, ytest=None, marginals=None, args=None):
	args.dataset = 'ocr'
	args.regularization = None
	args.save = "all"
	args.skip_line_search = False
	args.train_size = None
	args.test_size = None
	args.precision = 1e-7
	args.fixed_step_size = None
	if args.use_warm_start != 0:
		args.warm_start = marginals
	else:
		args.warm_start = None
	args.logdir = "SDCA/logs/"
	args.is_dense = (args.dataset == 'ocr')
	if args.line_search == 'golden':
		args.use_scipy_optimize = True
	elif args.line_search == 'newton':
		args.use_scipy_optimize = False

	args.time_stamp = time.strftime("%Y%m%d_%H%M%S")
	args.subprecision = 1e-3
	if train:
		args.data_train_path = "SDCA/data/ocr_train_struct_ckn.mat"
		train_data = get_dataset(args, train=True,test=False,xtrain=xtrain,ytrain=ytrain)
	if test: 
		args.data_test_path = "SDCA/data/ocr_test_struct_ckn.mat"
		test_data = get_dataset(args, train=False,test=True,xtest=xtest,ytest=ytest)
	if train:
		predicted_labels_train = infer_probas(
			train_data, weights, include_eos=include_eos)
	if test:
		predicted_labels_test = infer_probas(
			test_data, weights, include_eos=include_eos)
	if train and test:
		return predicted_labels_train, predicted_labels_test
	if not train and test:
		return predicted_labels_test
	if train and not test:
		return predicted_labels_train
	return


def sdca_marginals(self, param_x, param_y, w, num_classes=None):
	ys = []
	marginalss = []
	for x in param_x:
		if self.inference_method == "max-product":
			y = inference_max_product(self._get_unary_potentials(
				x, w), self._get_pairwise_potentials(x, w), self._get_edges(x), relaxed=True)
			marginals = self.turn_dense_to_one_matrix(y, num_classes)
		elif self.inference_method == "ad3":
			y, marginals = inference_ad3_local(self._get_unary_potentials(x, w), self._get_pairwise_potentials(
				x, w), self._get_edges(x), relaxed=True, branch_and_bound=False, return_marginals=True)
		else:
			print("Unknown inference method !")
		ys.append(y)
		marginalss.append(marginals)
	return np.array(ys), np.array(marginalss)


def synchonize_y(original=None, new_y=None, min_y=None, max_y=None):
	new_y_local = new_y.copy()
	if original is not None:
		min_y = np.min(original)
	new_min_y = np.min(new_y_local)
	if new_min_y == min_y - 1:
		new_y_local += 1
	elif new_min_y == min_y + 1:
		new_y_local -= 1
	elif new_min_y == min_y:
		pass
	else:
		diff = min_y - new_min_y
		new_y_local += diff
	return new_y_local


def words_to_letters(x, y):
	letters_x = []
	letters_y = []
	lengths = []
	for i in range(x.shape[0]):
		lengths.append(x[i].shape[0])
		for j in range(x[i].shape[0]):
			letters_x.append(x[i][j])
			letters_y.append(y[i][j])
	return np.array(letters_x), np.array(letters_y), lengths


def reconstruct(x, y, lengths):
	words_x = []
	words_y = []
	index = 0
	for length in lengths:
		new_word_x = [x[i].numpy() for i in range(index, index + length)]
		new_word_y = [y[i] for i in range(index, index + length)]

		words_x.append(np.array(new_word_x))
		words_y.append(np.array(new_word_y))
		index = index + length
	return np.array(words_x), np.array(words_y)


def permute_array(x):
	new_x = []
	for i in range(x.shape[0]):
		new_x.append(np.transpose(x[i], (0, 2, 1)))
	return np.array(new_x)


def flatten_array(x):
	new_x = []
	for i in range(x.shape[0]):
		new_x.append(x[i].reshape((x[i].shape[0], -1)))
	return np.array(new_x)
	
def print_args(args):
	return (
		f"Learning rate: {args.lr} \n"
		f"Size of CKN patch: {args.size_patch} \n"
		f"Predictor: {args.predictor} \n"
	)
	
					
||||||| merged common ancestors
=======
import random
import sys
import time

import numpy as np
import torch
from pystruct.inference.maxprod import inference_max_product

from SDCA.sdca4crf.arguments import get_information_string, init_logdir
from SDCA.sdca4crf.get_datasets import get_datasets
from SDCA.sdca4crf.sdca import sdca
from SDCA.sdca4crf.utils import infer_probas
from inference_methods_local import inference_ad3_local

global current_gpu
global list_gpus


def select_word(x, y, length, beginning, index):
    begin = beginning[index]
    end = beginning[index] + length[index] - 1
    word_x = x[begin:end + 1]
    word_y = y[begin:end + 1]
    return word_x, word_y


def select_batch_words(x, y, length, beginning, indices):
    words_x = []
    words_y = []
    for i in indices:
        if i < len(beginning):
            word_x, word_y = select_word(x, y, length, beginning, i)
            if torch.is_tensor(word_x):
                words_x.append(word_x.cpu().numpy())
            else:
                words_x.append(word_x)
            words_y.append(word_y)
    return np.array(words_x), np.array(words_y)


def select_probs(probabilities, length, beginning, indices):
    probs = []
    for i in indices:
        begin = beginning[i]
        end = beginning[i] + length[i] - 1
        prob = probabilities[begin:end + 1]
        probs.append(prob)
    return np.concatenate(probs, 0)


def sentences_to_words(x, y):
    letters_x = []
    letters_y = []
    lengths = []
    for i in range(x.shape[0]):
        lengths.append(x[i].shape[0])
        for j in range(x[i].shape[0]):
            try:
                letters_x.append(x[i][j].toarray().reshape(-1))
            except AttributeError:
                letters_x.append(x[i][j])
            letters_y.append(y[i][j])
    return np.array(letters_x), np.array(letters_y), lengths


def load_data(x_train_flat, y_train_flat, len_sentences_train, begin_sent_indices_train, sentences_y_train, batch,
              use_cuda, dim1, dim2, komninos, return_test, verbose=True):
    batch_indices = batch
    shuffled = list(range(len(sentences_y_train)))
    num_labels = len(np.unique(y_train_flat, return_counts=False))
    random.shuffle(shuffled)
    for i in range(num_labels):
        for j in shuffled:
            if i in sentences_y_train[j]:
                if j not in batch_indices:
                    batch_indices.append(j)
                break
    if verbose:
        print("Batch :", batch_indices)

    sentences_x_train2, sentences_y_train2 = select_batch_words(
        x_train_flat, y_train_flat, len_sentences_train, begin_sent_indices_train, batch_indices)
    x_train_flat2, y_train_flat2, len_sentences_train2 = sentences_to_words(
        sentences_x_train2, sentences_y_train2)
    data = torch.LongTensor(x_train_flat2.astype("float32"))
    if use_cuda:
        data = data.cuda()
    n_d, dim11, dim21 = data.size()
    assert dim11 == dim1 and dim21 == dim2
    data = data.view(n_d, dim1, dim2)
    del x_train_flat2
    if komninos:
        data3 = np.load("conll_data.npz")
        sentences_x_test = data3["sentences_x_test"]
        sentences_y_test = data3["sentences_y_test"]
        del data3
    else:
        test = np.load("elmo_conll_data_test.npz")
        sentences_x_test = test["sentences_x_test"]
        sentences_y_test = test["sentences_y_test"]
        del test

    x_test_flat, y_test_flat, len_sentences_test = sentences_to_words(
        sentences_x_test, sentences_y_test)
    begin_sent_indices_test = np.concatenate([[0], [int(np.sum(
        len_sentences_test[:j])) for j in range(1, len(len_sentences_test))]]).reshape(-1)

    if return_test:
        return data, y_train_flat2, sentences_x_train2, sentences_y_train2, len_sentences_train2, x_test_flat, y_test_flat, sentences_x_test, sentences_y_test, len_sentences_test, begin_sent_indices_test
    else:
        return data, y_train_flat2, sentences_x_train2, sentences_y_train2, len_sentences_train2


def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]


def check_nan(x):
    for array in x:
        if np.isnan(array).any():
            return True
    return False


def remove_nans(x, y, return_indices=False):
    rows = []
    for index in range(x.shape[0]):
        if np.isnan(x[index].astype(np.float64)).any():
            rows.append(index)
    print("Removing NaNs")
    print("Shape of indices : ", len(rows))
    if return_indices:
        return np.delete(x, rows, 0), np.delete(y, rows, 0), rows
    else:
        return np.delete(x, rows, 0), np.delete(y, rows, 0)


def turn_dense_to_one_matrix(a, m):
    n = len(a)
    b = np.ones((n, m)) * 1e-4
    b[np.arange(n), a] = 1
    b = b / b.sum(axis=1, keepdims=1)
    return b


def batch_marginals(X, Y, w, crf, num_classes=None):
    ys = []
    marginalss = []
    for x in X:
        y = inference_max_product(crf._get_unary_potentials(x, w),
                                  crf._get_pairwise_potentials(x, w),
                                  crf._get_edges(x),
                                  relaxed=True)
        marginals = turn_dense_to_one_matrix(y, num_classes)

        ys.append(y)
        marginalss.append(marginals)
    return np.array(ys), np.array(marginalss)


def free_memory():
    print('Allocated:', round(torch.cuda.memory_allocated() / 1024. ** 3., 1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_cached() / 1024. ** 3., 1), 'GB')
    for i in list_gpus:
        torch.cuda.set_device(i)
        try:
            torch.cuda.empty_cache()
        except RuntimeError:
            pass
    torch.cuda.set_device(current_gpu)
    print('Allocated:', round(torch.cuda.memory_allocated() / 1024. ** 3., 1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_cached() / 1024. ** 3., 1), 'GB')


class Unbuffered(object):
    def __init__(self, stream):
        self.stream = stream

    def write(self, data):
        self.stream.write(data)
        self.stream.flush()

    def writelines(self, datas):
        self.stream.writelines(datas)
        self.stream.flush()

    def __getattr__(self, attr):
        return getattr(self.stream, attr)


def redirect_buffer_in(filename):
    from datetime import datetime
    now = datetime.now()
    dt_string = now.strftime("%Y_%m_%d_%H_%M_%S")
    orig_stdout = sys.stdout
    log_filename = filename + "_" + dt_string + ".log"
    f = open(log_filename, 'w')
    return f, orig_stdout


def fit_sdca(include_eos=False, filename=None, xtrain=None, ytrain=None, xtest=None, ytest=None, marginals=None,
             args=None):
    args.dataset = 'ocr'
    if args.regParSDCA != -1:
        args.regularization = args.regParSDCA
    else:
        args.regularization = None
    args.save = None
    args.skip_line_search = False

    args.train_size = None
    args.test_size = None
    args.precision = 1e-7
    args.fixed_step_size = None
    if args.use_warm_start != 0:
        print("Using warm start")
        args.warm_start = marginals
    else:
        args.warm_start = None
    args.logdir = "SDCA/logs/"

    args.is_dense = (args.dataset == 'ocr')

    if args.line_search == 'golden':
        args.use_scipy_optimize = True
    elif args.line_search == 'newton':
        args.use_scipy_optimize = False

    args.time_stamp = time.strftime("%Y%m%d_%H%M%S")
    args.subprecision = 1e-3
    if (filename is None) and (not (xtrain is None)) and (not (ytrain is None)) and (not (xtest is None)) and (
            not (ytest is None)):
        train_data, test_data = get_datasets(
            args, xtrain, ytrain, xtest, ytest)
    else:
        args.data_train_path = "SDCA/data/ocr_train_yy_" + filename + ".mat"
        args.data_test_path = "SDCA/data/ocr_test_yy_" + filename + ".mat"
        train_data, test_data = get_datasets(args)
    if args.regularization is None:
        args.regularization = 1 / args.train_size
    infostring = get_information_string(args, train_data, test_data)
    init_logdir(args, infostring)
    optweights, optmargs = sdca(
        trainset=train_data, testset=test_data, args=args)
    predicted_labels_train = infer_probas(
        train_data, optweights, include_eos=include_eos)
    if args.save == 'all':
        np.save(args.logdir + '/opttransition.npy', optweights.transition)
        np.save(args.logdir + '/optbias.npy', optweights.bias)
        np.save(args.logdir + '/optemission.npy', optweights.emission)
        marginals_dic = {'marginal' + str(i): margs.binary for i, margs in enumerate(optmargs)}
        np.savez_compressed(args.logdir + '/optmarginals.npy', **marginals_dic)
    return predicted_labels_train, optweights, optmargs


def predict_sdca(weights, train=True, test=False, include_eos=False, filename=None, xtrain=None, ytrain=None,
                 xtest=None, ytest=None, marginals=None, args=None):
    args.dataset = 'ocr'
    args.regularization = None
    args.save = "all"
    args.skip_line_search = False
    args.train_size = None
    args.test_size = None
    args.precision = 1e-7
    args.fixed_step_size = None
    if args.use_warm_start != 0:
        args.warm_start = marginals
    else:
        args.warm_start = None
    args.logdir = "SDCA/logs/"

    args.is_dense = (args.dataset == 'ocr')

    if args.line_search == 'golden':
        args.use_scipy_optimize = True
    elif args.line_search == 'newton':
        args.use_scipy_optimize = False

    args.time_stamp = time.strftime("%Y%m%d_%H%M%S")
    args.subprecision = 1e-3

    if (filename is None) and (not (xtrain is None)) and (not (ytrain is None)) and (not (xtest is None)) and (
            not (ytest is None)):
        train_data, test_data = get_datasets(
            args, xtrain, ytrain, xtest, ytest)
    else:
        args.data_train_path = "SDCA/data/ocr_train_yy_" + filename + ".mat"
        args.data_test_path = "SDCA/data/ocr_test_yy_" + filename + ".mat"
        train_data, test_data = get_datasets(args)

    if train:
        predicted_labels_train = infer_probas(
            train_data, weights, include_eos=include_eos)
    if test:
        predicted_labels_test = infer_probas(
            test_data, weights, include_eos=include_eos)
    if train and test:
        return predicted_labels_train, predicted_labels_test
    if not train and test:
        return predicted_labels_test
    if train and not test:
        return predicted_labels_train
    return


def sdca_marginals(self, param_x, param_y, w, num_classes=None):
    ys = []
    marginalss = []
    for x in param_x:
        if self.inference_method == "max-product":
            y = inference_max_product(self._get_unary_potentials(
                x, w), self._get_pairwise_potentials(x, w), self._get_edges(x), relaxed=True)
            marginals = self.turn_dense_to_one_matrix(y, num_classes)
        elif self.inference_method == "ad3":
            y, marginals = inference_ad3_local(self._get_unary_potentials(x, w), self._get_pairwise_potentials(
                x, w), self._get_edges(x), relaxed=True, branch_and_bound=False, return_marginals=True)
        else:
            print("Unknown inference method !")
        ys.append(y)
        marginalss.append(marginals)
    return np.array(ys), np.array(marginalss)


def synchonize_y(original=None, new_y=None, min_y=None, max_y=None):
    new_y_local = new_y.copy()
    if original is not None:
        min_y = np.min(original)
    new_min_y = np.min(new_y_local)
    if new_min_y == min_y - 1:
        new_y_local += 1
    elif new_min_y == min_y + 1:
        new_y_local -= 1
    elif new_min_y == min_y:
        pass
    else:
        diff = min_y - new_min_y
        new_y_local += diff
    return new_y_local


def words_to_letters(x, y):
    letters_x = []
    letters_y = []
    lengths = []
    for i in range(x.shape[0]):
        lengths.append(x[i].shape[0])
        for j in range(x[i].shape[0]):
            letters_x.append(x[i][j])
            letters_y.append(y[i][j])
    return np.array(letters_x), np.array(letters_y), lengths


def reconstruct(x, y, lengths):
    words_x = []
    words_y = []
    index = 0
    for length in lengths:
        new_word_x = [x[i].numpy() for i in range(index, index + length)]
        new_word_y = [y[i] for i in range(index, index + length)]

        words_x.append(np.array(new_word_x))
        words_y.append(np.array(new_word_y))
        index = index + length
    return np.array(words_x), np.array(words_y)


def permute_array(x):
    new_x = []
    for i in range(x.shape[0]):
        new_x.append(np.transpose(x[i], (0, 2, 1)))
    return np.array(new_x)


def flatten_array(x):
    new_x = []
    for i in range(x.shape[0]):
        new_x.append(x[i].reshape((x[i].shape[0], -1)))
    return np.array(new_x)
    
def print_args(args):
    return (
        f"Learning rate: {args.lr} \n"
        f"Size of CKN patch: {args.size_patch} \n"
        f"Predictor: {args.predictor} \n"
    )
    
                    
>>>>>>> 824794b94a0c0d5dbc17a904ff250da5dfdac052
