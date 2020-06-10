import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from .Cell import *
import inspect

def count_parameters(model):
	return sum(p.numel() for p in model.parameters() if p.requires_grad)


def turn_dense_to_one_matrix(a, m, zero_prob=0):
	n = len(a)
	b = np.ones((n, m)) * zero_prob
	b[np.arange(n), a] = 1
	b = b / b.sum(axis=1, keepdims=1)
	return b


# CNN Model


class CKN(nn.Module):
	def __init__(self, dim1=28, dim2=28, n_components=[10, 10, 10], n_layers=3, n_patches=[5, 2, 2],
				 subsampling_factors=[2, 2, 1], quantiles=[0.1, 0.1, 0.1], spacing=[2, 2, 2],
				 batch_size=[100, 200, 500], iter_max=100, n_patches_per_graph=[30, 20, 20],
				 use_cuda=True, lr=0.1):
		super(CKN, self).__init__()
		self.Kernel = {k: Cell(n_components=n_components[k], size_patch=n_patches[k], iter_max=iter_max,
								 subsampling_factor=subsampling_factors[k], spacing=spacing[k],
								 n_patches_per_graph=n_patches_per_graph[k], dim1=dim1, dim2=dim2, use_cuda=use_cuda,
								 lr=lr) for k in
						 range(n_layers)}
		self.n_patches = n_patches
		self.n_layers = n_layers
		self.n_patches_per_graph = n_patches_per_graph
		self.subsampling_factors = subsampling_factors
		self.model_patch = {k: rel_distance_patch(
			n_patches[k], use_cuda=use_cuda) for k in range(n_layers)}
		self.batch_size = batch_size
		self.embedding_layer = None
		self.linear_layer = None

	def train_network_sup(self, X, letters_y_train, clf, scaler, struct_model=None, X_train_words=None,
							y_train_words=None, lengths_train=None, init=True, verbose=False, opt=None,
							numberEpochsCKN=10, use_scaler=True, numberEpochsFW=1, X_test=None, letters_y_test=None,
							X_test_words=None, y_test_words=None, lengths_test=None, use_embedding=False,
							embedding_dim=10, return_test_acc=True, use_cuda=True, use_clfs=None, embedding_vocab=0,
							zero_prob=0, args=None, save_checkpoint=True):
		'''
		trains each cell individually, using the outputs of the k-1th layer as
		input to the kth layer

		INPUT
		----------------------------------------------------------
		X:	training dataset
		OUTPUT
		----------------------------------------------------------
		self: trained architecture
		input_map:	the last activation map of the last layer.
		'''
		if X_train_words is None or y_train_words is None or lengths_train is None or struct_model is None:
			use_fw2 = False
		else:
			use_fw2 = True
		use_logistic_regression, use_fw, use_linear_svc, use_sdca4crf = use_clfs
		assert (use_fw2 == use_fw)
		if scaler is None:
			use_scaler = False
		output_map = X
		if init:
			self.test_accs = []
		if init:
			self.train_accs = []
		if init:
			self.num_labels = max(len(np.unique(letters_y_train, return_counts=False)), len(
				np.unique(letters_y_test, return_counts=False)))
		if not (args is None):
			warm_start = args.use_warm_start
		if init:	# changed
			if use_embedding:
				if verbose:
					print("size of output map before embedding : ", X.size())
				if init:
					self.embedding_layer = nn.Embedding(
						embedding_vocab, embedding_dim)
				if use_cuda:
					self.embedding_layer.cuda()
				output_map = self.embedding_layer(X)
				if verbose:
					print("size of output map after embedding : ",
							output_map.size())
				if verbose:
					print("first element :", output_map[0])
			else:
				self.embedding_layer = None
			for k in range(self.n_layers):
				if verbose:
					print(output_map.size())
				if k == 0:
					self.Kernel[k].init_W(
						X, patches_given=False, use_cuda=use_cuda, init_unsup=True)
					print("finished initializing")
					input_map = self.Kernel[0].all_patches
					if verbose:
						print('input size at layer ', k,
								' : ', input_map.size())
				else:
					input_map = output_map
					if verbose:
						print('input size at layer ', k,
								' : ', input_map.size())
					self.Kernel[k].init_W(
						input_map, patches_given=True, use_cuda=use_cuda, init_unsup=True)
				self.Kernel[k].training_data_has_been_normalized = False
				self.Kernel[k].training_data_has_been_scaled = False
				self.Kernel[k].fit(
					input_map, init=False, use_cuda=use_cuda, retrain_graph=False)	# 1508?
				output_map = self.Kernel[k].get_activation_map(
					k, verbose=False, use_cuda=use_cuda).permute(1, 0, 2)
		else:
			output_map = self.propagate_through_network(
				X, patches_given=False, use_cuda=use_cuda).permute(1, 0, 2)
		k = 0
		N = output_map.size()[0]
		if verbose:
			print("N = ", N)
		expected_output = Variable(torch.LongTensor(letters_y_train), requires_grad=False)
		if use_cuda:
			expected_output = expected_output.cuda()
		loss_func = nn.CrossEntropyLoss()
		if verbose:
			print('size input: ', output_map.size())
		if verbose:
			print('size labels: ', letters_y_train.shape)
		if verbose:
			print('First 100 labels: ', letters_y_train[:100])
		if verbose:
			print("will define optimizer")
		if init is True and opt is None:
			if self.Kernel[0].optimizer is not None:
				optimizer = self.Kernel[0].optimizer
			else:
				if use_embedding:
					parameters = [self.Kernel[k].eta,
									self.Kernel[k].W, self.embedding_layer.weight]
				else:
					parameters = [self.Kernel[k].eta, self.Kernel[k].W]
				model_parameters = filter(
					lambda p: p.requires_grad, parameters)
				optimizer = optim.Adamax(model_parameters, lr=self.Kernel[k].lr)
		else:
			optimizer = opt
		if verbose:
			print("Using optimizer : ", optimizer)
		self.Kernel[k].batch_size = 100
		batch_nb = N // self.Kernel[k].batch_size
		batch_size = self.Kernel[k].batch_size
		if verbose:
			print("batch_nb : ", batch_nb)
		p_size = output_map.size()[2]
		D = self.Kernel[k].n_components
		list_batches = list(range(batch_nb))
		for t in range(numberEpochsCKN):
			print("Struct-CKN -- Epoch ", t)
			if init:
				self.train_acc = 0
			random.shuffle(list_batches)
			if t != 0:
				self.Kernel[k].training_data_has_been_normalized = False
				self.Kernel[k].training_data_has_been_scaled = False
				output_map = self.propagate_through_network(
					X, patches_given=False, use_cuda=use_cuda, ).permute(1, 0, 2)
			if verbose:
				print("output_map.size() ", output_map.size())
			if use_fw:
				data_for_clf = output_map.data.cpu().numpy().reshape((N, -1))
				if use_scaler:
					scaled_train = scaler.fit_transform(data_for_clf)
				else:
					scaled_train = data_for_clf
				if verbose:
					print("fit_transform finished")
				X2_train, y2_train = reconstruct(
					scaled_train, letters_y_train, lengths_train, cuda=False)
				# print("First 10 - 2408 : " , y2_train[:10])
				# if t == 0 : #and init == True :
				#	clf.fit(X2_train,y2_train,initialize=True)
				if t == 0 and init:
					marginals = np.ones(
						(N, self.num_labels), dtype=np.float32) * (1. / float(self.num_labels))
				else:
					self.train_acc = clf.score(X2_train, y2_train)
					self.train_accs.append(self.train_acc)
					y, marginals = struct_model.batch_marginals(
						X2_train, y2_train, clf.w, num_classes=self.num_labels, zero_prob=args.zero_prob)
				if t == 0 and init:	# or init == False :
					clf.fit(X2_train, y2_train, initialize=True)
				else:
					clf.fit(X2_train, y2_train, initialize=warm_start)	# 1508?
				print("Train score : ", self.train_acc)
				if return_test_acc:
					test_acc = self.test_clf_sup(X_test, letters_y_test, clf, scaler, struct_model,
												 X_test_words, y_test_words, lengths_test, use_scaler, use_cuda,
												 verbose=False)
				if return_test_acc:
					self.test_accs.append(test_acc)
				if return_test_acc:
					print("Test acc: ", test_acc)
				if (t != 0) or (not init):
					y = np.concatenate(y, axis=0)
				if (t != 0) or (not init):
					marginals = np.concatenate(marginals, axis=0)
				marginals = marginals.reshape(N, self.num_labels)
			elif use_sdca4crf:
				data_for_clf = output_map.data.cpu().numpy().reshape((N, -1))
				if use_scaler:
					scaled_train = scaler.fit_transform(
						np.nan_to_num(data_for_clf))
					if verbose:
						print("fit_transform finished")
				else:
					scaled_train = data_for_clf
				if verbose:
					print("fit_transform finished")
				X2_train, y2_train = write_mat_file(
					scaled_train, letters_y_train, lengths_train, train=True, write_file=False)
				scaled_data_for_test = self.propagate_sdca(
					X_test, letters_y_test, clf, scaler, struct_model, X_test_words, y_test_words, lengths_test,
					use_scaler, use_cuda)
				X2_test, y2_test = write_mat_file(
					scaled_data_for_test, letters_y_test, lengths_test, train=False, write_file=False)
				if t == 0 and init:
					predicted_labels_train, optweights, optmargs = fit_sdca(
						include_eos=False, xtrain=X2_train, ytrain=y2_train,
						xtest=X2_test, ytest=y2_test, args=args)
					predicted_labels_train = synchonize_y(
						original=letters_y_train, new_y=predicted_labels_train)
				else:
					predicted_labels_train, predicted_labels_test = predict_sdca(
						optweights, train=True, test=True, include_eos=False,
						xtrain=X2_train, ytrain=y2_train, xtest=X2_test, ytest=y2_test, marginals=optmargs, args=args)
					predicted_labels_train = synchonize_y(
						original=letters_y_train, new_y=predicted_labels_train)
					predicted_labels_test = synchonize_y(
						original=letters_y_test, new_y=predicted_labels_test)
					print("Train Acc : ", np.sum((np.array(predicted_labels_train ==
															 letters_y_train)) * 1), " / ",
							np.array(letters_y_train).shape[0])
					print("Test Acc : ", np.sum((np.array(predicted_labels_test ==
															letters_y_test)) * 1), " / ",
							np.array(letters_y_test).shape[0])
					predicted_labels_train_next, optweights, optmargs = fit_sdca(
						include_eos=False, xtrain=X2_train, ytrain=y2_train,
						xtest=X2_test, ytest=y2_test, marginals=optmargs, args=args)

			else:
				data_for_clf = output_map.data.cpu().numpy().reshape((N, -1))
				if use_scaler:
					scaled_train = scaler.fit_transform(data_for_clf)
					if verbose:
						print("fit_transform finished")
				else:
					scaled_train = data_for_clf
				clf.fit(scaled_train, letters_y_train)
				if return_test_acc:
					test_acc = self.test_clf_sup(
						X_test, letters_y_test, clf, scaler, struct_model, X_test_words, y_test_words, lengths_test,
						use_scaler, use_cuda)
				if return_test_acc:
					self.test_accs.append(test_acc)
				if return_test_acc:
					print("Test acc: ", test_acc)

			if verbose:
				print("fit finished")
			for b in list_batches:
				def closure():
					XX = output_map[b * batch_size:(b + 1) * batch_size, :,
						 :].contiguous().view((batch_size, -1, p_size))
					data_for_clf = XX.data.cpu().numpy().reshape(batch_size, -1)
					if use_scaler:
						data_for_clf = scaler.transform(
							np.nan_to_num(data_for_clf))

					if use_fw:
						predicted_probas = marginals[b *
													 batch_size:(b + 1) * batch_size]
					elif use_sdca4crf:
						batch_predicted_labels = predicted_labels_train[b * batch_size:(
																								 b + 1) * batch_size]
						batch_true_labels = letters_y_train[b *
															batch_size:(b + 1) * batch_size]
						predicted_probas = turn_dense_to_one_matrix(
							batch_predicted_labels, self.num_labels, zero_prob=zero_prob)
					elif use_logistic_regression:
						predicted_probas = clf.predict_proba(data_for_clf)
						score_clf = clf.score(
							data_for_clf, letters_y_train[b * batch_size:(b + 1) * batch_size])
						if verbose:
							print("score of clf: ", score_clf)
					elif use_linear_svc:
						y = clf.predict(data_for_clf)
						predicted_probas = turn_dense_to_one_matrix(
							y, self.num_labels, zero_prob=zero_prob)
						score_clf = clf.score(
							data_for_clf, letters_y_train[b * batch_size:(b + 1) * batch_size])
						if verbose:
							print("score of clf: ", score_clf)
					else:
						raise (Exception("classifier not recognized."))
					with torch.no_grad():
						predicted_probas = Variable(torch.Tensor(
							predicted_probas), requires_grad=False)
					if use_cuda:
						predicted_probas = predicted_probas.cuda()
					loss = loss_func(
						predicted_probas, expected_output[b * batch_size:(b + 1) * batch_size])
					loss = Variable(loss.data, requires_grad=True)
					if not (use_fw or use_sdca4crf):
						self.train_acc += score_clf
					optimizer.zero_grad()
					loss.backward()
					return loss

				optimizer.step(closure)
			if t != 0:
				if not (use_fw or use_sdca4crf):
					self.train_acc /= batch_nb
					self.train_accs.append(self.train_acc)
					print('Epoch:', t)
				if use_fw:
					print("train acc : ", self.train_acc)
				else:
					print("train acc : ", self.train_acc)
			self.Kernel[k].eta = F.relu(self.Kernel[k].eta)
			if save_checkpoint: self.save_checkpoint(args.time_stamp)
		output_map = self.propagate_through_network(
			X, patches_given=False, use_cuda=use_cuda)
		print("train_accs = ", self.train_accs)
		if return_test_acc:
			print("test_accs = ", self.test_accs)
		if clf is not None and verbose:
			print("ssvm.objective_curve_ = ", clf.objective_curve_)
			print("ssvm.primal_objective_curve_ = ",
					clf.primal_objective_curve_)
		return output_map, clf, scaler, optimizer

	def propagate_through_network(self, X, patches_given=True, use_cuda=True, verbose=False):
		if self.embedding_layer is not None:
			output_map = self.embedding_layer(X.type(torch.LongTensor).cuda())
		elif self.linear_layer is not None:
			X = X.reshape((-1, 42))
			output_map = self.linear_layer(
				X.type(torch.FloatTensor).cuda()).reshape((-1, 200, 1))
		else:
			output_map = X
		for k in range(self.n_layers):
			if k == 0:
				if not patches_given:
					input_map = extract_patches_from_image(
						output_map, self.n_patches[0], use_cuda)
					if len(input_map.size()) == 4:
						input_map = input_map.view(
							input_map.size()[0], input_map.size()[1], -1)
				else:
					input_map = extract_patches_from_vector(
						output_map, self.n_patches[0])
			else:
				input_map = extract_patches_from_vector(
					output_map, self.n_patches[k])
			if verbose:
				print(k, input_map.size())
			self.Kernel[k].training_data_has_been_normalized = False
			self.Kernel[k].data_has_been_scaled = False
			if verbose:
				print('input map in layer ', k, ' has size ', input_map.size())
			output_map = self.Kernel[k].get_activation_map(
				k, X=input_map, norms=None, verbose=False, use_cuda=use_cuda)
		output_map = output_map.cpu()
		if use_cuda:
			output_map = output_map.cuda()
		return output_map

	def get_output(self):
		return self.Kernel[len(self.Kernel) - 1].get_activation_map(len(self.Kernel) - 1)

	def propagate_sdca(self, X_test, letters_y_test, clf, scaler, struct_model=None, X_test_words=None,
						 y_test_words=None, lengths_test=None, use_scaler=True, use_cuda=True, verbose=False):
		if scaler is None:
			use_scaler = False
		k = 0
		self.Kernel[k].training_data_has_been_normalized = False
		self.Kernel[k].training_data_has_been_scaled = False
		output_map = self.propagate_through_network(
			X_test, patches_given=False, use_cuda=use_cuda, verbose=verbose).permute(1, 0, 2)
		N = output_map.size()[0]
		if verbose:
			print("output_map.size() ", output_map.size())

		if use_scaler:
			output_map = output_map.data.cpu().numpy().reshape((N, -1))
			output_map = scaler.transform(np.nan_to_num(output_map))
			output_map = torch.from_numpy(output_map).cuda()
			if verbose:
				print("fit_transform finished")
		else:
			output_map = output_map.reshape((N, -1))
		return output_map

	def test_clf_sup(self, X_test, letters_y_test, clf, scaler, struct_model=None, X_test_words=None, y_test_words=None,
					 lengths_test=None, use_scaler=True, use_cuda=True, verbose=False):
		if X_test_words is None or y_test_words is None or lengths_test is None or struct_model is None:
			use_fw = False
		else:
			use_fw = True
		if scaler is None:
			use_scaler = False
		k = 0
		self.Kernel[k].training_data_has_been_normalized = False
		self.Kernel[k].training_data_has_been_scaled = False
		output_map = self.propagate_through_network(
			X_test, patches_given=False, use_cuda=use_cuda, verbose=verbose).permute(1, 0, 2)
		N = output_map.size()[0]
		if verbose:
			print("output_map.size() ", output_map.size())

		if use_scaler:
			output_map = output_map.data.cpu().numpy().reshape((N, -1))
			output_map = scaler.transform(np.nan_to_num(output_map))
			output_map = torch.from_numpy(output_map).cuda()
			if verbose:
				print("fit_transform finished")
		else:
			output_map = output_map.reshape((N, -1))

		if use_fw:
			X2_train, y2_train = reconstruct(
				output_map, letters_y_test, lengths_test, cuda=use_cuda)
			if check_nan(X2_train):
				X2_train, y2_train = remove_nans(X2_train, y2_train)
			score_clf = clf.score(X2_train, y2_train)
		else:
			if use_cuda:
				output_map = output_map.cpu()
			score_clf = clf.score(output_map, letters_y_test)
		if verbose:
			print("score : ", score_clf)
		return score_clf

	
	def save_checkpoint(self,time_stamp):
		if not os.path.exists("./pretrained/"):
			os.mkdir("./pretrained/")
		dictionary = {}
		attributes = inspect.getmembers(self, lambda a:not(inspect.isroutine(a)))
		tuples = [a for a in attributes if ( not((a[0].startswith('__') and a[0].endswith('__')) or	a[0].startswith('_') ))]
		for param in tuples:
			dictionary[param[0]] = param[1]
		torch.save(dictionary, './pretrained/CKN.pth')
	 
	def load_checkpoint(self):
		if not os.path.exists('./pretrained/CKN.pth'):
			raise Exception("./pretrained/CKN.pth not found.\n"+
            "Please load the pretrained model from the following link:\n" +
            "https://bit.ly/Struct-CKN")
		else:
			checkpoint = torch.load('./pretrained/CKN.pth')
			for k, v in checkpoint.items():
				setattr(self, k, v)
