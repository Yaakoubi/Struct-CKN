import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import RobustScaler
from sklearn.utils import check_random_state
from torch.autograd import Variable
import utils_local
from image_processing_utils import *


class Cell:

    def __init__(self, dim1=28, dim2=28, n_components=50, iter_max=5, random_state=None, size_patch=5,
                 n_patches_per_graph=10, lr=0.1, batch_size=100, subsampling_factor=2, spacing=1, type_optim='adamax',
                 use_cuda=True):
        if use_cuda:
            self.eta = Variable(torch.Tensor(
                1.0 / n_components * (np.ones((n_components,)))).cuda(), requires_grad=True)
        else:
            self.eta = Variable(torch.Tensor(
                1.0 / n_components * (np.ones((n_components,)))), requires_grad=True)
        if use_cuda:
            self.W = Variable(torch.Tensor().cuda())
        else:
            self.W = Variable(torch.Tensor())
        self.sigma = None
        self.spacing = spacing
        self.training_data_has_been_normalized = False
        self.data_has_been_scaled = False
        self.training_data_has_been_scaled = False
        self.norms = None
        self.n_components = n_components
        self.random_state = random_state
        self.iter_max = iter_max
        self.print_lag = 15
        self.lr = lr
        self.all_patches = None
        self.training_data = None
        self.size_patch = size_patch
        self.patches = None
        self.distances = None
        self.subsampling_factor = subsampling_factor
        self.batch_size = batch_size
        self.output = None
        self.training_output = None
        self.pca = None
        self.standardize = None
        self.n_patches_per_graph = n_patches_per_graph
        self.type_optim = type_optim
        self.dim1 = dim1
        self.dim2 = dim2
        self.optimizer = None
        self.km = None

    def select_training_patches(self, param_x, patches_given=True, verbose=False, use_cuda=True):
        id_patch = []
        n_patches_per_graph = np.min(
            [self.n_patches_per_graph, param_x.size()[0]])
        size_patch = self.size_patch
        # STEP 0: extract patches from input map
        if patches_given == False:
            patches = extract_patches_from_image(
                param_x, size_patch, zero_padding=True, use_cuda=use_cuda)

            if len(patches.size()) == 4:
                patches = patches.view(patches.size()[0], patches.size()[
                    1], patches.size()[2] * patches.size()[3])
        else:
            if size_patch == 1:
                patches = param_x
            else:
                patches = extract_patches_from_vector(
                    param_x, size_patch, zero_padding=True, use_cuda=use_cuda)
        del param_x
        import gc
        gc.collect()
        print("patches extracted")
        self.all_patches = normalize_output(patches, verbose=verbose)
        size_patches = patches.size()
        del patches
        self.norms = torch.sqrt(torch.mean(self.all_patches ** 2, 2))
        self.norms = torch.clamp(self.norms, 0.0, float(
            np.percentile(self.norms.cpu().detach().numpy(), 95)))
        if use_cuda:
            self.norms = self.norms.cuda()
        self.training_data_has_been_normalized = True
        n_p, n_d, p_dim = size_patches
        standard = RobustScaler(quantile_range=(5.0, 95.0))
        if standard is not None:
            try:
                x_tilde2 = standard.fit_transform(
                    self.all_patches.view(-1, p_dim).cpu().detach().numpy())
            except:
                x_tilde2 = standard.fit_transform(
                    self.all_patches.reshape(-1, p_dim).cpu().detach().numpy())
        else:
            x_tilde2 = self.all_patches.view(-1, p_dim).cpu().detach().numpy()
        if use_cuda:
            self.all_patches = torch.Tensor(
                x_tilde2).cuda().contiguous().view(n_p, n_d, -1)
        else:
            self.all_patches = torch.Tensor(
                x_tilde2).contiguous().view(n_p, n_d, -1)
        self.standardize = standard
        self.training_data_has_been_scaled = True
        print('Training patches have been standardized')
        for i in range(size_patches[1]):
            for j in range(n_patches_per_graph):
                nx, ny = np.random.choice(range(size_patches[0]), 2)
                id_patch += [[i, nx, ny]]
        if len(size_patches) == 4:
            selected_patches = torch.Tensor(
                len(id_patch), 2, size_patches[2] * size_patches[3])
            if use_cuda:
                selected_patches = selected_patches.cuda()
        else:
            selected_patches = torch.Tensor(len(id_patch), 2, size_patches[2])
            if use_cuda:
                selected_patches = selected_patches.cuda()
        it_j = {}
        not_found = 0
        for j in range(len(id_patch)):
            it_j[j] = 0
            while torch.sum(torch.abs(self.all_patches[id_patch[j][1], id_patch[j][0], :])) == 0 and torch.sum(
                    torch.abs(self.all_patches[id_patch[j][2], id_patch[j][0], :])) == 0 and (it_j[j] < 100):
                nx, ny = np.random.choice(range(size_patches[0]), 2)
                id_patch[j] = [id_patch[j][0], nx, ny]
                # print(len(id_patch),"-",size_patches[0],"-",nx,"-",ny,"-",it_j[j])
                it_j[j] += 1
            if it_j[j] == 100:
                not_found += 1
                if not_found % 10000 == 0:
                    print(j, " -- Not found : ",
                          not_found, " / ", len(id_patch))
            selected_patches[j, 0, :] = self.all_patches[id_patch[j]
                                                         [1], id_patch[j][0], :]
            selected_patches[j, 1, :] = self.all_patches[id_patch[j]
                                                         [2], id_patch[j][0], :]
        self.training_data = selected_patches
        return selected_patches

    def chunks(self, l, n):
        for i in range(0, len(l), n):
            yield l[i:i + n]

    def fit(self, X=None, init=True, patches_given=True, use_cuda=True, retrain_graph=False):
        rnd = check_random_state(self.random_state)
        d = self.n_components
        if init is True:
            self.init_W(X, patches_given=patches_given)
        if use_cuda:
            x_input = Variable(self.training_data.cuda(), requires_grad=False)
        else:
            x_input = Variable(self.training_data, requires_grad=False)
        n = x_input.size()[0]
        loss_func = nn.MSELoss()
        if self.optimizer is None:
            if self.type_optim == 'adamax':
                self.optimizer = optim.Adamax([self.W, self.eta], lr=self.lr)
            elif self.type_optim == 'LBFGS':
                self.optimizer = optim.LBFGS(
                    [self.W, self.eta], lr=self.lr, max_iter=4000)
            else:
                self.optimizer = optim.Adamax([self.W, self.eta], lr=self.lr)
        batch_nb = n // self.batch_size
        batch_size = self.batch_size
        p_size = x_input.size()[2]
        self.training_loss = []
        for t in range(self.iter_max):
            overall_loss = 0
            for b in range(batch_nb):
                def closure():
                    expected_output = torch.exp(
                        torch.div(torch.sum((x_input[:, 0, :] - x_input[:, 1, :]) ** 2, 1), -2 * self.sigma))
                    XX = x_input[b * batch_size:(b + 1) * batch_size, 0, :].contiguous().view(
                        (1, batch_size, p_size)).expand(d, batch_size, p_size)
                    YY = x_input[b * batch_size:(b + 1) * batch_size, 1, :].contiguous().view(
                        (1, batch_size, p_size)).expand(d, batch_size, p_size)
                    output = (XX - self.W.view(d, 1, p_size).expand(d, batch_size, p_size)
                              ) ** 2 + (YY - self.W.view(d, 1, p_size).expand(d, batch_size, p_size)) ** 2
                    output = torch.div(torch.sum(output, 2), -self.sigma)
                    if use_cuda:
                        output = output.cuda()
                    output2 = torch.exp(output)
                    output2 = torch.matmul(F.relu(self.eta), output2)
                    loss = loss_func(output2, expected_output[b * batch_size:(
                        b + 1) * batch_size]) + torch.sum(
                        (F.relu(self.eta) - self.eta) ** 2)
                    self.optimizer.zero_grad()
                    loss.backward(retain_graph=retrain_graph)
                    return loss

                self.optimizer.step(closure)

    def init_W(self, X, patches_given=True, use_cuda=True, use_batch_mode_select=False, init_unsup=True):
        carroussel = 0
        if use_batch_mode_select:
            batch_size = 512
            training_data = []
            all_patches = []
            batches = self.chunks(
                range(0, X.size()[0]), batch_size)  # X.size()[0]
            for batch in batches:
                torch.cuda.set_device(utils_local.list_gpus[carroussel])
                batch_training_data = self.select_training_patches(
                    X[batch], patches_given=patches_given, use_cuda=use_cuda, verbose=False)
                training_data.append(batch_training_data)
                all_patches.append(self.all_patches)
                carroussel += 1
                carroussel %= len(utils_local.list_gpus)
            torch.cuda.set_device(utils_local.current_gpu)
            self.training_data = torch.cat(training_data, dim=0)
            self.all_patches = torch.cat(all_patches, dim=1)
            if self.training_data_has_been_normalized == False:
                print('Error training data has not been normalized!!!!!')
                self.training_data = normalize_output(
                    self.training_data, verbose=False)
                self.training_data_has_been_normalized = True
            n_p, n_d, p_dim = self.all_patches.size()
            n_p_train, _, p_dim_train = self.training_data.size()
            X_tilde = torch.cat(
                (self.training_data[:, 0, :], self.training_data[:, 1, :]), dim=0)
            distances2 = torch.sum(
                (self.training_data[:, 0, :] - self.training_data[:, 1, :]) ** 2, dim=1)
            if self.sigma is None:
                # set to be quantile
                # compute the distance between patches
                self.sigma = np.max(
                    [np.percentile(distances2.cpu().numpy(), 10.0), 0.0001])
            if init_unsup and self.km is None:
                self.km = MiniBatchKMeans(
                    n_clusters=self.n_components, max_no_improvement=None, n_init=100, reassignment_ratio=1,
                    max_iter=1000)
            inds = range(int(X_tilde.size()[0]))
            np.random.shuffle(list(inds))
            self.km.partial_fit(X_tilde.cpu().numpy())
            if use_cuda:
                self.W = Variable(torch.Tensor(
                    self.km.cluster_centers_).cuda(), requires_grad=True)
            else:
                self.W = Variable(torch.Tensor(
                    self.km.cluster_centers_), requires_grad=True)

        else:
            self.training_data = self.select_training_patches(
                X, patches_given=patches_given, use_cuda=use_cuda)
            if self.training_data_has_been_normalized is False:
                print('Error training data has not been normalized!!!!!')
                self.training_data = normalize_output(
                    self.training_data, verbose=False)
                self.training_data_has_been_normalized = True
            n_p, n_d, p_dim = self.all_patches.size()
            n_p_train, _, p_dim_train = self.training_data.size()
            X_tilde = torch.cat(
                (self.training_data[:, 0, :], self.training_data[:, 1, :]), dim=0)
            distances2 = torch.sum(
                (self.training_data[:, 0, :] - self.training_data[:, 1, :]) ** 2, dim=1)
            if self.sigma is None:
                self.sigma = np.max(
                    [np.percentile(distances2.cpu().numpy(), 10.0), 0.0001])
            if init_unsup and self.km is None:
                self.km = MiniBatchKMeans(
                    n_clusters=self.n_components, max_no_improvement=None, n_init=100, reassignment_ratio=1,
                    max_iter=1000)
            inds = range(int(X_tilde.size()[0]))
            np.random.shuffle(list(inds))
            self.km.partial_fit(X_tilde.cpu().numpy())
            if use_cuda:
                self.W = Variable(torch.Tensor(
                    self.km.cluster_centers_).cuda(), requires_grad=True)
            else:
                self.W = Variable(torch.Tensor(
                    self.km.cluster_centers_), requires_grad=True)

    def get_activation_map(self, k, X=None, norms=None, verbose=False, use_cuda=True):
        p_shape = self.size_patch  # p_shape: dimension of the new patches
        if X is None:
            input_map = self.all_patches  # input map: should check that it is of dim
            n_p, n_d, p_dim = input_map.size()
            self.training_data_has_been_normalized = False
            if self.training_data_has_been_normalized is False:
                self.norms = torch.sqrt(torch.mean(input_map ** 2, dim=2))
                input_map = normalize_output(input_map)
                self.training_data_has_been_normalized = True
            if self.training_data_has_been_scaled is False:
                if self.standardize is not None:
                    input_map = self.standardize.transform(
                        input_map.contiguous().view(-1, input_map.size()[2]).cpu().numpy())
                    input_map = torch.Tensor(input_map).view(n_p, n_d, -1)
                    if use_cuda:
                        input_map = input_map.cuda()
                    n_p, n_d, p_dim = input_map.size()
                if self.pca is not None:
                    input_map = self.pca.transform(
                        input_map.contiguous().view(-1, input_map.size()[2]).numpy())
                    input_map = torch.Tensor(
                        input_map).cuda().view(n_p, n_d, -1)
                    n_p, n_d, p_dim = input_map.size()
                self.training_data_has_been_scaled = True

        else:
            input_map = X.cpu()
            if norms is not None:
                self.norms = norms
            if self.training_data_has_been_normalized is False:
                self.norms = torch.sqrt(torch.mean(input_map ** 2, dim=2))
                input_map = normalize_output(input_map)
                self.training_data_has_been_normalized = True
            n_p, n_d, p_dim = X.size()
            if self.data_has_been_scaled is False:
                if self.standardize is not None:
                    input_map = self.standardize.transform(
                        input_map.contiguous().view(-1, input_map.size()[2]).detach().cpu().numpy())
                    if use_cuda:
                        input_map = torch.Tensor(
                            input_map).cuda().view(n_p, n_d, -1)
                    else:
                        input_map = torch.Tensor(input_map).view(n_p, n_d, -1)
                    n_p, n_d, p_dim = input_map.size()
                if self.pca is not None:
                    input_map = torch.Tensor(
                        input_map).cuda().view(n_p, n_d, -1)
                    n_p, n_d, p_dim = input_map.size()
                self.data_has_been_scaled = True
        if len(input_map.size()) == 4:
            input_map = input_map.view(
                input_map.size()[0], input_map.size()[1], -1).contiguous()
        n_p, n_d, p_dim = input_map.size()
        spacing = self.spacing
        gamma = self.subsampling_factor
        beta = gamma * spacing
        batch_size = self.batch_size
        D = self.n_components
        sigma = self.sigma
        dim = int(np.sqrt(input_map.size()[0]))
        size_patch = p_shape
        if k == 0:
            mpatches = extract_patch_mask(batch_size, [
                self.dim1, self.dim2], size_patch, beta=beta, zero_padding=True, use_cuda=use_cuda)
            if use_cuda:
                mpatches = mpatches.view(-1, self.dim1 * self.dim2).cuda()
            else:
                mpatches = mpatches.view(-1, self.dim1 * self.dim2)

        else:
            mpatches = extract_patch_mask(
                batch_size, [dim, dim], size_patch, beta=beta, zero_padding=True, use_cuda=use_cuda)
            mpatches = mpatches.view(-1, dim * dim).cuda()
        selected_pixels = [i * dim + j for j in np.arange(
            0, dim, self.subsampling_factor) for i in np.arange(0, dim, self.subsampling_factor)]
        if use_cuda:
            output_map = torch.Tensor(len(selected_pixels), n_d, D).cuda()
        else:
            output_map = torch.Tensor(len(selected_pixels), n_d, D)

        for b in range(n_d // batch_size):
            XX = input_map[:, b * batch_size:(b + 1) * batch_size,
                           :].contiguous().view(-1, p_dim).cpu()
            tempo2 = XX.unsqueeze(1).expand(n_p * batch_size, D, p_dim) - \
                self.W.data.cpu().unsqueeze(0).expand(n_p * batch_size, D, p_dim)
            tot = torch.exp(
                torch.div(torch.sum((tempo2.cpu() ** 2), dim=2), -sigma))
            del tempo2
            if verbose:
                print('tot', torch.max(tot))
            w = torch.sqrt(self.eta.unsqueeze(
                0).data.expand_as(tot)).cpu() * tot
            del tot
            if verbose:
                temp = w
            Reg = self.norms[:, b * batch_size:(b + 1) * batch_size].contiguous().view(
                n_p * batch_size, 1).expand_as(w).cpu()
            if verbose:
                print('max Reg', torch.max(Reg))
            if use_cuda:
                zeta = (Reg * w).cuda()
            else:
                zeta = (Reg * w)
            if verbose:
                print('zeta', zeta.size())
            if verbose:
                print('nb 2 of null is ', torch.sum(zeta != zeta))
            del Reg
            zeta = zeta.view(n_p, batch_size, D)  # gather tensor
            if verbose:
                print(mpatches.size(), zeta.view(n_p, batch_size * D).size())
            test = torch.matmul(mpatches, zeta.view(n_p, batch_size * D))
            if verbose:
                print('test', zeta.size())
            del zeta
            out = test.view(n_p, batch_size, D)
            del test
            if use_cuda:
                output_map[:, torch.from_numpy(np.arange(int(b * batch_size), int((b + 1) * batch_size))).cuda(
                ), :] = out[torch.from_numpy(np.asarray(selected_pixels)).cuda(), :, :]
            else:
                output_map[:, torch.from_numpy(np.arange(int(b * batch_size), int((b + 1) * batch_size))).type(
                    torch.LongTensor), :] = out[torch.from_numpy(np.asarray(selected_pixels)).type(torch.LongTensor), :,
                                                :].cpu()
        self.output = output_map
        del output_map
        del input_map
        self.output *= np.sqrt(2.0 / math.pi)
        return self.output
