import numpy as np
import torch
from scipy.io import savemat

from utils_local import *


def centering(X, n_channels):
    size_channel = X.shape[1] / n_channels
    Y = np.zeros(X.shape)
    for ii in range(n_channels):
        XX = X[(ii - 1) * size_channel + 1:ii * size_channel, :]
        Y[(ii - 1) * size_channel + 1:ii * size_channel, :] = XX - np.mean(XX)
    return Y


def contrast_normalize(X):
    nrm = np.sqrt(sum(X ** 2, 0))

    def trim(x): return np.max([x, 0.00001])

    trim_v = np.vectorize(trim)
    nrm = trim_v(nrm)
    return nrm.dot(X)


def contrast_normalize_median(x):
    nrm = np.sqrt(sum(x ** 2, 0))
    med = np.max([np.median(nrm), 0.00001])
    return 1.0 / med * x


def get_zeromap(inpt, type_zerolayer):
    if not dtype(inpt) == float:
        inpt = double(inpt) / 255
    sx = inpt.size[0]
    sy = inpt.size[1]
    if 3 * sx == sy:
        if type_zerolayer == 3:
            out = inpt[:, sx + 1:2 * sx]
        else:
            out = inpt.reshape([sx, sx, 3])
    else:
        out = inpt
    return out


def rel_distance_patch(size_patch, use_cuda=False):
    model_patch = np.zeros((size_patch, size_patch))
    cntr_x, cntr_y = [size_patch // 2 + size_patch %
                      2 - 1, size_patch // 2 + size_patch % 2 - 1]
    for ii in range(size_patch):
        for jj in range(size_patch):
            model_patch[ii, jj] = (ii - cntr_x) ** 2 + (jj - cntr_y) ** 2
    if use_cuda:
        return torch.Tensor(model_patch).cuda()
    return torch.Tensor(model_patch)


def extract_patches_from_image(data, size_patch, zero_padding=True, use_cuda=False):
    size_image = data.size()[1:]
    a = size_patch + size_image[0]
    b = size_patch + size_image[1]
    if use_cuda:
        padded_image = torch.Tensor(np.zeros((data.size()[0], a, b))).cuda()
    else:
        padded_image = torch.Tensor(np.zeros((data.size()[0], a, b)))
    padded_image[:, size_patch // 2:(size_patch // 2) + size_image[0],
    size_patch // 2:(size_patch // 2) + size_image[1]] = data
    nx, ny = padded_image.size()[1:]
    if use_cuda:
        patches = torch.stack([padded_image[:, ii:ii + size_patch, jj:jj + size_patch].cpu()
                               for ii in np.arange(0, nx - size_patch, 1) for jj in
                               np.arange(0, ny - size_patch, 1)]).cuda()
    else:
        patches = torch.stack([padded_image[:, ii:ii + size_patch, jj:jj + size_patch].cpu()
                               for ii in np.arange(0, nx - size_patch, 1) for jj in np.arange(0, ny - size_patch, 1)])
    return patches


def extract_patches_from_vector(X, size_patch, zero_padding=True, use_cuda=False):
    nb_locations = X.size()[0]
    x_dim = int(np.sqrt(X.size()[0]))
    nb_data_points = X.size()[1]
    D = X.size()[2]
    size_image = [x_dim, x_dim]
    a = 2 * np.max([size_patch // 2, 1]) + size_image[0]
    b = 2 * np.max([size_patch // 2, 1]) + size_image[1]
    padded_image = np.zeros((a, b))
    padded_image[np.max([size_patch // 2, 1]):np.max([size_patch // 2, 1]) + size_image[0],
    np.max([size_patch // 2, 1]):(np.max([size_patch // 2, 1])) + size_image[1]] = 1
    nx, ny = padded_image.shape
    test = torch.Tensor([create_bw_mask(size_image, size_patch, ii, jj, use_cuda=use_cuda)[
                         np.max([size_patch // 2, 1]):np.max([size_patch // 2, 1]) + size_image[0],
                         np.max([size_patch // 2, 1]):(np.max([size_patch // 2, 1])) + size_image[1]]
                         for ii in np.arange(0, nx - 2 * np.max([size_patch // 2, 1]))
                         for jj in np.arange(0, ny - 2 * np.max([size_patch // 2, 1]))])
    test = test.view(test.size()[0], -1)
    test2 = torch.stack([torch.cat([X[u, :, :].t() for u in np.where(test[j, :].numpy() != 0)[0]]
                                   + [torch.Tensor(np.zeros((D, nb_data_points))).cuda()] * (
                                               size_patch ** 2 - len(np.where(test[j, :].numpy() != 0)[0]))) for j in
                         range(nb_locations)])

    return test2.permute(0, 2, 1)


def extract_selected_patches(data, id_patch, size_patch, zero_padding=True):
    selected_patches = torch.Tensor(len(id_patch), 2, size_patch ** 2)
    patches = extract_patches(data, size_patch, zero_padding=True)
    for j in range(len(id_patch)):
        while torch.sum(patches[id_patch[j][1], id_patch[j][0], :]) == 0 and torch.sum(
                patches[id_patch[j][2], id_patch[j][0], :]) == 0:
            nx, ny = np.random.choice(range(data.size()[1] * data.size()[2]), 2)
            id_patch[j] = [id_patch[j][0], nx, ny]
        selected_patches[j, 0, :] = patches[id_patch[j][1], id_patch[j][0], :]
        selected_patches[j, 1, :] = patches[id_patch[j][2], id_patch[j][0], :]
    return selected_patches


def extract_patch_mask(N, size_image, size_patch, beta=1, zero_padding=True, use_cuda=False):
    a = 2 * np.max([size_patch // 2, 1]) + size_image[0]
    b = 2 * np.max([size_patch // 2, 1]) + size_image[1]
    padded_image = np.zeros((a, b))
    padded_image[np.max([size_patch // 2, 1]):np.max([size_patch // 2, 1]) + size_image[0],
    np.max([size_patch // 2, 1]):(np.max([size_patch // 2, 1])) + size_image[1]] = 1
    nx, ny = padded_image.shape
    patches = torch.Tensor(np.array([(padded_image * create_distance_mask(size_image, size_patch, ii, jj, beta=beta,
                                                                          use_cuda=use_cuda))[
                                     np.max([size_patch // 2, 1]):np.max([size_patch // 2, 1]) + size_image[0],
                                     np.max([size_patch // 2, 1]):(np.max([size_patch // 2, 1])) + size_image[1]] for ii
                                     in np.arange(0, nx - 2 * np.max([size_patch // 2, 1])) for jj in
                                     np.arange(0, ny - 2 * np.max([size_patch // 2, 1]))]))
    return patches


def create_distance_mask(size_image, size_patch, ii, jj, beta=1, use_cuda=False):
    image = np.zeros((size_image[0] + 2 * np.max([size_patch // 2, 1]),
                      size_image[1] + 2 * np.max([size_patch // 2, 1])))
    if size_patch is not None:
        mask = rel_distance_patch(size_patch, use_cuda=use_cuda).cpu().numpy()
    else:
        mask = rel_distance_patch(size_patch, use_cuda=use_cuda).cpu().numpy()
    image[ii:ii + size_patch, jj:jj + size_patch] = np.exp(-1.0 / beta ** 2 * mask)
    return image


def create_bw_mask(size_image, size_patch, ii, jj, use_cuda=False):
    image = np.zeros((size_image[0] + 2 * np.max([size_patch // 2, 1]),
                      size_image[1] + 2 * np.max([size_patch // 2, 1])))
    image[ii:ii + size_patch, jj:jj + size_patch] = 1
    return image


def normalize_output(input_map, epsilon=0.0001, center_data=False, center=None, verbose=False):
    n_p, n_d, p = input_map.size()
    norm2 = torch.norm(input_map, p=2, dim=2)
    norm2[norm2 < epsilon] = 1
    norm2 = norm2.view(n_p, n_d, 1).expand_as(input_map)

    input_map /= norm2

    if verbose:
        print('dim norm1 ', norm2.size())
        print('max norm1', torch.max(norm2))
        print('min norm1', torch.min(norm2[norm2 > 0]))
    if center_data:
        if center is None:
            center = torch.mean(input_map.view(n_p * n_d, p),
                                0).view(1, 1, p).expand(n_p, n_d, p)
            input_map = input_map - center
        norm2 = torch.norm(input_map, p=2, dim=2).detach()
        norm2[norm2 < epsilon] = 1
        input_map = input_map.div(norm2.view(n_p, n_d, 1).expand_as(input_map))
        if verbose:
            print('dim norm2 ', norm2.size())
            print('max norm2 ', torch.max(norm2))
            print('min norm2 ', torch.min(norm2[norm2 > 0]))
    return input_map


def reconstruct(x, y, lengths, cuda=True):
    words_x = []
    words_y = []
    index = 0
    for length in lengths:
        if cuda:
            new_word_x = [x[i].cpu().numpy()
                          for i in range(index, index + length)]
        else:
            new_word_x = [x[i] for i in range(index, index + length)]
        new_word_y = [y[i] for i in range(index, index + length)]

        words_x.append(np.array(new_word_x))
        words_y.append(np.array(new_word_y))
        index = index + length
    return np.array(words_x), np.array(words_y)


def write_mat_file(x, y, lengths, train=True, write_file=True):
    if write_file:
        new_data_train_path = "SDCA/data/ocr_train_struct_ckn.mat"
        new_data_test_path = "SDCA/data/ocr_test_struct_ckn.mat"
    x_for_sdca = []
    y_for_sdca = []
    y_local = y.copy()
    y_local = synchonize_y(new_y=y_local, min_y=1)
    index = 0
    try:
        x = x.cpu().numpy()
    except:
        pass
    num_letters = x.shape[0]
    num_features = x[0].shape[0]
    shape0 = num_letters
    for length in lengths:
        for i in range(index, index + length):
            x_for_sdca.append(np.array(x[i]).reshape(num_features))
            y_for_sdca.append(y_local[i])
        x_for_sdca.append(list(np.zeros(num_features)))
        y_for_sdca.append(0)
        index = index + length
        shape0 += 1
    x_for_sdca = np.array(x_for_sdca).reshape((shape0, num_features))
    y_for_sdca = np.array(y_for_sdca).reshape((shape0))
    x_for_sdca = np.nan_to_num(x_for_sdca)
    if write_file:
        if train:
            savemat(new_data_train_path, {'X': x_for_sdca, 'y': y_for_sdca})
        else:
            savemat(new_data_test_path, {'X': x_for_sdca, 'y': y_for_sdca})
    return x_for_sdca, y_for_sdca
