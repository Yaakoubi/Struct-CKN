import argparse
import matplotlib
import numpy as np
import torch
from pystruct.datasets import load_letters
from pystruct.learners import *
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from CKN.CKN import *
from ChainCRFLocal import ChainCRFLocal
from FrankWolfeLocal import FrankWolfeSSVMLocal
from utils_local import print_args

#matplotlib.use('Agg')

parser = argparse.ArgumentParser()
parser.add_argument("--numberEpochsFW", metavar='numberEpochsFW', type=int, default=10)
parser.add_argument("--numberEpochsCKN", metavar='numberEpochsCKN', type=int, default=100)
parser.add_argument("--regParBCFW", metavar='regParBCFW', type=float, default=1.)
parser.add_argument("--regParSDCA", metavar='regParSDCA', type=float, default=0.0001)
parser.add_argument("--lr", metavar='lr', type=float, default=.1)
parser.add_argument("--gpu", metavar='gpu', type=int, default=0)
parser.add_argument("--scaler", metavar='scaler', type=int, default=0)
parser.add_argument("--npass", metavar='npass', type=int, default=50)
parser.add_argument('--sampling-scheme', type=str, default='gap', help='Type of sampling.',
                    choices=["uniform", "importance", "gap", "gap+", "max", "safemax"])
parser.add_argument('--non-uniformity', type=float, default=0.8,
                    help='between 0 and 1. probability of sampling non-uniformly.')
parser.add_argument('--sampler-period', type=int, default=None,
                    help='if not None, period to do a full batch update of the duality gaps, '
                         'for the non-uniform sampling. Expressed as a number of epochs. '
                         'This whole epoch will be counted in the number of pass used by sdca')
parser.add_argument('--line-search', type=str, choices=['golden', 'newton'], default='golden',
                    help='Use scipy.optimize.minimize_scalar bounded golden section search, '
                         'or a custom safe bounded Newton-Raphson line search on the derivative.')
parser.add_argument('--init-previous-step-size', type=int, default=0,
                    help='Use the previous step size taken for a given sample to initialize the line search?')
parser.add_argument("--size-patch", metavar='size-patch', type=int, default=5)
parser.add_argument("--zero-prob", metavar='zero-prob', type=float, default=0.0001)
parser.add_argument("--use-warm-start", metavar='use-warm-start', type=int, default=1)
parser.add_argument("--predictor", type=str, default="sdca", choices=["sdca", "fw", "logistic", "linear"])
parser.add_argument("--numberEpochsUnsupCKN", type=int, default=100)
parser.add_argument("--benchmark", metavar='benchmark', type=int, default=0)
args = parser.parse_args()
print_args(args)
numberEpochsFW = args.numberEpochsFW
numberEpochsCKN = args.numberEpochsCKN
regParBCFW = args.regParBCFW
lr = args.lr
zero_prob = args.zero_prob
numberEpochsUnsupCKN = args.numberEpochsUnsupCKN

if args.scaler == 0:
    use_scaler = False
else:
    use_scaler = True

gpu = args.gpu

try:
    torch.cuda.set_device(args.gpu)
    utils_local.current_gpu = args.gpu
    utils_local.list_gpus = [args.gpu]
except:
    print("No available GPUs")

if args.benchmark == 0:
    use_1_fold_for_train = False
else:
    use_1_fold_for_train = True

letters = load_letters()
X, y, folds = letters['data'], letters['labels'], letters['folds']
X, y = np.array(X), np.array(y)
if use_1_fold_for_train:
    X_train_words, X_test_words = X[folds == 9][:100], X[folds != 9]
    y_train_words, y_test_words = y[folds == 9][:100], y[folds != 9]
else:
    X_train_words, X_test_words = X[folds != 9], X[folds == 9]
    y_train_words, y_test_words = y[folds != 9], y[folds == 9]

letters_x_train, letters_y_train, lengths_train = words_to_letters(X_train_words, y_train_words)
letters_x_test, letters_y_test, lengths_test = words_to_letters(X_test_words, y_test_words)
data = torch.Tensor(letters_x_train)
n_d, p_dim = data.size()
data = data.view(n_d, 16, 8)

use_fw, use_sdca4crf, use_logistic_regression, use_linear_svc = False, False, False, False

if args.predictor == "fw":
    use_fw = True
elif args.predictor == "sdca":
    use_sdca4crf = True
elif args.predictor == "logistic":
    use_logistic_regression = True
elif args.predictor == "linear":
    use_linear_svc = True

assert (use_logistic_regression + use_fw + use_linear_svc + use_sdca4crf == 1)

use_clfs = use_logistic_regression, use_fw, use_linear_svc, use_sdca4crf
n_filters = 200
size_patch = args.size_patch

CKN = CKN(n_components=[n_filters],
          n_layers=1,
          iter_max=numberEpochsUnsupCKN,
          n_patches=[size_patch],
          subsampling_factors=[2],
          batch_size=[5],
          dim1=16,
          dim2=8,
          lr=lr)

if use_logistic_regression:
    clf = LogisticRegression(C=1.0,
                             penalty="l2",
                             fit_intercept=False,
                             solver="saga",
                             multi_class="multinomial",
                             warm_start=True,
                             max_iter=10,
                             n_jobs=-1,
                             verbose=1)
    struct_model = None
elif use_fw:
    struct_model = ChainCRFLocal(inference_method="max-product")
    clf = FrankWolfeSSVMLocal(model=struct_model,
                              c=regParBCFW,
                              max_iter=numberEpochsFW,
                              verbose=1,
                              show_loss_every=1,
                              check_dual_every=1,
                              tol=-np.inf)
elif use_linear_svc:
    clf = LinearSVC(random_state=0,
                    tol=1e-5,
                    dual=False,
                    verbose=1,
                    max_iter=100)
    struct_model = None
elif use_sdca4crf:
    clf = None
    struct_model = None
else:
    raise Exception
X_test = torch.Tensor(letters_x_test)
Y_test = torch.Tensor(letters_y_test)
n_d, p_dim = X_test.size()
X_test = X_test.view(n_d, 16, 8)
if args.scaler == 0:
    scaler = None
else:
    if args.scaler == 1:
        scaler = preprocessing.StandardScaler()
    elif args.scaler == 2:
        scaler = preprocessing.Normalizer(norm="l2")
    elif args.scaler == 3:
        scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    elif args.scaler == 4:
        scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    else:
        scaler = None
output_map, clf, scaler, optimizer = CKN.train_network_sup(data, letters_y_train, clf, scaler, struct_model,
                                                           X_train_words, y_train_words, lengths_train,
                                                           init=True, verbose=False,
                                                           numberEpochsCKN=numberEpochsCKN, use_scaler=use_scaler,
                                                           numberEpochsFW=numberEpochsFW,
                                                           X_test=X_test, letters_y_test=letters_y_test,
                                                           X_test_words=X_test_words, y_test_words=y_test_words,
                                                           lengths_test=lengths_test, use_clfs=use_clfs,
                                                           zero_prob=zero_prob, args=args)
if use_fw:
    CKN.test_clf_sup(X_test, letters_y_test, clf, scaler, struct_model,
                     X_test_words, y_test_words, lengths_test, use_scaler=use_scaler)
    CKN.test_clf_sup(data, letters_y_train, clf, scaler, struct_model,
                     X_train_words, y_train_words, lengths_train, use_scaler=use_scaler)
elif use_logistic_regression or use_linear_svc:
    CKN.test_clf_sup(X_test, letters_y_test, clf, scaler)
    CKN.test_clf_sup(data, letters_y_train, clf, scaler)
elif use_sdca4crf:
    pass
