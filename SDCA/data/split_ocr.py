import numpy as np
from scipy.io import loadmat, savemat

ocrdat = loadmat("data/ocr.mat", squeeze_me=True)

folds = ocrdat['fold']
print(folds[:40])

# remove the -1 from end of sequences
for i, fold in enumerate(folds):
    if fold == -1:
        folds[i] = folds[i - 1]
print(folds[:40])

# Modify here if you want to change the split structure.
# for now the train set is folds 1-9 and the test set is fold 0
mask = (folds == 0)

train = {}
test = {}
for tag in ['X', 'y']:
    train[tag] = ocrdat[tag][np.logical_not(mask)]
    test[tag] = ocrdat[tag][mask]

savemat('data/ocr_train', train, do_compression=True)
savemat('data/ocr_test', test, do_compression=True)
