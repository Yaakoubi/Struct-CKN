import numpy as np

from sdca4crf.arguments import get_args, get_information_string, init_logdir
from sdca4crf.get_datasets import get_datasets
from sdca4crf.sdca import sdca
np.set_printoptions(threshold=2000)

if __name__ == '__main__':
    args = get_args()

    # load datasets
    train_data, test_data = get_datasets(args)
    '''
    print(train_data.starts[0])
    print(train_data.ends[0])
    print(train_data.starts[1])
    print(train_data.ends[1]) 
    print(train_data.points[:20])
    print(train_data.labels[:20])
    '''
    if args.regularization is None:
        args.regularization = 1 / args.train_size

    infostring = get_information_string(args, train_data, test_data)
    print(infostring)
    init_logdir(args, infostring)

    # run optimization
    optweights, optmargs = sdca(
        trainset=train_data, testset=test_data, args=args)

    print("Optimization finished.")
    if args.save == 'all':
        #np.save(args.logdir + '/optweights.npy', optweights.to_array())
        np.save(args.logdir + '/opttransition.npy', optweights.transition)
        np.save(args.logdir + '/optbias.npy', optweights.bias)
        np.save(args.logdir + '/optemission.npy', optweights.emission)
        marginals_dic = {
            'marginal' + str(i): margs.binary for i, margs in enumerate(optmargs)}
        np.savez_compressed(args.logdir + '/optmarginals.npy', **marginals_dic)
        # optweights.
