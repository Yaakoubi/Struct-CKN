import time

import numpy as np

import SDCA.sdca4crf.monitor as monitor
from SDCA.sdca4crf.line_search import LineSearch
from SDCA.sdca4crf.parameters.initializer import compute_primal_direction, initialize
from SDCA.sdca4crf.sampler_wrap import SamplerWrap
from SDCA.sdca4crf.utils import infer_probas


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def sdca(trainset, testset=None, args=None):
    """Update alpha and weights with the stochastic dual coordinate ascent algorithm to fit
    the model to the trainset points x and the labels y.

    Unless warm_start is used, the initial point is a concatenation of the smoothed empirical
    distributions.

    :param trainset: training set in the LabeledSequenceData format
    :param testset: testing set in the LabeledSequenceData format. We test the prediction after
    every epochs.

    :return: tuple, optimal value of the (weights, marginals)
    """

    # INITIALIZE : the dual and primal variables
    marginals, weights, ground_truth_centroid = \
        initialize(args.warm_start, trainset, args.regularization)

    # OBJECTIVES : primal objective, dual objective and duality gaps.

    # fake estimate of the duality gaps
    gaps_array = 100 * np.ones(len(trainset))
    previous_step_sizes = .5 * np.ones(len(trainset))

    monitor_all_objectives = monitor.MonitorAllObjectives(args.regularization, weights, marginals,
                                                          ground_truth_centroid, trainset, testset,
                                                          args.use_tensorboard)
    monitor_dual_objective = monitor.MonitorDualObjective(
        args.regularization, weights, marginals)
    monitor_gap_estimate = monitor.MonitorDualityGapEstimate(gaps_array)
    monitor_sparsity = monitor.MonitorSparsity()
    monitor_speed = monitor.MonitorSpeed()

    del ground_truth_centroid, testset  # clean up the namespace

    # non-uniform sampling
    sampler = SamplerWrap(args.sampling_scheme, args.non_uniformity,
                          gaps_array, trainset, args.regularization)
    step_size_array = np.empty((len(trainset) * args.npass, 2))
    start_sdca = time.time()
    time_pass_on_line_search = 0
    try:

        ##################################################################################
        # MAIN LOOP
        ##################################################################################
        for step in range(1, len(trainset) * args.npass + 1):

            # SAMPLING
            i = sampler.sample()
            alpha_i = marginals[i]
            points_sequence_i = trainset.get_points_sequence(i)

            # MARGINALIZATION ORACLE
            beta_i, _ = weights.infer_probabilities(points_sequence_i)
            # ASCENT DIRECTION (primal to dual)
            log_dual_direction, signs_dual_direction = beta_i.logsubtractexp(
                alpha_i)
            dual_direction = log_dual_direction.exp().multiply(signs_dual_direction)
            # assert dual_direction.is_consistent() #2008
            # assert dual_direction.is_density(0) #2008

            # EXPECTATION of FEATURES (dual to primal)
            primal_direction = compute_primal_direction(points_sequence_i, dual_direction,
                                                        trainset.is_sparse, len(
                                                            trainset),
                                                        args.regularization)

            # NECESSARY VALUES
            individual_gap = alpha_i.kullback_leibler(beta_i)
            monitor_gap_estimate.update(i, individual_gap)

            primaldir_squared_norm = primal_direction.squared_norm()
            weights_dot_primaldir = weights.inner_product(primal_direction)

            # LINE SEARCH : find the optimal step size or use a fixed one
            # Update the dual objective monitor as well
            line_search_start = time.time()
            line_search = LineSearch(weights_dot_primaldir, primaldir_squared_norm,
                                     log_dual_direction,
                                     alpha_i, beta_i, individual_gap,
                                     args, previous_step_sizes[i])

            if args.fixed_step_size is not None:
                optimal_step_size = args.fixed_step_size
            else:
                optimal_step_size = line_search.run()
                previous_step_sizes[i] = optimal_step_size

            line_search_end = time.time()
            time_pass_on_line_search += line_search_end - line_search_start
            step_size_array[step - 1] = np.array([i, optimal_step_size])

            # UPDATE : the primal and dual parameters, and the sampler
            marginals[i] = alpha_i.convex_combination(
                beta_i, optimal_step_size)
            weights += primal_direction * optimal_step_size

            sampler.update(i, individual_gap, np.sqrt(
                primaldir_squared_norm), optimal_step_size)

            monitor_dual_objective.update(i, marginals[i].entropy(),
                                          line_search.norm_update(optimal_step_size))

            # ANNEX
            if args.use_tensorboard and step % (len(trainset) // 5) == 0:
                monitor_dual_objective.log_tensorboard(step)
                monitor_speed.update(step)
                monitor_speed.log_tensorboard()
                if args.fixed_step_size is None:
                    line_search.log_tensorboard(step)

            if step % len(trainset) == 0:
                # OBJECTIVES : after every epochs, compute the duality gap
                # Update the sampler if necessary (count_time==True)
                count_time = args.sampler_period is not None and step % (
                    args.sampler_period * len(trainset)) == 0

                gaps_array = monitor_all_objectives.full_batch_update(
                    weights, marginals, step, count_time)

                monitor_gap_estimate.log_tensorboard(
                    monitor_all_objectives.duality_gap, step)

                assert monitor.are_consistent(
                    monitor_dual_objective, monitor_all_objectives)

                if count_time:
                    # count the full batch in the number of steps
                    step += len(trainset)
                    monitor_gap_estimate = monitor.MonitorDualityGapEstimate(
                        gaps_array)
                    sampler.full_update(gaps_array)

                monitor_sparsity.log_tensorboard(weights, step)
                # Don't count the monitoring time in the speed.
                monitor_speed.update(step)
                end_sdca = time.time()
                ratio_time_line_search = time_pass_on_line_search / \
                    (end_sdca - start_sdca)
                monitor_speed.log_time_spent_on_line_search(
                    ratio_time_line_search, step)
                time_pass_on_line_search = 0
                start_sdca = time.time()

            # STOP condition
            if monitor_gap_estimate.get_value() < args.precision:
                monitor_all_objectives.full_batch_update(weights, marginals, step,
                                                         count_time=False)
                break
            '''
            if i == 500 :
                print("*****")
                predicted_labels_train = infer_probas(trainset, weights)
                print(len(predicted_labels_train))
                print(trainset.labels[:30])
                print(predicted_labels_train[:30])
            '''
    finally:  # save results no matter what.
        monitor_all_objectives.save_results(args.logdir)
        if args.save == 'all':
            step_size_array.dump(args.logdir + '.npy')
            np.savez(args.logdir+"/predicted_labels")
    return weights, marginals
