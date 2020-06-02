# SDCA4CRF

Stochastic dual coordinate ascent for training conditional random fields. 
Visit the [project webpage](https://remilepriol.github.io/research/sdca4crf.html) for more details.

<img src="doc/ner_primal_calls.png" alt="drawing" width="512px"/>

### Depends

Python 3.6, Numpy, Scipy, Matplotlib, [tensorboard_logger](https://github.com/TeamHG-Memex/tensorboard_logger).

### Usage

Call `main.py` with the desired arguments.
The full list of arguments is specified in `sdca4crf/arguments.py`.
The main training loop is in `sdca4crf/sdca.py`.
A typical use case is:

`python main.py --dataset ner --non-uniformity 0.8 --sampling-scheme gap`

You can use tensorboard to visualize training. Training curves and other results are also saved into pickle files at the end of training.

Four pre-processed datasets are available under `data/`.
To use another dataset, you should extract features and numberize them.

The folders named `experiments` contain a bunch of scripts used for the paper.
