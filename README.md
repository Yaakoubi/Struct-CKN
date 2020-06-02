# Structured Convolutional Kernel Networks for Airline Crew Scheduling

This repository is the official implementation of Structured Convolutional Kernel Networks for Airline Crew Scheduling.

## Requirements

To install requirements:

```setup
pip install --upgrade -r requirements.txt
```


Use python2 or python<=3.6.10 to intall pystruct.
If you use python3.7, see [this link](https://stackoverflow.com/questions/24764549/upgrade-python-packages-from-requirements-txt-using-pip-command) for directions on how to install pystruct:


## Training


To train the model(s) in the paper, run this command:

```train
python main.py
```

To use the block-coordinate Frank-Wolfe predictor, run this command:

```train
python main.py --predictor fw
```


For a complete list of arguments, run this command:
```train
python main.py --help
```

## Evaluation

To evaluate our model on the OCR dataset, run:

```eval
python eval.py
```

## Pre-trained Models

The pretrained model is in the the folder pretrained.

## Results

Our model achieves the following performance on :

### [OCR dataset]

| Model name         | Test error    |
| ------------------ |-------------- |
| Struct-CKN (SDCA)  |     3.40%     |
| Struct-CKN (BCFW)  |     3.42%     |





