# VOC 2007 / 2012 classification
This code trains and evaluates a 20-way binary classifier on the PASCAL VOC dataset. It uses the stardard sigmoid cross entropy loss and evaluates mAP.

## Setup
Clone this repo and also get my caffe future branch https://github.com/philkr/caffe/tree/future (I use a few custom layers, mainly HDF5 related) and compile it.

Download the VOC 2007 dataset from http://host.robots.ox.ac.uk/pascal/VOC/voc2007/.

Setup your own `user_config.py`, see `user_config.example.py` for an example. You'll need to setup `CAFFE_DIR` to point to the caffe future branch, and `VOC_DIR` to point to the Pascal VOC directory.

Now everything should be ready.

## Running the evaluation
The basic evaluation is run through
```bash
python3 src/train_cls.py path/to/your.prototxt path/to/your.caffemodel
```

If you want to automatically train all variations `fc8 only`, `fc6-fc8` and `conv1-fc8` for your model you can use the `make_matrix.sh` script. It relies on the `wait_for_gpu` script, which tries to wait for a free gpu before running a job (this is a very crude solution and you probably want to use something like `slurm` in production).
```bash
bash make_matrix.sh experiments/ALL_MODELS experiments/ALL_PARAMS output/directory
```
This script will run every model described in `experiments/ALL_MODELS` with every parameter setting described in `experiments/ALL_PARAMS`. You can then plot the results using the `make_table.py` script.
```bash
python3 make_table.py output/directory
```
NOTE: `make_table.py` currently doesn't to a perfect job in figuring out what parameters and models were used. If somebody want to fix this be my guest.


## Overfitting and 'tuning' the classifier
I have found the default settings to work in all the evaluations and comparisons I have performed. I'm sure there are certain learning rate and weight decay settings that will work better for some models. I however do not have the time to find them. The current parameters are tuned such that alexnet trained on imagenet works ok mAP of 80%.
If you want to tune the parameters for your own model, be my guest, but please do it also for all baseline models too!

## Results
I'll publish a up to date list of the top performing methods here, as soon as I get a chance.
