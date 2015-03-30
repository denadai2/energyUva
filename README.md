energyUva
=========

This project wants to show the work I've done during my Master's internship in Amsterdam. The focus of this work was finding outliers in the gas consumption of some buildings.

The results are described in this [page](http://www.marcodena.it/blog/detecting-anomalies-with-neural-newtorks/).

## Instructions

* Clone [this](https://github.com/lisa-lab/pylearn2/tree/08a4e4ab9d80eb1a7a83d91e64fd8d512d3d7e7c) pylearn2, which uses the exact pylearn2 version we used.
* Install the cloned pylearn2

## Commands
To train your model you can use

    python [PYLEARN_DIR]/pylearn2/scripts/train.py NN-cross_v2.yaml

To predict

    python [PYLEARN_DIR]/pylearn2/scripts/mlp/predict_csv.py bests/bestv2-0.pkl test.csv predict.csv -P 'regression'

