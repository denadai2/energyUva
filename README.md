energyUva
=========

This project wants to show the work I've done during my Master's internship in Amsterdam. The focus of this work was finding outliers in the gas consumption of some buildings.

The results are described in this [page](http://www.marcodena.it/blog/detecting-anomalies-with-neural-newtorks/).

## Instructions

* Clone [this](https://github.com/denadai2/pylearn2/tree/UVA) pylearn2 branch, which includes the MSLE cost function
* Install the cloned pylearn2

## Commands
To train your model you can use

    python [PYLEARN_DIR]/pylearn2/scripts/train.py NN_static_MLSE.yaml 

To predict

    python [PYLEARN_DIR]/pylearn2/scripts/mlp/predict_csv.py best.pkl test.csv predict.csv -P 'regression'

