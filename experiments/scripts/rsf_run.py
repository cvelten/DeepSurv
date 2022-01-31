import rpy2
import time
import logging
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
import rpy2.robjects as robjects
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import uuid
import h5py
import numpy as np
import pandas as pd
from collections import defaultdict
import argparse
import utils
import viz
import matplotlib
import sys
import os
sys.path.append("/DeepSurv/deepsurv")

# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')


# rpy2 lets you run R within Python
# Run a RandomSurvivalForest
rfSRC = importr('randomForestSRC')

# Converts pandas Dataframes into R Dataframes
pandas2ri.activate()


localtime = time.localtime()
TIMESTRING = time.strftime("%m%d%Y%M", localtime)

DURATION_COL = 'time'
EVENT_COL = 'status'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help='.h5 File containing the train/valid/test datasets')
    parser.add_argument('--treatment_idx', default=None, type=int,
                        help='(Optional) column index of treatment variable in dataset. If present, run treatment visualizations.')
    parser.add_argument('--num_trees', default=100, type=int, help='Hyper-parameter for number of trees to grow')
    parser.add_argument('--results_dir', default='/shared/results',
                        help='Directory to save resulting models and visualizations')
    return parser.parse_args()


def evaluate_model(model, dataset, bootstrap=False, trt_idx=None):
    def ci(model):
        def rsf_ci(**kwargs):
            data = utils.format_dataset_to_df(kwargs, DURATION_COL, EVENT_COL, trt_idx)
            pred_test = rfSRC.predict_rfsrc(model, data)
            return 1 - pred_test.rx('err.rate')[0][-1]
        return rsf_ci

    metrics = {}

    # Calculate c_index
    metrics['c_index'] = ci(model)(**dataset)
    if bootstrap:
        metrics['c_index_bootstrap'] = utils.bootstrap_metric(ci(model), dataset)

    return metrics


def rsf_treatment_rec(rfsc, test_tds):
    robjects.r('''
    tree.recommend <- function (treefit, x){ 
      # x is a data frame with a column labeled 'treat';
      #  the treat column is a binary "treatment" with values 0/1
      x.1 <- x
      x.0 <- x
      x.1$treat <- 1
      x.0$treat <- 0
      out.1 <- predict(treefit, x.1)$predicted
      out.0 <- predict(treefit, x.0)$predicted
      recommended <- max.col(cbind(out.1,out.0))
      recommended <- recommended - 1
    #   return(as.factor(recommended))
    }
    ''')
    tree_rec = robjects.r('tree.recommend')

    rsf_recs = np.array(tree_rec(rfsc, test_tds))
    return rsf_recs


def save_treatment_rec_visualizations(model, dataset, output_dir, trt_idx=None):
    tds = utils.format_dataset_to_df(dataset, DURATION_COL, EVENT_COL, trt_idx=trt_idx)

    rec_trt = rsf_treatment_rec(model, tds)

    # Reverse recommendations
    # print(rec_trt)
    # rec_trt = np.logical_not(rec_trt).astype(np.int32)

    rec_dict = utils.calculate_recs_and_antirecs(rec_trt, true_trt=trt_idx, dataset=dataset)

    output_file = os.path.join(output_dir, '_'.join(['rsf', TIMESTRING, 'rec_surv.pdf']))
    viz.plot_survival_curves(experiment_name='RSF', output_file=output_file, **rec_dict)

    np.save(f'{output_file.replace(".pdf","")}_rec_dict', rec_dict)


def rsf_treatment_risk(rfsc, test_tds):
    # robjects.r('''
    # tree.recommend <- function (treefit, x){
    #   # x is a data frame with a column labeled 'treat';
    #   #  the treat column is a binary "treatment" with values 0/1
    #   x.1 <- x
    #   x.0 <- x
    #   x.1$treat <- 1
    #   x.0$treat <- 0
    #   out.1 <- predict(treefit, x.1)$predicted
    #   out.0 <- predict(treefit, x.0)$predicted
    #   recommended <- max.col(cbind(out.1,out.0))
    #   recommended <- recommended - 1
    # #   return(as.factor(recommended))
    # }
    # ''')
    # tree_rec = robjects.r('tree.recommend')
    tree_risk = robjects.r('predict(treefit, tds')
    rsf_risk = np.array(tree_risk(rfsc, test_tds))
    return rsf_risk


def save_treatment_risk_data(model, dataset, norm_vals, output_dir,
                             trt_idx=0):
    tds = utils.format_dataset_to_df(dataset, DURATION_COL, EVENT_COL, trt_idx=trt_idx)

    rec_trt = rsf_treatment_risk(model, tds)

    output_file = os.path.join(output_dir, '_'.join(['rsf', TIMESTRING, 'rec_surv.csv']))

    colnames = ['cTxt'] + [f'c{i}' for i in range(dataset['x'].shape[1]-1)] + ['TxtRec']
    rec_df = pd.DataFrame(np.append(dataset['x'] * norm_vals['std'] +
                          norm_vals['mean'], rec_trt, axis=1), columns=colnames)

    output_file = os.path.join(output_dir, '_'.join(['rsf', TIMESTRING, 'rec_surv.csv']))
    print(output_file)
    rec_df.to_csv(output_file, index=False)


def save_model(model, output_dir):
    # TODO this currently breaks
    save_R = robjects.r('save')

    output_file = os.path.join(output_dir, "rsf_model" + str(uuid.uuid4()))
    save_R(rfsc, file=output_file)


if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    args = parse_args()
    print("Arguments:", args)

    # Load Dataset
    print("Loading datasets: " + args.dataset)
    datasets = utils.load_datasets(args.dataset)

    # Train CPH model
    print("Training RSF Model")
    train_df = utils.format_dataset_to_df(datasets['train'], DURATION_COL, EVENT_COL, trt_idx=args.treatment_idx)
    surv_f = robjects.r("as.formula(Surv(time,status) ~ .)")
    rfsc = rfSRC.rfsrc(surv_f, train_df, ntree=args.num_trees)

    print('Train/Valid C-Index:', 1 - rfsc.rx('err.rate')[0][-1])
    if 'valid' in datasets:
        metrics = evaluate_model(rfsc, datasets['valid'], trt_idx=args.treatment_idx)
        print("Valid metrics: " + str(metrics))

    if 'test' in datasets:
        metrics = evaluate_model(rfsc, datasets['test'], trt_idx=args.treatment_idx, bootstrap=True)
        print("Test metrics: " + str(metrics))

    if args.treatment_idx is not None:
        print("Calculating treatment recommendation survival curvs")
        save_treatment_rec_visualizations(rfsc, datasets['test'], output_dir=args.results_dir,
                                          trt_idx=args.treatment_idx)

    # Saving models doesn't work
    # if args.results_dir:
    #     save_model(rfsc, args.results_dir)

    exit(0)
