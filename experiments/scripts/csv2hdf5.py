import h5py
import numpy as np
import pandas as pd
from argparse import ArgumentParser


def dataframe_to_deepsurv_ds(df, event_col='Event', time_col='Time'):
    # Extract the event and time columns as numpy arrays
    e = df[event_col].values.astype(np.int32)
    t = df[time_col].values.astype(np.float32)

    # Extract the patient's covariates as a numpy array
    x_df = df.drop([event_col, time_col], axis=1)
    x = x_df.values.astype(np.float32)

    # Return the deep surv dataframe
    return {
        'x': x,
        'e': e,
        't': t
    }


def dataframes_to_hd5(df, ofile, event_col, time_col):
    with h5py.File(ofile, 'w') as h:
        for k in df:
            ds = dataframe_to_deepsurv_ds(df[k], event_col, time_col)
            group = h.create_group(k)
            group.create_dataset('x', data=ds['x'])
            group.create_dataset('e', data=ds['e'])
            group.create_dataset('t', data=ds['t'])


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('ifile')
    parser.add_argument('ofile')
    parser.add_argument('-e', '--event_col', default='OSEvent')
    parser.add_argument('-t', '--time_col', default='TTDy')
    parser.add_argument('--txcol', type=str, default='SBRT')
    parser.add_argument('--drop', help='drop columns', nargs='+', type=str)
    parser.add_argument('--droprows', help='drop rows where [cols] have value --droprowsval', nargs='+', type=str)
    parser.add_argument(
        '--droprowsval', help='value at which to drop the rows from --droprows, default 1', type=int, default=1)
    parser.add_argument('--droprows2', help='drop rows where [cols] have value --droprowsval2', nargs='+', type=str)
    parser.add_argument(
        '--droprowsval2', help='value at which to drop the rows from --droprows2, default 0', type=int, default=0)
    args = parser.parse_args()

    print(args)

    df = pd.read_csv(args.ifile)

    if not args.drop is None:
        if not args.droprows is None:
            drop_idx = df.where((df.loc[:, args.droprows] == args.droprowsval).any(axis='columns')).dropna().index
            df.drop(drop_idx, axis='rows', inplace=True)
        if not args.droprows2 is None:
            drop_idx = df.where((df.loc[:, args.droprows2] == args.droprowsval2).any(axis='columns')).dropna().index
            df.drop(drop_idx, axis='rows', inplace=True)
        df.drop(args.drop, axis='columns', inplace=True)

    # print(df)

    frac = 0.5

    df_train_treated = df[df[args.txcol] == 1].sample(frac=frac)
    df_train_untreated = df[df[args.txcol] == 0].sample(frac=frac)

    df_test_treated = df.loc[df[df[args.txcol] == 1].index.symmetric_difference(df_train_treated.index)]
    df_test_untreated = df.loc[df[df[args.txcol] == 0].index.symmetric_difference(df_train_untreated.index)]

    df_train = pd.concat([df_train_treated, df_train_untreated])
    df_test = pd.concat([df_test_treated, df_test_untreated])

    # dataframes = {'train': df_train, 'test': df_test, 'valid': df_test}
    # dataframes = {'train': df, 'test': df, 'valid': df}
    dataframes = {'train': df_train, 'test': df_test}

    # print(df_train)
    # print(df_test)

    dataframes_to_hd5(dataframes, args.ofile, args.event_col, args.time_col)
