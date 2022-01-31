import h5py
import numpy as np
import pandas as pd
import os
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
    parser.add_argument('ifile_os')
    parser.add_argument('ifile_pfs')
    # parser.add_argument('-e', '--event_col', default='OSEvent')
    # parser.add_argument('-t', '--time_col', default='TTDy')
    # parser.add_argument('--txcol', type=str, default='SBRT')
    # parser.add_argument('--drop', help='drop columns', nargs='+', type=str)
    # parser.add_argument('--droprows', help='drop rows where [cols] have value --droprowsval', nargs='+', type=str)
    # parser.add_argument(
    #     '--droprowsval', help='value at which to drop the rows from --droprows, default 1', type=int, default=1)
    # parser.add_argument('--droprows2', help='drop rows where [cols] have value --droprowsval2', nargs='+', type=str)
    # parser.add_argument(
    #     '--droprowsval2', help='value at which to drop the rows from --droprows2, default 0', type=int, default=0)
    args = parser.parse_args()

    print(args)

    df = pd.read_csv(args.ifile_os)

    # print(df)

    drop_sbrtVS = ['Treatment', 'RFA', 'SBRT_OR_RFA']
    drop_rfaVS = ['Treatment', 'SBRT', 'SBRT_OR_RFA']
    drop_sbrtORrfa = ['Treatment', 'SBRT', 'RFA']

    #
    # THIS IS FOR OS FIRST

    frac = 0.5
    ds = {
        'SBRT_train': df[df.SBRT == 1].sample(frac=frac),
        'RFA_train': df[df.RFA == 1].sample(frac=frac),
        'NONE_train': df[df.SBRT_OR_RFA == 0].sample(frac=frac)
    }
    ds |= {
        'SBRT_test': df.loc[df[df.SBRT == 1].index.symmetric_difference(ds['SBRT_train'].index)],
        'RFA_test': df.loc[df[df.RFA == 1].index.symmetric_difference(ds['RFA_train'].index)],
        'NONE_test': df.loc[df[df.SBRT_OR_RFA == 0].index.symmetric_difference(ds['NONE_train'].index)],
    }

    df_sbrtVSnone = {
        'train': pd.concat([ds['SBRT_train'], ds['NONE_train']]).drop(columns=drop_sbrtVS),
        'test': pd.concat([ds['SBRT_test'], ds['NONE_test']]).drop(columns=drop_sbrtVS)
    }
    df_rfaVSnone = {
        'train': pd.concat([ds['RFA_train'], ds['NONE_train']]).drop(columns=drop_rfaVS),
        'test': pd.concat([ds['RFA_test'], ds['NONE_test']]).drop(columns=drop_rfaVS)
    }
    df_sbrtVSrfa = {
        'train': pd.concat([ds['SBRT_train'], ds['RFA_train']]).drop(columns=drop_sbrtVS),
        'test': pd.concat([ds['SBRT_test'], ds['RFA_test']]).drop(columns=drop_sbrtVS)
    }
    df_sbrtORrfa = {
        'train': pd.concat([ds['SBRT_train'], ds['RFA_train'], ds['NONE_train']]).drop(columns=drop_sbrtORrfa),
        'test': pd.concat([ds['SBRT_test'], ds['RFA_test'], ds['NONE_test']]).drop(columns=drop_sbrtORrfa)
    }

    ofile_os = os.path.join(os.path.dirname(args.ifile_os), 'liver_os_sbrtVSnone.hd5')
    dataframes_to_hd5(df_sbrtVSnone, ofile_os, 'OSEvent', 'TTDy')

    ofile_os = os.path.join(os.path.dirname(args.ifile_os), 'liver_os_rfaVSnone.hd5')
    dataframes_to_hd5(df_rfaVSnone, ofile_os, 'OSEvent', 'TTDy')

    ofile_os = os.path.join(os.path.dirname(args.ifile_os), 'liver_os_sbrtVSrfa.hd5')
    dataframes_to_hd5(df_sbrtVSrfa, ofile_os, 'OSEvent', 'TTDy')

    ofile_os = os.path.join(os.path.dirname(args.ifile_os), 'liver_os_sbrtORrfa.hd5')
    dataframes_to_hd5(df_sbrtORrfa, ofile_os, 'OSEvent', 'TTDy')

    #
    # USE INDICES FROM OS FOR PFS

    df_PFS = pd.read_csv(args.ifile_pfs)
    df_sbrtVSnone_pfs = {
        'train': df_PFS.loc[df_sbrtVSnone['train'].index].drop(columns=drop_sbrtVS),
        'test': df_PFS.loc[df_sbrtVSnone['test'].index].drop(columns=drop_sbrtVS)
    }
    df_rfaVSnone_pfs = {
        'train': df_PFS.loc[df_rfaVSnone['train'].index].drop(columns=drop_rfaVS),
        'test': df_PFS.loc[df_rfaVSnone['test'].index].drop(columns=drop_rfaVS)
    }
    df_sbrtVSrfa_pfs = {
        'train': df_PFS.loc[df_sbrtVSrfa['train'].index].drop(columns=drop_sbrtVS),
        'test': df_PFS.loc[df_sbrtVSrfa['test'].index].drop(columns=drop_sbrtVS)
    }
    df_sbrtORrfa_pfs = {
        'train': df_PFS.loc[df_sbrtORrfa['train'].index].drop(columns=drop_sbrtORrfa),
        'test': df_PFS.loc[df_sbrtORrfa['test'].index].drop(columns=drop_sbrtORrfa)
    }

    ofile_pfs = os.path.join(os.path.dirname(args.ifile_os), 'liver_pfs_sbrtVSnone.hd5')
    dataframes_to_hd5(df_sbrtVSnone_pfs, ofile_pfs, 'PFSEvent', 'TTPy')

    ofile_pfs = os.path.join(os.path.dirname(args.ifile_os), 'liver_pfs_rfaVSnone.hd5')
    dataframes_to_hd5(df_rfaVSnone_pfs, ofile_pfs, 'PFSEvent', 'TTPy')

    ofile_pfs = os.path.join(os.path.dirname(args.ifile_os), 'liver_pfs_sbrtVSrfa.hd5')
    dataframes_to_hd5(df_sbrtVSrfa_pfs, ofile_pfs, 'PFSEvent', 'TTPy')

    ofile_pfs = os.path.join(os.path.dirname(args.ifile_os), 'liver_pfs_sbrtORrfa.hd5')
    dataframes_to_hd5(df_sbrtORrfa_pfs, ofile_pfs, 'PFSEvent', 'TTPy')
