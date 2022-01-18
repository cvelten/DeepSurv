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
	parser.add_argument('-e', '--event_col', default='Survival')
	parser.add_argument('-t', '--time_col', default='TTDy')
	args = parser.parse_args()

	df = pd.read_csv(args.ifile)

	frac = 0.5
	df_train_treated = df[df.SBRT == 1].sample(frac=frac)
	df_train_untreated = df[df.SBRT == 0].sample(frac=frac)

	df_test_treated = df.loc[df[df.SBRT == 1].index.symmetric_difference(df_train_treated.index)]
	df_test_untreated = df.loc[df[df.SBRT == 0].index.symmetric_difference(df_train_untreated.index)]

	df_train = pd.concat([df_train_treated, df_train_untreated])
	df_test = pd.concat([df_test_treated, df_test_untreated])

	# dataframes = {'train': df_train, 'test': df_test, 'valid': df_test}
	# dataframes = {'train': df, 'test': df, 'valid': df}
	dataframes = {'train': df_train, 'test': df_test}

	dataframes_to_hd5(dataframes, args.ofile, args.event_col, args.time_col)
