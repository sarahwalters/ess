import numpy as np
import pandas as pd
import csv

def test(csvpath='data/ess.csv'):
	return pd.read_csv(csvpath)

def read(csvpath='data/ess.csv'):
	# read df from ESS csv
	df = pd.read_csv(csvpath)

	# replace numerical NaN encodings with actual NaNs
	with open('data/codeinfo/clean.csv') as csvfile:
		reader = csv.reader(csvfile, delimiter=':')
		for row in reader:
			# config
			toReplace = row[0].split(',') # values which should be NaN
			cols = row[1].split(',') # applicable columns

			# perform the replace
			for col in cols:
				df[col].replace(toReplace, np.nan, inplace=True)

	# drop any unnamed cols
	for col in df.columns.values:
		if 'Unnamed' in col:
			df.drop(col, axis=1, inplace=True)

	return df