import numpy as np
import pandas as pd
import csv
import utils

values = utils.getCodeList('data/codeinfo/values.csv')

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

	# drop responses where any of the values are nans
	df.dropna(subset=values, inplace=True)
	valuesdf = df[values]

	# recode value scale -> now 6 = very like me and 1 = not like me at all
	cur = [i for i in range(1,7)]
	rev = list(reversed(cur))
	rf = df[values].replace(to_replace=cur, value=rev)

	# inplace doesn't seem to be working above, so here's a workaround
	for value in values:
		df[value] = rf[value]

	# center the values
	# http://www.europeansocialsurvey.org/docs/methodology/ESS1_human_values_scale.pdf
	valuesdf = df[values] # just the human values columns
	df['mrat'] = valuesdf.sum(axis=1)/21 
	for value in values:
		df[value+'_c'] = df[value] - df.mrat	

	return df