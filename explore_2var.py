import ess
import utils
import thinkplot
import thinkstats2
import pandas as pd

df = ess.read()
scales = utils.getCodeList('data/codeinfo/scales.csv')

def plotAll():
	n = len(scales)
	for i in range(n):
		for j in range(i+1, n):
			col0 = scales[i]
			col1 = scales[j]
			plotHexbin(col0, col1)

def plotHexbin(col0, col1, show=False):
	print (col0, col1)

	jit0 = thinkstats2.Jitter(df[col0], 0.5)
	jit1 = thinkstats2.Jitter(df[col1], 0.5)

	sub_df = pd.concat([jit0, jit1], axis=1)
	sub_df.dropna(subset=[col0, col1], inplace=True)

	xs = sub_df[col0]
	ys = sub_df[col1]

	# compute correlation
	corr = thinkstats2.Corr(xs, ys)
	switch = abs(corr)

	if switch > 0.25:
		# where to save
		folder = 'plots/twoVar/'
		if 0.25 < switch and switch <= 0.5:
			folder += 'medium/'
		elif 0.5 < switch and switch <= 1:
			folder += 'strong/'

		# hexbin plot
		thinkplot.HexBin(xs, ys)

		# fit and plot line
		inter, slope = thinkstats2.LeastSquares(xs, ys)
		fxs, fys = thinkstats2.FitLine(xs, inter, slope)
		thinkplot.Plot(fxs, fys)

		# config & show/save
		thinkplot.Config(xlabel=col0, ylabel=col1, title='Correlation: ' + str(round(corr,3)))
		if show:
			thinkplot.Show()
		else:
			thinkplot.Save(folder + col0 + '_' + col1, formats=['jpg'])