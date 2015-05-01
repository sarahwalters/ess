import ess
import thinkstats2
import thinkplot
import numpy as np
import utils

# SETUP
df = ess.read()
skip = utils.getCodeList('data/codeinfo/skip.csv')
forceHist = utils.getCodeList('data/codeinfo/forceHist.csv')
forcePmf = utils.getCodeList('data/codeinfo/centeredValuesScale.csv')


# MAIN FUNCTIONS
def plotAll():
	for col in df.columns.values:
		if col not in skip:
			print col
			if df[col].dtype == np.float64:
				plotCdf(col, forcePmf=(col in forcePmf))
				if col in forceHist:
					plotHist(col)
			elif df[col].dtype == np.int64:
				plotCdf(col)
				plotHist(col)


def plotCdf(col, forcePmf=False, show=False):
	cdf = thinkstats2.Cdf(df[col], label=col)
	thinkplot.Config(xlabel=col + ', centered', ylabel='CDF', title='Cumulative distribution of ' + col, legend=False)
	thinkplot.Cdf(cdf)
	if show:
		thinkplot.Show()
	else:
		thinkplot.Save('plots/oneVar/' + col + '_cdf', formats=['jpg'])

	if forcePmf:
		# create histogram with 0.1-wide bins
		binStart = int(10*df[col].min())/10.0
		binStop = int(10*df[col].max()+1)/10.0
		numBins = (binStop-binStart)/0.1 + 1
		bins = np.linspace(binStart, binStop, numBins)
		digitized = np.digitize(df[col], bins)

		# create another (sparser) linspace for labeling the axis
		intStart = int(binStart)
		intStop = int(binStop)+1 # round up
		intNumBins = intStop-intStart+1
		sparseBins = np.linspace(intStart, intStop, intNumBins)

		# bin the column
		hist = {}
		for dig in digitized:
			key = bins[dig-1]
			if key in hist:
				hist[key] += 1
			else:
				hist[key] = 1

		# plot
		thinkplot.Clf()
		thinkplot.Config(xticks=np.linspace(-5,5,11), xlim=(-5,5), xlabel=col + ', centered', ylabel='Frequency', title='Distribution of ' + col, legend=False)
		pmf = thinkstats2.Hist(hist, label=col)
		thinkplot.Hist(pmf)
		if show:
			thinkplot.Show()
		else:
			thinkplot.Save('plots/oneVar/' + col + '_pmf', formats=['jpg'])


def plotHist(col, show=False):
	hist = thinkstats2.Hist(df[col], label=col)
	thinkplot.Config(xlabel=col, ylabel='Frequency', title='Distribution of ' + col, legend=False)
	thinkplot.Hist(hist)
	if show:
		thinkplot.Show()
	thinkplot.Save('plots/oneVar/' + col + '_hist', formats=['jpg'])


# UNUSED
def eisced(show=False):
	cdf0 = thinkstats2.Cdf(df.eisced, label='respondent')
	cdf1 = thinkstats2.Cdf(df.eiscedp, label='partner')
	cdf2 = thinkstats2.Cdf(df.eiscedf, label='father')
	cdf3 = thinkstats2.Cdf(df.eiscedm, label='mother')

	thinkplot.Cdfs([cdf0, cdf1, cdf2, cdf3])
	if show:
		thinkplot.Show()
	else:
		thinkplot.Save('plots/oneVar/eisced_all_cdf', formats=['jpg'])


def rlgdnm(show=False):
	cdf0 = thinkstats2.Hist(df.rlgdnm, label='current')
	cdf1 = thinkstats2.Hist(df.rlgdnme, label='past')

	thinkplot.Hist(cdf0, width=0.4, align='right')
	thinkplot.Hist(cdf1, width=0.4, align='left')
	if show:
		thinkplot.Show()
	else:
		thinkplot.Save('plots/oneVar/rlgdnm_all_hist', formats=['jpg'])