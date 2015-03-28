import ess
import thinkstats2
import thinkplot
import numpy as np
import csv

# SETUP
def getCodeList(csvpath):
	res = []
	with open(csvpath) as csvfile:
		reader = csv.reader(csvfile, delimiter=',')
		for row in reader:
			res += row
	return res

df = ess.read()
skip = getCodeList('data/codeinfo/skip.csv')
forceHist = getCodeList('data/codeinfo/forceHist.csv')


# MAIN FUNCTIONS
def plotCdf(col, show=False):
	cdf = thinkstats2.Cdf(df[col], label=col)
	thinkplot.Cdf(cdf)
	if show:
		thinkplot.Show()
	thinkplot.Save('plots/oneVar/' + col + '_cdf', formats=['jpg'])


def plotHist(col, show=False):
	hist = thinkstats2.Hist(df[col], label=col)
	thinkplot.Hist(hist)
	if show:
		thinkplot.Show()
	thinkplot.Save('plots/oneVar/' + col + '_hist', formats=['jpg'])

def plotAll():
	for col in df.columns.values:
		if col not in skip:
			print col
			if df[col].dtype == np.float64:
				plotCdf(col)
				if col in forceHist:
					plotHist(col)
			elif df[col].dtype == np.int64:
				plotCdf(col)
				plotHist(col)

def eisced():
	cdf0 = thinkstats2.Cdf(df.eisced, label='respondent')
	cdf1 = thinkstats2.Cdf(df.eiscedp, label='partner')
	cdf2 = thinkstats2.Cdf(df.eiscedf, label='father')
	cdf3 = thinkstats2.Cdf(df.eiscedm, label='mother')

	thinkplot.Cdfs([cdf0, cdf1, cdf2, cdf3])
	thinkplot.Show()
	thinkplot.Save('explore/eisced_all_cdf', formats=['jpg'])