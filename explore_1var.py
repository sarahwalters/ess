import ess
import thinkstats2
import thinkplot
import numpy as np
import utils

# SETUP
df = ess.read()
skip = utils.getCodeList('data/codeinfo/skip.csv')
forceHist = utils.getCodeList('data/codeinfo/forceHist.csv')


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
	#thinkplot.Show()
	thinkplot.Save('plots/oneVar/eisced_all_cdf', formats=['jpg'])

def rlgdnm():
	cdf0 = thinkstats2.Hist(df.rlgdnm, label='current')
	cdf1 = thinkstats2.Hist(df.rlgdnme, label='past')

	thinkplot.Hist(cdf0, width=0.4, align='right')
	thinkplot.Hist(cdf1, width=0.4, align='left')
	#thinkplot.Show()
	thinkplot.Save('plots/oneVar/rlgdnm_all_hist', formats=['jpg'])