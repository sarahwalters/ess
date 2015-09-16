import ess
import thinkstats2
import thinkplot
import numpy as np
import random
import math
import utils
import pandas as pd

df = ess.read()
unaffiliated = df[df.rlgblg == 2] # unaffiliated
scales = utils.getCodeList('data/codeinfo/scales.csv')
values = utils.getCodeList('data/codeinfo/values.csv')
centeredValues = utils.getCodeList('data/codeinfo/centeredValues.csv')


def plotAll():
	rlgLookup = [(1, 'Catholic'), (2, 'Protestant'), (5, 'Jewish'), (6, 'Islamic')]
	rlgdfs = [df[df.rlgdnm==tup[0]] for tup in rlgLookup]
	rlglabels = [tup[1] for tup in rlgLookup]

	# all unweighted plots
	# for col in scales:
	# 	# unaffiliated means
	# 	estimateMean(col)

	# 	# diff in means compared to unaffiliated, for each religion
	# 	diffMeansEsts = []
	# 	for i in range(len(rlgdfs)):
	# 		est = estimateDiffMeans(col, rlgdfs[i], rlglabels[i])
	# 		diffMeansEsts.append(est)

	# 	# plot all diff means together
	# 	diffMeansPlot(col, diffMeansEsts)

	# weighted plots for centeredValues only (takes forever)
	for col in centeredValues:
		# diff in means compared to unaffiliated, for each religion
		diffMeansEsts = []
		for i in range(len(rlgdfs)):
			est = estimateDiffMeans(col, rlgdfs[i], rlglabels[i], m=100, weight=True)
			diffMeansEsts.append(est)

		# plot all diff means together
		diffMeansPlot(col, diffMeansEsts, weight=True)

def debug():
	rlgLookup = [(1, 'Catholic'), (2, 'Protestant'), (5, 'Jewish'), (6, 'Islamic')]
	rlgdfs = [df[df.rlgdnm==tup[0]] for tup in rlgLookup]
	rlglabels = [tup[1] for tup in rlgLookup]

	# all unweighted plots
	for col in ['imptrad_c']:
		# unaffiliated means
		estimateMean(col)

		# diff in means compared to unaffiliated, for each religion
		diffMeansEsts = []
		for i in range(len(rlgdfs)):
			est = estimateDiffMeans(col, rlgdfs[i], rlglabels[i])
			diffMeansEsts.append(est)

		# plot all diff means together
		diffMeansPlot(col, diffMeansEsts, show=True)


def estimateMean(col, n=10, m=1000, show=False):
	# get data
	data = unaffiliated[col] # length: ~2e6
	data.dropna(inplace=True)
	data.name = 'data'

	weights = unaffiliated.pweight * unaffiliated.dweight
	weights.name = 'weights'

	unaff = pd.concat([data, weights], axis=1)
	unaff.dropna(subset=['data'], inplace=True)

	# make labels
	xlabel='Estimated mean of ' + col + ' among unaffiliated respondents'
	savepath = 'plots/estimation/mean/'+ col

	# run "experiments"
	est = ResampledEstimate([unaff], m, n, xlabel, savepath)
	est.plot(show=show)

	return est


def estimateDiffMeans(col, rlgdf, rlglabel, n=10, m=1000, show=False, weight=False):
	# get data
	dataReligion = rlgdf[col]
	dataReligion.name = 'data'
	dataUnaffiliated = unaffiliated[col]	
	dataUnaffiliated.name = 'data'

	weightsReligion = rlgdf.pweight * rlgdf.dweight
	weightsReligion.name = 'weights'
	weightsUnaffiliated = unaffiliated.pweight * unaffiliated.dweight
	weightsUnaffiliated.name = 'weights'

	rlg = pd.concat([dataReligion, weightsReligion], axis=1)
	unaff = pd.concat([dataUnaffiliated, weightsUnaffiliated], axis=1)

	rlg.dropna(subset=['data'], inplace=True)
	unaff.dropna(subset=['data'], inplace=True)

	# make labels
	xlabel = 'Estimated ' + rlglabel + '/unaffiliated diff in means of ' + col
	savepath = 'plots/estimation/diffMeans/' + col + '_' + rlglabel

	# run "experiments"
	est = ResampledEstimate([rlg, unaff], m, n, xlabel, savepath, rlg=rlglabel, weight=weight)
	est.plot(show=show)

	return est


class ResampledEstimate:
	def __init__(self, data, m, n, xlabel, savepath, rlg='All', weight=False):
		# labels
		self.rlg = rlg
		self.xlabel = xlabel
		self.savepath = savepath
		if weight:
			self.savepath += '_weight'

		# estimators
		self.data = data

		# do estimation
		self.estimated = []
		if len(data) == 1: # single mean
			for _ in range(m):
				if weight:
					xs = thinkstats2.ResampleRowsWeighted(self.data[0])
					mean = np.mean(xs.data)
				else:
					xs = thinkstats2.Resample(self.data[0].data)
					mean = np.mean(xs)
				self.estimated.append(mean)

		elif len(data) == 2: # diff means
			self.diffMeans = []
			for i in range(m):
				if weight:
					print 'loop ' + str(i) + ' of ' + str(m)
					xs_rlg = thinkstats2.ResampleRowsWeighted(self.data[0])
					xs_unaff = thinkstats2.ResampleRowsWeighted(self.data[1])
					diffMeans = np.mean(xs_rlg.data) - np.mean(xs_unaff.data)
				else:
					xs_rlg = thinkstats2.Resample(self.data[0].data)
					xs_unaff = thinkstats2.Resample(self.data[1].data)
					diffMeans = np.mean(xs_rlg) - np.mean(xs_unaff)
				self.estimated.append(diffMeans)

		# make sampling cdf
		self.samplingCdf = thinkstats2.Cdf(self.estimated, label='dummy label')
		self.confInt = self.samplingCdf.ConfidenceInterval()

		# for displaying nicely
		#self.printMean = 'Mean ' + str(round(mu,3))
		self.printConfInt = 'CI [' + str(round(self.confInt[0], 3)) + ', ' + str(round(self.confInt[1], 3)) + ']'
		#self.printSE = 'SE ' + str(round(RMSE(self.means, mu), 3))

		# give cdf proper label
		self.title = self.printConfInt
		self.samplingCdf.label = self.rlg + ': ' + self.title

	
	def plot(self, show=False):
		thinkplot.Clf()

		thinkplot.Config(xlabel=self.xlabel, ylabel='CDF', title=self.title, legend=False)

		VertLine(self.confInt[0])
		VertLine(self.confInt[1])
		#VertLine(self.mu, color='0.2')

		thinkplot.Cdf(self.samplingCdf)

		if show:
			thinkplot.Show()
		else:
			thinkplot.Save(self.savepath, formats=['jpg'])


def diffMeansPlot(col, diffMeansEsts, show=False, weight=False):
	thinkplot.Clf()

	if col in values:
		thinkplot.Config(xlim=(-1.4,1.4)) # keep all value axes consistent
	elif col in centeredValues:
		thinkplot.Config(xticks=np.linspace(-1.4, 1.4, 15), xlim=(-1.5,1.5))

	thinkplot.Config(xlabel='Estimated religious/unaffiliated diff in means of ' + col, ylabel='CDF', legend=True, loc='lower center', bbox_to_anchor=(0.5, -0.6), htscale=0.825)
	
	samplingCdfs = [est.samplingCdf for est in diffMeansEsts]
	thinkplot.Cdfs(samplingCdfs)

	if show:
		thinkplot.Show()
	else:
		path = 'plots/estimation/diffMeansAll/' + col
		if weight:
			path += '_weight'
		thinkplot.Save(path, formats=['jpg'])


# from ThinkStats2
def RMSE(estimates, actual):
    """Computes the root mean squared error of a sequence of estimates.

    estimate: sequence of numbers
    actual: actual value

    returns: float RMSE
    """
    e2 = [(estimate-actual)**2 for estimate in estimates]
    mse = np.mean(e2)
    return math.sqrt(mse)

def VertLine(x, y=1, color='0.8'):
		thinkplot.Plot([x, x], [0, y], color=color, linewidth=3)