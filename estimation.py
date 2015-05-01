import ess
import thinkstats2
import thinkplot
import numpy as np
import random
import math
import utils

df = ess.read()
unaffiliated = df[df.rlgblg == 2] # unaffiliated
scales = utils.getCodeList('data/codeinfo/scales.csv')
values = utils.getCodeList('data/codeinfo/values.csv')
centeredValues = utils.getCodeList('data/codeinfo/centeredValues.csv')

def plotAll():
	rlgLookup = [(1, 'Catholic'), (2, 'Protestant'), (5, 'Jewish'), (6, 'Islamic')]
	rlgdfs = [df[df.rlgdnm==tup[0]] for tup in rlgLookup]
	rlglabels = [tup[1] for tup in rlgLookup]

	for col in scales:
		# unaffiliated means
		estimateMean(col)

		# diff in means compared to unaffiliated, for each religion
		diffMeansEsts = []
		for i in range(len(rlgdfs)):
			est = estimateDiffMeans(col, rlgdfs[i], rlglabels[i])
			diffMeansEsts.append(est)

		# plot all diff means together
		diffMeansPlot(col, diffMeansEsts)


def estimateMean(col, n=10, m=1000, show=False):
	# get data
	data = unaffiliated[col] # length: ~2e6
	data.dropna(inplace=True)

	# estimators
	mu = data.mean()
	sigma = data.std()
	
	# make labels
	xlabel='Estimated mean of ' + col + ' among unaffiliated respondents'
	savepath = 'plots/estimation/mean/'+ col

	# run "experiments"
	#est = GaussianEstimate(mu, sigma, m, n, xlabel, savepath)
	est = ResampledEstimate([data], m, n, xlabel, savepath)
	est.plot(show=show)

	return est


def estimateDiffMeans(col, rlgdf, rlglabel, n=10, m=1000, show=False):
	# get data
	dataReligion = rlgdf[col]
	dataUnaffiliated = unaffiliated[col]

	dataReligion.dropna(inplace=True)
	dataUnaffiliated.dropna(inplace=True)

	# estimators
	mu = dataReligion.mean() - dataUnaffiliated.mean()
	sigma = (dataReligion.std()**2/len(dataReligion) + dataUnaffiliated.std()**2/len(dataUnaffiliated))**0.5
	# standard dev. from http://www.kean.edu/~fosborne/bstat/06b2means.html

	# make labels
	xlabel = 'Estimated ' + rlglabel + '/unaffiliated diff in means of ' + col
	savepath = 'plots/estimation/diffMeans/' + col + '_' + rlglabel

	# run "experiments"
	#est = GaussianEstimate(mu, sigma, m, n, xlabel, savepath, rlg=rlglabel)
	est = ResampledEstimate([dataReligion, dataUnaffiliated], m, n, xlabel, savepath, rlg=rlglabel)
	est.plot(show=show)

	return est


class ResampledEstimate:
	def __init__(self, data, m, n, xlabel, savepath, rlg='All'):
		# labels
		self.rlg = rlg
		self.xlabel = xlabel
		self.savepath = savepath

		# estimators
		self.data = data

		# do estimation
		self.estimated = []
		if len(data) == 1: # single mean
			for _ in range(m):
				xs = thinkstats2.Resample(self.data[0])
				mean = np.mean(xs)
				self.estimated.append(mean)

		elif len(data) == 2: # diff means
			self.diffMeans = []
			for _ in range(m):
				xs_rlg = thinkstats2.Resample(self.data[0])
				xs_unaff = thinkstats2.Resample(self.data[1])
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


def diffMeansPlot(col, diffMeansEsts, show=False):
	thinkplot.Clf()

	if col in values:
		thinkplot.Config(xlim=(-1.4,1.4)) # keep all value axes consistent
	elif col in centeredValues:
		thinkplot.Config(xticks=np.linspace(-1, 1, 11), xlim=(-0.9,0.9))

	thinkplot.Config(xlabel='Estimated religious/unaffiliated diff in means of ' + col, ylabel='CDF', legend=True, loc='lower center', bbox_to_anchor=(0.5, -0.6), htscale=0.825)
	
	samplingCdfs = [est.samplingCdf for est in diffMeansEsts]
	thinkplot.Cdfs(samplingCdfs)

	if show:
		thinkplot.Show()
	else:
		thinkplot.Save('plots/estimation/diffMeansAll/' + col, formats=['jpg'])


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