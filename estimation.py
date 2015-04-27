import ess
import thinkstats2
import thinkplot
import numpy as np
import random
import math
import utils

df = ess.read()
df.rlgdnm.fillna(value=0, inplace=True) # assuming NaN == not religious
unaffiliated = df[df.rlgdnm == 0] # unaffiliated
scales = utils.getCodeList('data/codeinfo/scales.csv')
values = utils.getCodeList('data/codeinfo/values.csv')

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

def debug():
	rlgLookup = [(1, 'Catholic'), (2, 'Protestant'), (5, 'Jewish'), (6, 'Islamic')]
	rlgdfs = [df[df.rlgdnm==tup[0]] for tup in rlgLookup]
	rlglabels = [tup[1] for tup in rlgLookup]

	estimateMean('ipcrtiv')


def estimateMean(col, n=10, m=1000, show=False):
	# get data
	data = unaffiliated[col] # length: ~2e6

	# estimators
	mu = data.mean()
	sigma = data.std()
	
	# make labels
	xlabel='Estimated mean of ' + col + ' among unaffiliated respondents'
	savepath = 'plots/estimation/mean/'+ col

	# run "experiments"
	est = GaussianEstimate(mu, sigma, m, n, xlabel, savepath)
	est.plot(show=show)

	return est


def estimateDiffMeans(col, rlgdf, rlglabel, n=10, m=1000, show=False):
	# get data
	dataReligion = rlgdf[col] # length: ~2e6
	dataUnaffiliated = unaffiliated[col]

	# estimators
	mu = dataReligion.mean() - dataUnaffiliated.mean()
	sigma = (dataReligion.std()**2/len(dataReligion) + dataUnaffiliated.std()**2/len(dataUnaffiliated))**0.5
	# standard dev. from http://www.kean.edu/~fosborne/bstat/06b2means.html

	# make labels
	xlabel = 'Estimated ' + rlglabel + '/unaffiliated diff in means of ' + col
	savepath = 'plots/estimation/diffMeans/' + col + '_' + rlglabel

	# run "experiments"
	est = GaussianEstimate(mu, sigma, m, n, xlabel, savepath)
	est.plot(show=show)

	return est


class GaussianEstimate:
	def __init__(self, mu, sigma, m, n, xlabel, savepath):
		# labels
		self.xlabel = xlabel
		self.savepath = savepath

		# estimators
		self.mu = mu
		self.sigma = sigma

		# do estimation
		self.means = []
		for _ in range(m):
			xs = [random.gauss(mu,sigma) for _ in range(n)]
			xbar = np.mean(xs)
			self.means.append(xbar)

		# make sampling cdf
		self.samplingCdf = thinkstats2.Cdf(self.means, label='dummy label')
		self.confInt = self.samplingCdf.ConfidenceInterval()

		# for displaying nicely
		self.printMean = 'Sample mean: ' + str(round(mu,3))
		self.printConfInt = 'Confidence interval: [' + str(round(self.confInt[0], 3)) + ', ' + str(round(self.confInt[1], 3)) + ']'
		self.printSE = 'Standard error: ' + str(round(RMSE(self.means, mu), 3))

		# give cdf proper label
		self.title = self.printMean + ' \n ' + self.printConfInt + ' , ' + self.printSE
		self.samplingCdf.label = self.title

	
	def plot(self, show=False):
		thinkplot.Clf()

		thinkplot.Config(xlabel=self.xlabel, ylabel='CDF', title=self.title, legend=False)

		VertLine(self.confInt[0])
		VertLine(self.confInt[1])
		VertLine(self.mu, color='0.2')

		thinkplot.Cdf(self.samplingCdf)

		if show:
			thinkplot.Show()
		else:
			thinkplot.Save(self.savepath, formats=['jpg'])


def diffMeansPlot(col, diffMeansEsts, show=False):
	thinkplot.Config(xlabel='Estimated religious/unaffiliated diff in means of ' + col, ylabel='CDF', loc='lower center', bbox_to_anchor=(0.5, -0.6), htscale=0.825)
	if col in values:
		thinkplot.Config(xlim=(-1.25,1.25))

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