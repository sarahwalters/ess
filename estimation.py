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
		for i in range(len(rlgdfs)):
			estimateDiffMeans(col, rlgdfs[i], rlglabels[i])

		# plot all diff means together
		estimateDiffMeans_all(col, rlgdfs, rlglabels)


def estimateMean(col, n=10, m=1000, show=False):
	# for plotting confidence interval & mean
	def VertLine(x, y=1, color='0.8'):
		thinkplot.Plot([x, x], [0, y], color=color, linewidth=3)

	# get data
	data = unaffiliated[col] # length: ~2e6

	# estimators
	mu = data.mean()
	sigma = data.std()
	
	# run "experiments"
	means = []
	for _ in range(m):
		xs = [random.gauss(mu,sigma) for _ in range(n)]
		xbar = np.mean(xs)
		means.append(xbar)

	# sampling CDF & summary statistics
	samplingCdf = thinkstats2.Cdf(means, label='sampling')
	confInt = samplingCdf.ConfidenceInterval()

	printMean = 'Sample mean: ' + str(round(mu,3))
	printConfInt = 'Confidence interval: [' + str(round(confInt[0], 3)) + ', ' + str(round(confInt[1], 3)) + ']'
	printSE = 'Standard error: ' + str(round(RMSE(means, mu), 3))

	title = printMean + ' \n ' + printConfInt + ' , ' + printSE

	thinkplot.Config(xlabel='Estimated mean of ' + col + ' among unaffiliated respondents', ylabel='CDF', title=title)
	VertLine(confInt[0])
	VertLine(confInt[1])
	VertLine(mu, color='0.2')
	thinkplot.Cdf(samplingCdf)

	if show:
		thinkplot.Show()
	else:
		thinkplot.Save('plots/estimation/mean/'+ col, formats=['jpg'])


def estimateDiffMeans(col, rlgdf, rlglabel, n=10, m=1000, show=False):
	# for plotting confidence interval & mean
	def VertLine(x, y=1, color='0.8'):
		thinkplot.Plot([x, x], [0, y], color=color, linewidth=3)

	# get data
	dataReligion = rlgdf[col] # length: ~2e6
	dataUnaffiliated = unaffiliated[col]

	# estimators
	mu = dataReligion.mean() - dataUnaffiliated.mean()
	sigma = (dataReligion.std()**2/len(dataReligion) + dataUnaffiliated.std()**2/len(dataUnaffiliated))**0.5
	# standard dev. from http://www.kean.edu/~fosborne/bstat/06b2means.html

	# run "experiments"
	means = []
	for _ in range(m):
		xs = [random.gauss(mu,sigma) for _ in range(n)]
		xbar = np.mean(xs)
		means.append(xbar)

	# sampling CDF & summary statistics
	samplingCdf = thinkstats2.Cdf(means, label='sampling')
	confInt = samplingCdf.ConfidenceInterval()

	printMean = 'Sample mean: ' + str(round(mu,3))
	printConfInt = 'Confidence interval: [' + str(round(confInt[0], 3)) + ', ' + str(round(confInt[1], 3)) + ']'
	printSE = 'Standard error: ' + str(round(RMSE(means, mu), 3))

	title = printMean + ' \n ' + printConfInt + ' , ' + printSE

	thinkplot.Config(xlabel='Estimated ' + rlglabel + '/unaffiliated diff in means of ' + col, ylabel='CDF', title=title)
	VertLine(confInt[0])
	VertLine(confInt[1])
	VertLine(mu, color='0.2')
	thinkplot.Cdf(samplingCdf)

	if show:
		thinkplot.Show()
	else:
		thinkplot.Save('plots/estimation/diffMeans/' + col + '_' + rlglabel, formats=['jpg'])


class Estimate:
	def __init__(self, mu, sigma, m, n):
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
		self.samplingCdf = thinkstats2.Cdf(means, label='sampling')
		self.confInt = samplingCdf.ConfidenceInterval()

		# for displaying nicely
		self.printMean = 'Sample mean: ' + str(round(mu,3))
		self.printConfInt = 'Confidence interval: [' + str(round(self.confInt[0], 3)) + ', ' + str(round(self.confInt[1], 3)) + ']'
		self.printSE = 'Standard error: ' + str(round(RMSE(self.means, mu), 3))



def estimateDiffMeans_all(col, rlgdfs, rlglabels, n=10, m=1000, show=False):
	# get data
	rlgSeriesArray = [rlgdf[col] for rlgdf in rlgdfs]
	unaffSeries = unaffiliated[col]

	# estimators
	muArray = [rs.mean() - unaffSeries.mean() for rs in rlgSeriesArray]
	sigmaArray = [(rs.std()**2/len(rs) + unaffSeries.std()**2/len(unaffSeries))**0.5 for rs in rlgSeriesArray]
	# standard dev. from http://www.kean.edu/~fosborne/bstat/06b2means.html
	
	# run "experiments"
	samplingCdfs = []
	for i in range(len(muArray)):
		mu = muArray[i]
		sigma = sigmaArray[i]

		means = []
		for _ in range(m):
			xs = [random.gauss(mu,sigma) for _ in range(n)]
			xbar = np.mean(xs)
			means.append(xbar)

		# sampling CDF & summary statistics
		# ... for making the label
		samplingCdf = thinkstats2.Cdf(means, label='dummy title')
		
		# ... make the label		
		confInt = samplingCdf.ConfidenceInterval()
		printMean = 'Mean: ' + str(round(mu,3))
		printConfInt = 'CI: [' + str(round(confInt[0], 3)) + ', ' + str(round(confInt[1], 3)) + ']'
		printSE = 'SE: ' + str(round(RMSE(means, mu), 3))

		title = rlglabels[i] + ' = ' + printMean + ', ' + printConfInt + ' , ' + printSE
		
		# ...assign the label
		samplingCdf = thinkstats2.Cdf(means, label=title) # real cdf with label
		samplingCdfs.append(samplingCdf)

	thinkplot.Config(xlabel='Estimated religious/unaffiliated diff in means of ' + col, ylabel='CDF', loc='lower center', bbox_to_anchor=(0.5, -0.6), htscale=0.825)
	if col in values:
		thinkplot.Config(xlim=(-1.25,1.25))

	thinkplot.Cdfs(samplingCdfs)

	if show:
		thinkplot.Show()
	else:
		thinkplot.Save('plots/estimation/diffMeans/' + col + '_all', formats=['jpg'])

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