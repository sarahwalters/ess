import ess
import thinkstats2
import thinkplot
import numpy as np
import random
import math
import utils

df = ess.read()
jewish = df[df.rlgdnm == 5]
other = df[df.rlgdnm != 5]
scales = utils.getCodeList('data/codeinfo/scales.csv')

def plotAll():
	# estimate means
	for col in scales:
		estimateMean(col)
		estimateDiffMeans(col)


def estimateMean(col, n=10, m=1000, show=False):
	# for plotting confidence interval & mean
	def VertLine(x, y=1, color='0.8'):
		thinkplot.Plot([x, x], [0, y], color=color, linewidth=3)

	# get data
	data = df[col] # length: ~2e6

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

	thinkplot.Config(xlabel='Estimated mean of ' + col, ylabel='CDF', title=title)
	VertLine(confInt[0])
	VertLine(confInt[1])
	VertLine(mu, color='0.2')
	thinkplot.Cdf(samplingCdf)

	if show:
		thinkplot.Show()
	else:
		thinkplot.Save('plots/estimation/mean/'+col, formats=['jpg'])

def estimateDiffMeans(col, n=10, m=1000, show=False):
	# for plotting confidence interval & mean
	def VertLine(x, y=1, color='0.8'):
		thinkplot.Plot([x, x], [0, y], color=color, linewidth=3)

	# get data
	dataJewish = jewish[col] # length: ~2e6
	dataOther = other[col]

	# estimators
	mu = dataJewish.mean() - dataOther.mean()
	sigma = (dataJewish.std()**2/len(dataJewish) - dataOther.std()**2/len(dataOther))**0.5
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

	thinkplot.Config(xlabel='Estimated Jewish/other diff in means of ' + col, ylabel='CDF', title=title)
	VertLine(confInt[0])
	VertLine(confInt[1])
	VertLine(mu, color='0.2')
	thinkplot.Cdf(samplingCdf)

	if show:
		thinkplot.Show()
	else:
		thinkplot.Save('plots/estimation/diffMeans/'+col, formats=['jpg'])


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