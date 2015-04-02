import ess
import thinkstats2
import thinkplot
import numpy as np
import random
import math
import utils

df = ess.read()
scales = utils.getCodeList('data/codeinfo/scales.csv')

def plotAll():
	for col in scales:
		estimate(col)

def estimate(col, n=10, m=1000, show=False):
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

	printMean = 'Sample mean: ' + str(round(mu,2))
	printConfInt = 'Confidence interval: [' + str(round(confInt[0], 2)) + ', ' + str(round(confInt[1], 2)) + ']'
	printSE = 'Standard error: ' + str(round(RMSE(means, mu), 2))

	title = printMean + ' \n ' + printConfInt + ' , ' + printSE

	thinkplot.Config(xlabel='Estimated mean of ' + col, ylabel='CDF', title=title)
	VertLine(confInt[0])
	VertLine(confInt[1])
	VertLine(mu, color='0.2')
	thinkplot.Cdf(samplingCdf)

	if show:
		thinkplot.Show()
	else:
		thinkplot.Save('plots/estimation/'+col, formats=['jpg'])


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