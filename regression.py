import ess
import statsmodels.formula.api as smf
import pandas as pd
import numpy as np
import thinkplot

df = ess.read()
df['jewish'] = (df.rlgdnm==5).astype(int)


def plotAll(show=False):
	# get formulas
	formula1 = 'jewish ~ happy + stflife + lrscale + eduyrs + sclmeet + ipshabt + ipsuces + imprich + impfun + rlgatnd'
	formula2 = splitScaleFormula()

	# make classifiers
	clf1 = Clf(formula1, 'scales')
	clf2 = Clf(formula2, 'binaries')

	# summaries
	clf1.summary()
	clf2.summary()
	
	# plot
	clf1.roc()
	clf2.roc()

	thinkplot.Config(axis=[0,1,0,1], xlabel='Fall out', ylabel='Sensitivity', title='ROC curve', legend=True)

	if show:
		thinkplot.Show()
	else:
		thinkplot.Save('plots/regression/roc', formats=['jpg'])


def splitScaleFormula():
	# already ready to use
	fitCols = ['eduyrs']

	# to modify
	scales_10 = ['happy','lrscale']
	scales_7 = ['stflife','sclmeet','rlgatnd']
	scales_6 = ['ipshabt','ipsuces','imprich','impfun']

	mod = [(10,scales_10), (7,scales_7), (6,scales_6)]

	for tup in mod:
		gradations = tup[0]
		codes = tup[1]
		for c in codes:
			for i in range(1, gradations+1):
				newCode = c + str(i)
				df[newCode] = (df[c] == i).astype(int)
				fitCols.append(newCode)

	formula = 'jewish ~ ' + ' + '.join(fitCols)
	return formula

	
class Clf:
	def __init__(self, formula, label=None):
		self.formula = formula
		self.model = smf.logit(formula, data=df) # Logit object
		self.results = self.model.fit() # BinaryResults object
		self.probabilities = self.results.predict()
		self.endog = pd.DataFrame(self.model.endog, columns=[self.model.endog_names])
		self.exog = pd.DataFrame(self.model.exog, columns=self.model.exog_names)
		self.label = label

	def summary(self):
		print self.results.summary()

	def evaluation(self, threshold=0.5):
		predicted = (results.predict() >= threshold)
		actual = endog.jewish
		ce = ClfEval(predicted, actual)
		ce.report()

	def roc(self, thresholds=np.linspace(0,1,100)):
		# over threshold sweep, collect false alarm rates & probabilities of detection
		false_alarm = []
		prob_detection = []
		for t in thresholds:
			# evaluate classifier
			predicted = pd.Series((self.probabilities > t).astype(int))
			actual = self.endog.jewish
			ce = ClfEval(predicted, actual)

			# collect relevant stats
			false_alarm.append(ce.fallout)
			prob_detection.append(ce.sensitivity)

		# approximate area under curve (trapezoidal Riemann sum)
		area = 0
		for i in range(1, len(thresholds)):
			# note: moving left on roc as you increase threshold

			# right point
			rfa = false_alarm[i-1]
			rpd = prob_detection[i-1]

			# left point
			lfa = false_alarm[i]
			lpd = prob_detection[i]

			# trapezoid geometry
			b = rfa-lfa # width of trapezoid
			h = (lpd+rpd)/2.0 # average height
			a = b*h

			area += a

		area = str(round(area, 3))

		# plot (thinkplot collects)
		if self.label:
			thinkplot.Plot(false_alarm, prob_detection, label=self.label + ': ' + area)
		else:
			thinkplot.Plot(false_alarm, prob_detection, label=area)
	

class ClfEval:
	def __init__(self, predicted, actual):
		self.predicted = predicted
		self.actual = actual

		self.tp = sum(predicted*actual) # true positives
		self.tn = sum((1-predicted) * (1-actual)) # true negatives
		self.fp = sum(predicted*(1-actual)) # false positives
		self.fn = sum((1-predicted) * actual) # false negatives
		self.count = len(actual) # number of responses
		self.cp = sum(actual) # condition positives
		self.cn = self.count - self.cp # condition negatives

		self.accuracy = (self.tp + self.tn)/float(self.count)

		self.sensitivity = self.tp/float(self.cp)
		self.fallout = self.fp/float(self.cn)
		self.false_neg_rate = self.fn/float(self.cp)
		self.specificity = self.tn/float(self.cn)

	def report(self):
		print 'Accuracy: ' + str(self.accuracy)
		print 'Sensitivity: ' + str(self.sensitivity)
		print 'Fall-out: ' + str(self.fallout)
		print 'FNR: ' + str(self.false_neg_rate)
		print 'Specificity: ' + str(self.specificity)
