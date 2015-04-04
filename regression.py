import ess
import statsmodels.formula.api as smf
import pandas as pd

df = ess.read()
df['jewish'] = (df.rlgdnm==5).astype(int)


def regr1():
	formula = 'jewish ~ happy + stflife + lrscale + eduyrs + sclmeet + ipshabt + ipsuces + imprich + impfun + rlgatnd'
	clf(formula)


def regr2():
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
	clf(formula)


def clf(formula):
	model = smf.logit(formula, data=df) # Logit object
	results = model.fit() # BinaryResults object
	#print results.summary()

	endog = pd.DataFrame(model.endog, columns=[model.endog_names])
	exog = pd.DataFrame(model.exog, columns=model.exog_names)

	# Accuracy of model
	predict = (results.predict() >= 0.5)
	actual = endog['jewish']

	true_pos = predict*actual
	true_neg = (1-predict) * (1-actual)
	false_pos = predict*(1-actual)
	false_neg = (1-predict) * actual

	acc = (sum(true_pos) + sum(true_neg))/len(actual)

	sens = sum(true_pos)/sum(actual)
	fall = sum(false_pos)/(len(actual) - sum(actual))
	fnr = sum(false_neg)/sum(actual)
	spec = sum(true_neg)/(len(actual) - sum(actual))
	
	print 'Accuracy: ' + str(acc)
	print 'Sensitivity: ' + str(sens)
	print 'Fall-out: ' + str(fall)
	print 'FNR: ' + str(fnr)
	print 'Specificity: ' + str(spec)