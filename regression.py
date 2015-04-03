import ess
import statsmodels.formula.api as smf
import pandas as pd

df = ess.read()
df['jewish'] = (df.rlgdnm==5).astype(int)


def clf():
	model = smf.logit('jewish ~ happy + stflife + lrscale + eduyrs + sclmeet + ipshabt + ipsuces + imprich + impfun + rlgatnd', data=df) # Logit object
	results = model.fit() # BinaryResults object
	print results.summary()
	print results.pvalues

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

	print false_neg
	print sum(actual)

	sens = sum(true_pos)/sum(actual)
	fall = sum(false_pos)/(len(actual) - sum(actual))
	fnr = sum(false_neg)/sum(actual)
	spec = sum(true_neg)/(len(actual) - sum(actual))
	
	print 'Accuracy: ' + str(acc)
	print 'Sensitivity: ' + str(sens)
	print 'Fall-out: ' + str(fall)
	print 'FNR: ' + str(fnr)
	print 'Specificity: ' + str(spec)