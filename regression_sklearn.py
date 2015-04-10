import ess
import numpy as np
from matplotlib import pyplot as pl
from sklearn import cross_validation, preprocessing, linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.lda import LDA
from sklearn.qda import QDA

gbrt = GradientBoostingClassifier()
log = linear_model.LogisticRegression()
dec = DecisionTreeClassifier()
ada = AdaBoostClassifier()
nb = GaussianNB()
lda = LDA()
lin = linear_model.LinearRegression()
knn = KNeighborsClassifier()
svc = SVC()
qda = QDA()
rfc = RandomForestClassifier()
# http://scikit-learn.org/stable/modules/classes.html#module-sklearn.ensemble

clfs = [gbrt, ada, qda, log, lda, nb, lin,]
clfTags = ['GBRT', 'AdaBoost', 'QDA', 'Logistic reg', 'LDA', 'Naive Bayes', 'Linear reg']

def plotAll():
	(s, b) = prepare_ndarrays()
	clfSweep(s, clfs, clfTags)
	clfSweep(b, clfs, clfTags)


def prepare_ndarrays():
	df = ess.read()
	df['jewish'] = (df.rlgdnm==5).astype(int)

	ready = ['eduyrs']
	scales_10 = ['happy','lrscale']
	scales_7 = ['stflife','sclmeet','rlgatnd']
	scales_6 = ['ipshabt','ipsuces','imprich','impfun']

	scales = ready + scales_10 + scales_7 + scales_6

	# construct binaries
	# ...start with columns which are already ready
	binaries = ready

	# ...loop through scales & split them up
	mod = [(10,scales_10), (7,scales_7), (6,scales_6)]
	for tup in mod:
		gradations = tup[0]
		codes = tup[1]
		for c in codes:
			for i in range(1, gradations+1):
				newCode = c + str(i)
				df[newCode] = (df[c] == i).astype(int)
				binaries.append(newCode)

	s = DataHandler(df, scales, 'jewish', 'scales')
	b = DataHandler(df, binaries, 'jewish', 'binaries')

	return (s, b)


def clfSweep(dh, clfs, clfTags, show=False):
	pl.clf()
	fams = []
	pdms = []

	cl='rgbycmk' # color order
	for i, clf in enumerate(clfs):
		print clfTags[i]

		fa = []
		pd = []

		for train, test in cross_validation.StratifiedKFold(dh.y,10):
			print 'test'

			# scale  
			minmaxscaler = preprocessing.MinMaxScaler()
			features_scaled_train = minmaxscaler.fit_transform(dh.x[train,:])
			features_scaled_test = minmaxscaler.transform(dh.x[test,:])

			# fit
			clf.fit(features_scaled_train, dh.y[train])
			actual = dh.y[test]

			# compute probability of detection and false alarm rate, sweeping threshold
			if hasattr(clf, "decision_function"):
				margin = clf.decision_function(features_scaled_test) 
			else:
				margin = clf.predict_proba(features_scaled_test)[:,1]

			facv=[]
			pdcv=[]

			thresholds = np.linspace(-10, 10, 1000)
			for t in thresholds:
				predictions = (margin-t > 0).astype(int)
				facv += [np.mean(predictions[actual==0]==1)]
				pdcv += [np.mean(predictions[actual==1]==1)]

			fa += [np.array(facv)] # list of arrays for this train/test split
			pd += [np.array(pdcv)]

		fam = np.array(fa).mean(0) # stack arrays in list up and take average at each index
		pdm = np.array(pd).mean(0)

		# find area
		area = 0
		for j in range(1, len(fam)):
			# note: moving left on roc as you increase threshold

			# right point
			rfa = fam[j-1]
			rpd = pdm[j-1]

			# left point
			lfa = fam[j]
			lpd = pdm[j]

			# trapezoid geometry
			b = rfa-lfa # width of trapezoid
			h = (lpd+rpd)/2.0 # average height
			a = b*h

			area += a

		area = str(round(area, 3))
		pl.plot(fam, pdm, cl[j%7], label=clfTags[j] + ' ' + dh.label + ': ' + area, lw=2, zorder=1)

	for j in range(len(fams)):
		fam = fams[j]
		pdm = pdms[j]

		
	
	pl.legend(loc='lower right')
	pl.xlabel('False alarm rate')
	pl.ylabel('Detection rate')
	pl.title('ROC 10-fold CV for varying classifiers')

	if show:
		pl.show()
	else:
		pl.savefig('plots/regression/clfsweep-'+dh.label+'.jpg')


class DataHandler:
	def __init__(self, df, cols, y_col, label):
		self.df = df.dropna(subset=cols)
		self.x = self.df[cols].values
		self.y = self.df[y_col].values
		self.cols = cols
		self.label = label