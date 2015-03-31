import ess
import utils
import thinkplot
import thinkstats2
import pandas as pd

# DO SETUP
df = ess.read()
scales = utils.getCodeList('data/codeinfo/scales.csv')


# MAIN METHODS
def plotAll(df=df):
	# all
	plotDf()
	organizeSummaryFile('all')

	# by religion
	religions = [(1,'catholic'), (2,'protestant'), (3,'easternOrthodox'), (4,'otherChristian'), (5,'jewish'), (6,'islamic'), (7,'easternReligious'), (8,'otherNonChristian')]
	for r in religions:
		plotDf(df=df[df.rlgdnm==r[0]], name=r[1])
		organizeSummaryFile(r[1])


def plotDf(df=df, name='all'):
	# set up summary file
	clearSummaryFile(name)
	sf = openSummaryFile(name)

	# do the plotting
	path = 'plots/twoVar/'+name+'/'
	n = len(scales)
	for i in range(n):
		for j in range(i+1, n):
			col0 = scales[i]
			col1 = scales[j]
			plotHexbin(col0, col1, df=df, sf=sf, folder=path)

	# close the file
	sf.close()

def plotHexbin(col0, col1, show=False, df=df, sf=allFile, folder='plots/twoVar/all'):
	print (col0, col1)

	jit0 = thinkstats2.Jitter(df[col0], 0.5)
	jit1 = thinkstats2.Jitter(df[col1], 0.5)

	sub_df = pd.concat([jit0, jit1], axis=1)
	sub_df.dropna(subset=[col0, col1], inplace=True)

	xs = sub_df[col0]
	ys = sub_df[col1]

	# compute correlation
	corr = round(thinkstats2.SpearmanCorr(xs, ys),3)
	switch = abs(corr)

	if switch > 0.4:
		sf.write(str(corr) + ',' + col0 + ' ' + col1 + '\n')
		# where to save
		if 0.4 < switch and switch <= 0.6:
			folder += 'medium/'
		elif 0.6 < switch and switch <= 1:
			folder += 'strong/'

		# hexbin plot
		thinkplot.HexBin(xs, ys)
		xl = xs.min()
		xr = xs.max()
		yb = ys.min()
		yt = ys.max()

		# fit and plot line
		inter, slope = thinkstats2.LeastSquares(xs, ys)
		fxs, fys = thinkstats2.FitLine(xs, inter, slope)
		thinkplot.Plot(fxs, fys)

		# config & show/save
		thinkplot.Config(axis=[xl, xr, yb, yt], xlabel=col0, ylabel=col1, title='Correlation: ' + str(corr))
		if show:
			thinkplot.Show()
		else:
			thinkplot.Save(folder + col0 + '_' + col1, formats=['jpg'])


# SUMMARY FILE METHODS
def clearSummaryFile(name):
	with open('plots/twoVar/'+name+'/summary.txt', 'w') as f:
		f.write('')
		f.close()

def openSummaryFile(name):
	return open('plots/twoVar/'+name+'/summary.txt', 'a')

def organizeSummaryFile(name):
	filepath = 'plots/twoVar/'+name+'/summary.txt'
	# build list
	data = []
	read = open(filepath, 'r')
	for line in read:
		data.append(line.split(',')) # (corr, pair) tuple

	# sort list
	data.sort(key=lambda tup: abs(float(tup[0]))) # sort by abs(corr)
	data.reverse() # highest to lowest

	print data

	# reformat and write
	for i,line in enumerate(data):
		data[i] = ','.join(line)

	header = name + '\nCount: ' + str(len(data)) + '\n'
	toWrite = header + ''.join(data)
	
	with open(filepath, 'w') as f:
		f.write(toWrite)