import ess
import thinkstats2
import pandas as pd
import utils

# METHODS NEEDED FOR SETUP
def rlgProportions(subdf):
	rlgblg = subdf.rlgblg.value_counts()
	rlgdnm = subdf.rlgdnm.value_counts()
	rlgdnm = rlgdnm.append(pd.Series(rlgblg.loc[2], index=[0])) # not religious
	
	counts = rlgdnm.sort_index()
	counts.name = 'counts'
	props = counts/sum(rlgdnm)
	props.name = 'props'
	info = pd.concat([props, counts], axis=1)
	
	return info


# SETUP
df = ess.read()
totalInfo = rlgProportions(df)
countries = utils.getCountryDict()


# MAIN METHODS
def idealCountry():
	# group by country
	gp = df.groupby('cntry')

	# check fits
	fits = []
	for cntry,subdf in gp:
		info = rlgProportions(subdf)
		fit = compareProportions(info.props)
		fits.append((cntry,fit,len(subdf)))
	fits.sort(key=lambda tup: tup[1])

	# print all fits
	addedChars = len(' ()  ')
	cntryWidth = countries['longest'] + addedChars

	print '-------------------------------------------------------'
	print 'Compare fits:'
	print 'Cntry' + ' '*(cntryWidth+3) + 'Proportion RMSE     Respondents'

	for fit in fits:
		# format country
		cntry = str(fit[0]) + ' (' + countries[fit[0]] + ')'
		while len(cntry) < cntryWidth:
			cntry += ' '

		# format rest of info
		print cntry + '        %.2f                %s' %(fit[1], fit[2])

	# print proportions for top country & overall proportions
	print '-------------------------------------------------------'
	print 'Overall:'
	print totalInfo
	print '---'
	for fit in fits:
		country = fit[0]
		print country + ' (' + countries[country] + '):'
		print rlgProportions(gp.get_group(country))
		print '---'


def countrySet(countries):
	# filter df down
	selector = (df.cntry == countries.pop())
	while len(countries) > 0:
		selector = selector | (df.cntry == countries.pop())
	print rlgProportions(df[selector])


def compareProportions(props):
	diff = props - totalInfo.props
	mses = diff**2
	rmse = mses.sum()**0.5
	return rmse