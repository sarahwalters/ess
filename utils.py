import csv

def getCodeList(csvpath):
	res = []
	with open(csvpath) as csvfile:
		reader = csv.reader(csvfile, delimiter=',')
		for row in reader:
			res += row
	return res

def getCountryDict(csvpath='data/codeinfo/countryLabels.csv'):
	countryDict = {}
	with open(csvpath) as csvfile:
		reader = csv.reader(csvfile, delimiter=' ')
		for row in reader:
			countryDict[row[0]] = row[1]

	lengths = [len(v) for v in countryDict.values()]
	countryDict['longest'] = max(lengths)

	return countryDict
