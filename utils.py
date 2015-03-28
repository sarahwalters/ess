import csv

def getCodeList(csvpath):
	res = []
	with open(csvpath) as csvfile:
		reader = csv.reader(csvfile, delimiter=',')
		for row in reader:
			res += row
	return res