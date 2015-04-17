import explore_1var
import explore_2var
import estimation
import hypothesis

packages = [explore_1var, explore_2var, estimation, hypothesis]

def run():
	for pkg in packages:
		pkg.plotAll()

if __name__ == '__main__':
	run()