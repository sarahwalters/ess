import ess
import thinkstats2
import thinkplot
import numpy as np
import utils

df = ess.read()
jewish = df[df.rlgdnm == 5]
other = df[df.rlgdnm != 5]
scales = utils.getCodeList('data/codeinfo/scales.csv')

# from ThinkStats2
class DiffMeansPermute(thinkstats2.HypothesisTest):
    """Tests a difference in means by permutation."""

    def TestStatistic(self, data):
        """Computes the test statistic.

        data: data in whatever form is relevant        
        """
        group1, group2 = data
        test_stat = abs(group1.mean() - group2.mean())
        return test_stat

    def MakeModel(self):
        """Build a model of the null hypothesis.
        """
        group1, group2 = self.data
        self.n, self.m = len(group1), len(group2)
        self.pool = np.hstack((group1, group2))

    def RunModel(self):
        """Run the model of the null hypothesis.

        returns: simulated data
        """
        np.random.shuffle(self.pool)
        data = self.pool[:self.n], self.pool[self.n:]
        return data

def plotAll(show=False):
    #for col in scales:
    for col in ['happy', 'stflife', 'lrscale', 'eduyrs', 'sclmeet', 'ipshabt', 'ipsuces', 'imprich', 'impfun', 'rlgatnd', 'pray', 'imptrad']:
        data = (jewish[col].dropna().values, other[col].dropna().values)
        dmp = DiffMeansPermute(data)
        p_value = dmp.PValue(iters=1000)
        dmp.PlotCdf()

        title = 'P-value'
        if (p_value < 0.001):
            title += ' < 0.001'
        else:
            title += ': ' + str(round(p_value, 4))
        thinkplot.Config(xlabel=col + ' under Jewish == other null hypothesis', ylabel='CDF', title=title)

        if show:
            thinkplot.Show()
        else:
            thinkplot.Save('plots/hypothesis/diffMeans/'+col, formats=['jpg'])