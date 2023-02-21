import numpy as np
import datetime as dt
import yfinance as yf
import scipy.optimize as sc


# get data
def getData(stocks, start, end):
    stockData = yf.download(stocks, start=start, end=end)
    stockData = stockData['Close']
    
    returns = stockData.pct_change()
    meanReturns = returns.mean()
    covMatrix = returns.cov()
    return meanReturns, covMatrix

def portfolioPerformance(weights, meanReturns, covMatrix):
    returns = np.sum(meanReturns * weights) * 252
    std = np.sqrt(np.dot(weights.T, np.dot(covMatrix, weights))) * np.sqrt(252)
    return returns, std

'''
    In order to maximize the sharpe ratio, we can minimize
    the negative sharpe ratio.
'''
def negativeSharpRatio(weights, meanReturns, covMatrix, riskFreeRate=0):
    pReturns, pStd = portfolioPerformance(weights, meanReturns, covMatrix)
    return -(pReturns - riskFreeRate) / pStd

def maxSharpeRatio(meanReturns, covMatrix, riskFreeRate = 0, constraintSet=(0,1)):
    # Minimize the negative sharpe ratio by altering the weights of the portfolio
    numAssets = len(meanReturns)
    args = (meanReturns, covMatrix, riskFreeRate)

    # all of the summations of the weights in the portfolio have to
    # add up to 1
    constraints = ({ 'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = constraintSet
    bounds = tuple(bound for asset in range(numAssets))

    # first guess is equal distribution
    result = sc.minimize(negativeSharpRatio, numAssets * [1./numAssets], args=args,
                        method = 'SLSQP', bounds=bounds, constraints=constraints)
    return result

def maxSortinoRatio(meanReturns, riskFreeRate):
    # function to calculate the sortino ratio
    def negativeSortinoRatio(weights):
        portfolio_return = np.dot(meanReturns, weights)
        negative_returns = np.minimum(returns - portfolio_return, 0)
        downside_deviation = np.sqrt(np.mean(negative_returns ** 2))
        
        return -(portfolio_return - riskFreeRate) / downside_deviation

    numAssets = len(meanReturns)
    initial_weights = np.ones(numAssets) / numAssets
    
    # define the optimization constraints (weights must sum to 1)
    constraints = [{'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}]
    
    # define the bounds on the weights (between 0-1)
    bounds = [(0,1)] * numAssets

    result = sc.minimize(negativeSortinoRatio, initial_weights,
                        method = 'SLSQP', bounds=bounds, constraints=constraints)
    return result

stockList = ['AAPL', 'WMT', 'HD', 'AMZN']
endDate = dt.datetime.now()
startDate = endDate - dt.timedelta(days=365)

weights = np.array([0.4, 0.3, 0.2, 0.1])

'''
    Sharpe Ratio Calculation
'''


meanReturns, covMatrix = getData(stockList, start=startDate, end=endDate)
returns, std = portfolioPerformance(weights, meanReturns, covMatrix)

sharpeResult = maxSharpeRatio(meanReturns, covMatrix)
maxSharpeRatio, maxWeights = sharpeResult['fun'], sharpeResult['x']



'''
    Sortino Ratio Calculation
'''

meanReturns, covMatrix = getData(stockList, start=startDate, end=endDate)
returns, std = portfolioPerformance(weights, meanReturns, covMatrix)

sortinoResult = maxSortinoRatio(meanReturns, 0)
maxSortinoRatio, maxWeights = sortinoResult['fun'], sortinoResult['x']


print(maxSharpeRatio, maxWeights)
print(maxSortinoRatio, maxWeights)

