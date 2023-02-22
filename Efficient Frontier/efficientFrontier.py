import numpy as np
import datetime as dt
import yfinance as yf
import scipy.optimize as sc


def getData(stocks, start, end):
    stockData = yf.download(stocks, start=start, end=end)
    stockData = stockData['Adj Close']
    returns = stockData.pct_change()
    meanReturns = returns.mean()
    covMatrix = returns.cov()
    return meanReturns, covMatrix

def portfolioPerformance(weights, meanReturns, covMatrix, riskFreeRate=0):
    returns = np.sum(meanReturns*weights)*252
    std = np.sqrt(np.dot(weights.T,np.dot(covMatrix, weights)))*np.sqrt(252)
    sharpe = (returns - riskFreeRate) / std
    return returns, std, sharpe

def negativeSharpeRatio(weights, meanReturns, covMatrix, riskFreeRate = 0):
    pReturns, pStd, pSharpe = portfolioPerformance(weights, meanReturns, covMatrix, riskFreeRate)
    return -pSharpe

def maxSharpeRatio(meanReturns, covMatrix, riskFreeRate = 0, constraintSet=(0,1)):
    "Minimize the negative SR, by altering the weights of the portfolio"
    numAssets = len(meanReturns)
    args = (meanReturns, covMatrix, riskFreeRate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = constraintSet
    bounds = tuple(bound for asset in range(numAssets))
    result = sc.minimize(negativeSharpeRatio, numAssets*[1./numAssets], args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    return result

def portfolioVarianceSharpe(weights, meanReturns, covMatrix):
    return portfolioPerformance(weights, meanReturns, covMatrix)[1]

def minimizeVarianceSharpe(meanReturns, covMatrix, riskFreeRate=0, constraintSet=(0,1)):
    # minimize the portfolio variance by altering the weights/allocation of assets in the portfolio
    numAssets = len(meanReturns)
    args = (meanReturns, covMatrix)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = constraintSet
    bounds = tuple(bound for asset in range(numAssets))
    result = sc.minimize(portfolioVarianceSharpe, numAssets*[1./numAssets], 
                         args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    return result

def getData2(stocks, start, end):
    stockData = yf.download(stocks, start=start, end=end)
    stockData = stockData['Close']
    returns = stockData.pct_change()
    meanReturns = returns.mean()
    covMatrix = returns.cov()
    return meanReturns, covMatrix

# Define the function to calculate portfolio performance
def portfolioPerformance2(weights, meanReturns, covMatrix):
    returns = np.sum(meanReturns * weights) * 252
    std = np.sqrt(
        np.dot(weights.T, np.dot(covMatrix, weights))
    ) * np.sqrt(252)
    return returns, std

# Define the function to calculate negative Sortino ratio
def negativeSortinoRatio(weights, meanReturns, covMatrix, riskFreeRate=0.0, mar=0.0):
    pReturns, pStd = portfolioPerformance2(weights, meanReturns, covMatrix)
    downsideDeviation = np.sqrt(np.dot(weights.T, np.dot(covMatrix.loc[meanReturns < mar, meanReturns < mar], weights))) * np.sqrt(252)
    return -(pReturns - riskFreeRate) / downsideDeviation

# Define the function to maximize Sortino ratio
def maxSortinoRatio(meanReturns, covMatrix, riskFreeRate=0.0, mar=0.0, constraintSet=(0,1)):
    numAssets = len(meanReturns)
    args = (meanReturns, covMatrix, riskFreeRate, mar)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple(constraintSet for asset in range(numAssets))
    result = sc.minimize(negativeSortinoRatio, numAssets*[1./numAssets], args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    return result

def portfolioVarianceSortino(weights, meanReturns, covMatrix):
    return portfolioPerformance2(weights, meanReturns, covMatrix)[1]

def minimizeVarianceSortino(meanReturns, covMatrix, riskFreeRate=0, constraintSet=(0,1)):
    numAssets = len(meanReturns)
    args = (meanReturns, covMatrix)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple(constraintSet for asset in range(numAssets))
    result = sc.minimize(portfolioVarianceSortino, numAssets*[1./numAssets], args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    return result

stockList = ['AAPL', 'MSFT', 'JNJ']
endDate = '2022-12-31'
startDate = '2022-01-01'

weights = np.array([0.3, 0.5, 0.2])

'''
    Sharpe Ratio Calculation
'''
# Example usage
meanReturns, covMatrix = getData(['AAPL', 'WMT', 'HD', 'AMZN'], '2021-01-01', '2022-01-01')
result = maxSharpeRatio(meanReturns, covMatrix, riskFreeRate=0)
print("Sharpe ratio: ", -result.fun)
print("Optimal weights: ", result.x)

print("-------------------------------------")

minVarResult = minimizeVarianceSharpe(meanReturns, covMatrix)
minVar, minVarWeights = minVarResult['fun'], minVarResult['x']
print("Minimum Portfolio Variance", minVar)
print("Optimal Weights", minVarResult['x'])


'''
    Sortino Ratio Calculation
'''


meanReturns, covMatrix = getData2(stockList, start=startDate, end=endDate)
mar = 0.001
result = maxSortinoRatio(meanReturns, covMatrix, mar=mar)
weights = result.x
pReturns, pStd = portfolioPerformance2(weights, meanReturns, covMatrix)
downsideReturns = meanReturns[meanReturns < mar]

if len(downsideReturns) == 0:
    downsideDeviation = 0
else:
    downsideDeviation = np.sqrt(np.dot(weights.T, np.dot(covMatrix.loc[meanReturns < mar, meanReturns < mar], weights))) * np.sqrt(252)
sortinoRatio = (pReturns - mar) / downsideDeviation

print("-------------------------------------")

#print(maxSharpeRatio, maxWeights)
print("Portfolio returns:", pReturns)
print("Weights:", weights)
print("Portfolio downside deviation:", downsideDeviation)
print("Sortino ratio:", sortinoRatio)

print("-------------------------------------")

minVarResult = minimizeVarianceSortino(meanReturns, covMatrix)
minVar, minVarWeights = minVarResult['fun'], minVarResult['x']
print("Minimum Portfolio Variance", minVar)
print("Optimal Weights", minVarResult['x'])

