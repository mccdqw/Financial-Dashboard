import numpy as np
import pandas as pd
import yfinance as yf
import scipy.optimize as sc
import plotly.graph_objects as go
import matplotlib.pyplot as plt


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

print("Portfolio returns:", pReturns)
print("Weights:", weights)
print("Portfolio downside deviation:", downsideDeviation)
print("Sortino ratio:", sortinoRatio)

print("-------------------------------------")

minVarResult = minimizeVarianceSortino(meanReturns, covMatrix)
minVar, minVarWeights = minVarResult['fun'], minVarResult['x']
print("Minimum Portfolio Variance", minVar)
print("Optimal Weights", minVarResult['x'])


def portfolioReturn(weights, meanReturns, covMatrix):
    return portfolioPerformance(weights, meanReturns, covMatrix)[0]
    

def efficientFrontierOpt(meanReturns, covMatrix, returnTarget, constraintSet=(0,1)):
    '''
        For each return target, we want to optimize the portfolio for minimum variance
    '''
    numAssets = len(meanReturns)
    args = (meanReturns, covMatrix)
    
    # we are only interested in the top half of the efficient frontier
    # above the min vol portfolio return
    constraints = ({'type': 'eq', 'fun': lambda x: portfolioReturn(x, meanReturns, covMatrix) - returnTarget},
                    {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple(constraintSet for asset in range(numAssets))
    efficientFrontierResult = sc.minimize(portfolioVarianceSharpe, numAssets*[1./numAssets], args=args,
                                          method='SLSQP', bounds=bounds, constraints=constraints)
    return efficientFrontierResult
    
    


def calculatedResults(meanReturns, covMatrix, riskFreeRate=0, constraintSet=(0,1)):
    # Read in mean, cov matrix, and other financial information
    # Ouput max sharpe ratio, min volatility and efficient frontier
    
    # Max Sharpe Ratio Portfolio
    maxSharpeRatioPortfolio = maxSharpeRatio(meanReturns, covMatrix)
    maxSharpeRatioReturns, maxSharpeRatioStd, maxSR = portfolioPerformance(maxSharpeRatioPortfolio['x'], meanReturns, covMatrix)
    maxSharpeRatioAllocation = pd.DataFrame(maxSharpeRatioPortfolio['x'], index=meanReturns.index, columns=['allocation'])
    maxSharpeRatioAllocation.allocation = [round(i*100, 0) for i in maxSharpeRatioAllocation.allocation]
    
    # Min Volatility Portfolio
    minVolPortfolio = minimizeVarianceSharpe(meanReturns, covMatrix)
    minVolReturns, minVolStd, minVol = portfolioPerformance(minVolPortfolio['x'], meanReturns, covMatrix)
    minVolAllocation = pd.DataFrame(minVolPortfolio['x'], index=meanReturns.index, columns=['allocation'])
    minVolAllocation.allocation = [round(i*100, 0) for i in minVolAllocation.allocation]
    
    # get the curve of returns
    
    efficientFrontierList = []
    targetReturns = np.linspace(minVolReturns, maxSharpeRatioReturns, 20)
    for target in targetReturns:
        # 'fun' returns the objective function value for the minimization problem
        efficientFrontierList.append(efficientFrontierOpt(meanReturns, covMatrix, target)['fun'])
        
    maxSharpeRatioReturns, maxSharpeRatioStd = round(maxSharpeRatioReturns*100, 2), round(maxSharpeRatioStd*100, 2)
    minVolReturns, minVolStd = round(minVolReturns*100, 2), round(minVolStd*100, 2)

    
    return maxSharpeRatioReturns, maxSharpeRatioStd, maxSharpeRatioAllocation, minVolReturns, minVolStd, minVolAllocation, efficientFrontierList, targetReturns

print(calculatedResults(meanReturns, covMatrix))

def efficientFrontierGraph(meanReturns, covMatrix, riskFreeRate=0, constraintSet=(0,1)):
    '''
        Return a graph plotting the min vol, max sharpe ratio and efficient frontier
    '''
    maxSharpeRatioReturns, maxSharpeRatioStd, maxSharpeRatioAllocation, minVolReturns, minVolStd, minVolAllocation, efficientFrontierList, targetReturns = calculatedResults(meanReturns, covMatrix, riskFreeRate, constraintSet)
    
    # Max Sharpe Ratio
    MaxSharpeRatio = go.Scatter(name='Maximum Sharpe Ratio', mode='markers',
                                x=[maxSharpeRatioStd], y=[maxSharpeRatioReturns],
                                marker=dict(color='red', size=14, line=dict(width=3, color='black')))
    MinVol = go.Scatter(name='Minimum Volatility', mode='markers',
                                x=[minVolStd], y=[minVolReturns],
                                marker=dict(color='green', size=14, line=dict(width=3, color='black')))
    EfficientFrontierCurve = go.Scatter(name='EfficientFrontier', mode='lines',
                                x=[round(efficientFrontierStd*100, 2) for efficientFrontierStd in efficientFrontierList], y=[round(target*100, 2) for target in targetReturns],
                                line=dict(color='black', width=4, dash='dashdot'))
    data = [MaxSharpeRatio, MinVol, EfficientFrontierCurve]
    
    layout = go.Layout(
            title='Portfolio Optimization with the Efficient Frontier',
            yaxis=dict(title='Annualized Return (%)'),
            xaxis=dict(title='Annualized Volatility (%)'),
            showlegend=True,
            legend=dict(
                x=0.75, y=0, traceorder='normal',
                bgcolor='#E2E2E2', bordercolor='black',
                borderwidth=2
            ),
            width=800,
            height=600
    )
    fig = go.Figure(data=data, layout=layout)
    return fig

fig = efficientFrontierGraph(meanReturns, covMatrix)
fig.write_image('figure.png')
plt.imshow(plt.imread('figure.png'))
plt.show()

