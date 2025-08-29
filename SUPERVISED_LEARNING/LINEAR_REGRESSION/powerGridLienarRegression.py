'''
The word problem :::

California ISO : manages power supply of California
- route power to needing regions
- predict power demands - in Megawatts - ahead in advance for regions
- hypothesis : power demand rises in summer

Integrate with NWS : National Weather Service - to gather weather-related data ( as our features ).

Feature = temperautre 
Labels = Historical power demand @ regional levels 

Solve LOB : line of best fit
Run LOB calculations.

Can we learn a trend, even with noise values in the data?
Is coefficient reasonable?
- check p-values ( p < 5% ).
- low chance of coeff being due to randomness : statistically-significant

Coeffs are just estimates : based on samples.
Need confidence interval : (3100,9500)
    -> 95% chance that a value lies in an interval

    


Correlation is quantifiable ( R) .
R^2 ( line fits so well ).
Correlation ain't causation ( doesn't tell X influences Y ).

How well did we explain our variance? The goal of a high R^2 term.



'''
import numpy as np




# celsiusAvgDailyTemps = np.random.randn(100)
# megaWattPowerDemand = np.random.randn(100)
# numSamples = 300
# endpointStatus = True
# celsiusAvgDailyTemps = np.linspace(60,90,numSamples,endpointStatus)
# megaWattPowerDemand = np.linspace(100,300,numSamples,endpointStatus)

# print(celsiusAvgDailyTemps)
# print(megaWattPowerDemand)

# Plot {X1,X2} data


# sum(x-xHat)(y-yHat)/sum(
# b : avg y, avg x <-- have a
# Solve `m` -> then solve `b`?



# Solve for our R^2 values
# n = len(celsiusAvgDailyTemps)

# Solve for t-statistic with t-distributions
# Areas under curve : obtain p-values


# Features as independent variables, label as dependent variable.
# What features influence power demand?
# Goal : read a feature dataset.
# One hot-encoding : only if a single category at a time ( not multiple categories at same time )
# one state as `0` for OHE <--- save on a single bit :-)

# How to handle correlated features ( vs non-correlated )
# Curse of (multi)collinearity
# Perf no degradation, but change coeffs interpretation.


# VIF <- variance inflation factor computation
# >= 5 : severe collinearity
# Centering of features : mitigation strategy
# subtract(avg) reduce VIF

# Self multiply the features
# Winter-months modeling.
# square temperature -> can fit a curve ( equation ) or a cubic ( bends )
# more powers = more bends in the model/LOBs
# don't put to many terms :-( overfitting of data.
# coef*x ( linearReg ) coeff*x*coeff ( nonLinearReg )

# Avoid simpsons paradox
# Region expansion of model : large->med->small : added data -> model learned to differentiate
# If we didn't add knowledge -> we learn a negative line ( gaaaah ) 
# Increase(dimensions) to model to segment data to avoid simpson's paradox.
# Data segmentation avoids simpson's paradox

# Stats model ( lin reg package )
# Multiple regression,r-sq,p-vals,VIF
# Closed-form linear regression : n^3
# SVD for linReg more desired.


















