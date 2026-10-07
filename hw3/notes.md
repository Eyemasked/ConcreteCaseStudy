Predictors: Cement, BlastFurnaceSlag, FlyAsh, Water, Superplasticizer, CoarseAggregate, FineAggregate, Age

Dependant Variable: Strength

# Observations

- Cement, vaguely positive linear
- Water, Coarse Aggregate, and Fine Aggregate are very similar

Doing backwards selection

# Steps

 
[OLD] Out[7]: ✓ Done
                            OLS Regression Results                            
==============================================================================
Dep. Variable:               Strength   R-squared:                       0.628
Model:                            OLS   Adj. R-squared:                  0.624
Method:                 Least Squares   F-statistic:                     139.5
Date:                Thu, 01 Oct 2026   Prob (F-statistic):          1.66e-136
Time:                        17:29:13   Log-Likelihood:                -2517.0
No. Observations:                 670   AIC:                             5052.
Df Residuals:                     661   BIC:                             5092.
Df Model:                           8                                         
Covariance Type:            nonrobust                                         
====================================================================================
                       coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------------
const               -7.3851     29.794     -0.248      0.804     -65.887      51.116
Cement               0.1169      0.010     11.831      0.000       0.097       0.136
BlastFurnaceSlag     0.1011      0.012      8.529      0.000       0.078       0.124
FlyAsh               0.0833      0.015      5.602      0.000       0.054       0.113
Water               -0.1662      0.045     -3.700      0.000      -0.254      -0.078
Superplasticizer     0.2766      0.111      2.497      0.013       0.059       0.494
CoarseAggregate      0.0131      0.011      1.233      0.218      -0.008       0.034
FineAggregate        0.0118      0.012      0.951      0.342      -0.013       0.036
Age                  0.1142      0.007     15.947      0.000       0.100       0.128
==============================================================================
Omnibus:                        6.466   Durbin-Watson:                   2.029
Prob(Omnibus):                  0.039   Jarque-Bera (JB):                6.585
Skew:                          -0.189   Prob(JB):                       0.0372
Kurtosis:                       3.305   Cond. No.                     9.55e+04
==============================================================================
Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The condition number is large, 9.55e+04. This might indicate that there are
strong multicollinearity or other numerical problems.




Out[_]: Never Run
                            OLS Regression Results                            
==============================================================================
Dep. Variable:               Strength   R-squared:                       0.627
Model:                            OLS   Adj. R-squared:                  0.624
Method:                 Least Squares   F-statistic:                     185.9
Date:                Thu, 01 Oct 2026   Prob (F-statistic):          1.90e-138
Time:                        17:29:13   Log-Likelihood:                -2517.7
No. Observations:                 670   AIC:                             5049.
Df Residuals:                     663   BIC:                             5081.
Df Model:                           6                                         
Covariance Type:            nonrobust                                         
====================================================================================
                       coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------------
const               26.3594      5.257      5.014      0.000      16.038      36.681
Cement               0.1080      0.005     20.686      0.000       0.098       0.118
BlastFurnaceSlag     0.0903      0.006     14.643      0.000       0.078       0.102
FlyAsh               0.0716      0.010      7.485      0.000       0.053       0.090
Water               -0.2085      0.026     -7.959      0.000      -0.260      -0.157
Superplasticizer     0.2367      0.102      2.316      0.021       0.036       0.437
Age                  0.1137      0.007     16.011      0.000       0.100       0.128
==============================================================================
Omnibus:                        6.533   Durbin-Watson:                   2.030
Prob(Omnibus):                  0.038   Jarque-Bera (JB):                6.627
Skew:                          -0.193   Prob(JB):                       0.0364
Kurtosis:                       3.298   Cond. No.                     4.68e+03
==============================================================================
Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The condition number is large, 4.68e+03. This might indicate that there are
strong multicollinearity or other numerical problems.
