#!/usr/bin/env python
# coding: utf-8

# In[1]:


### FITTING MULTI LINEAR REGRESSION MODEL FOR MICROINSURANCE DATASET


# In[2]:


## Modules required
import pandas as pd
import seaborn as sns
import numpy as np
import pylab
import math
import matplotlib.pyplot as plt


# In[3]:


from scipy import stats
import statsmodels.api as sm
from statsmodels.stats import diagnostic as diag
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


## Load the dataset into pandas
MICDATA=pd.read_excel('MIP.xlsx')

## set the index equal to the year column
MICDATA.index = MICDATA['Year']
MICDATA = MICDATA.drop(['Year', 'CCODE'], axis = 1)
MICDATA.head()


# In[5]:


## Get the summary of our original data set
desc_MICDATA = MICDATA.describe()

## Add the standard deviation metric
desc_MICDATA.loc['+3_std']=desc_MICDATA.loc['mean']+(desc_MICDATA.loc['std']*3)
desc_MICDATA.loc['-3_std']=desc_MICDATA.loc['mean']-(desc_MICDATA.loc['std']*3)
desc_MICDATA


# In[6]:


## Data preprocessing ##
## How is the distribution of the dependent variables?


# In[7]:


## Condisder GDPCAPITA
GDPCAPITA = MICDATA.GDPCAPITA

pd.Series(GDPCAPITA).hist()
plt.show()


stats.probplot(GDPCAPITA, dist="norm", plot=pylab)
pylab.show()


# In[8]:


## Performing data transformation on this variable for normality
GDPCAPITA_bc, lmda = stats.boxcox(GDPCAPITA)
pd.Series(GDPCAPITA_bc).hist()
plt.show()

stats.probplot(GDPCAPITA_bc, dist = "norm", plot=pylab)
pylab.show()
print("lambda parameter for Box-Cox Transformation is {}".format(lmda))


# In[ ]:





# In[9]:


## Condisder EODB
EODB = MICDATA.EODB

pd.Series(EODB).hist()
plt.show()


stats.probplot(EODB, dist="norm", plot=pylab)
pylab.show()


# In[10]:


## Performing data transformation on this variable for normality
EODB_bc, lmda = stats.boxcox(EODB)
pd.Series(EODB_bc).hist()
plt.show()

stats.probplot(EODB_bc, dist= "norm", plot=pylab)
pylab.show()
print("lambda parameter for Box-Cox Transformation is {}".format(lmda))


# In[11]:


MICDATA["GDPCAPITA"] = GDPCAPITA_bc
MICDATA["EODB"] = EODB_bc


# In[12]:


## Checking the Model Assumptions
######## Multicolinearity #################
## printing out correlation matrix of the data frame
corr=MICDATA.corr()

## Display the correlation matrix
display(corr)


# In[13]:


## plot a heatmap
sns.heatmap(corr, xticklabels = corr.columns, yticklabels = corr.columns, cmap="RdBu")


# In[15]:


### Using the VIF to measure to detect the above and dropping all variable with greater than 10 VIF
MICDATA_before = MICDATA
MICDATA_after = MICDATA.drop(['INSP','INSDENS','FINSF','FISFRED'], axis = 1)


x1 = sm.tools.add_constant(MICDATA_before)
x2 = sm.tools.add_constant(MICDATA_after)

#Create a series for both

series_before = pd.Series([variance_inflation_factor(x1.values, i) for i in range(x1.shape[1])], index = x1.columns)
series_after = pd.Series([variance_inflation_factor(x2.values, i) for i in range(x2.shape[1])], index = x2.columns)

## dispay the series
print('DATA BEFORE')
print('-'*100)
display(series_before)


print('DATA AFTER')
print('-'*100)
display(series_after)


# In[16]:


MICDATA_after


# In[17]:


#### Building the model ####
## considering GDP PER CAPITA as our dependent Variable ##
## ## Full Model, MODEL 4
## define our input variable and our output variable where ###
x = MICDATA_after.drop(['GDPCAPITA', 'EODB'], axis = 1)
y = MICDATA_after[['GDPCAPITA']]


# In[18]:


y


# In[19]:


x


# In[20]:


## Split dataset into training and testing portion
from sklearn.model_selection import train_test_split
import numpy as np

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 1)

## Scale the independent variables gives
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
import numpy as np

min_max_scaler= preprocessing.MinMaxScaler()
x_train_minmax = min_max_scaler.fit_transform(x_train)
x_test_minmax = min_max_scaler.fit_transform(x_test)

 


# In[21]:


x_train = x_train_minmax
x_test= x_test_minmax 


# In[22]:


## Create an instance of our model
regression_model = LinearRegression()

## Full Model
## Fit the model
regression_model.fit(x_train, y_train)


# In[23]:


##Grab the intercept and the coeffitients
intercept = regression_model.intercept_[0]
coef = regression_model.coef_[0]

print("The intercept for our model is {:.4}".format(intercept))
print('_'*100)

## Loop through dictionary and print the data
for cf in zip(x.columns, coef):
    print("The Coeffitient for {} is {:.4}".format(cf[0],cf[1]))


# In[24]:


## Getting multiple prediction
y_predict = regression_model.predict(x_test)

## Show the first five
y_predict[:5]


# In[25]:


## Evaluating the model
import statsmodels.api as sm
from statsmodels.stats import diagnostic as diag
from statsmodels.stats.outliers_influence import variance_inflation_factor

## Define our input variable
x2 = sm.add_constant(x)

## Create an OLS model
model = sm.OLS(y, x2)

## fit the data
est = model.fit()


# In[26]:


## Testing the Model Assumptions
# Heteroscedasticity using the Breusch-Pegan test

#H0:σ2=σ2
#H1:σ2!=σ2

## Grab the p-values 
_, pval, _, f_pval = diag.het_breuschpagan(est.resid, est.model.exog)
print(pval, f_pval)
print('_'*100)
          
if pval > 0.05:
    print("For the Breusch Pagan's Test")
    print("The p-value was {:.4}".format(pval))
    print("we fail to reject the null hypothesis, and conclude that there is no heteroscedasticity.")
else:
    print("For the Breusch Pagan's Test")
    print("The p-value was {:.4}".format(pval))
    print("we reject the null hypothesis, and conclude that there is heteroscedasticity.")


# In[27]:


### Checking for Autocorrelation using the Ljungbox test
#H0: The data are random
#H1: The data are not random

## Calculate the lag
lag = min(10, (len(x)//5))
print('The number of lags will be {}'.format(lag))
print('_'*100)

## Perform the test          
test_results = diag.acorr_ljungbox(est.resid, lags = lag)
 
## print the result for the test
print(test_results)

## Grab the P-Value and the test statistics
ibvalue, p_val = test_results          
          

## print the result for the test
if min(p_val) > 0.05:
    print("The lowest p-value found was {:.4}".format(min(p_val)))
    print("we fail to reject the null hypothesis, and conclude that there is no Autocorrelation.")
    print('_'*100)
else:
    print("The lowest p-value found was {:.4}".format(min(p_val)))
    print("we reject the null hypothesis, and conclude that there is Autocorrelation.")
    print('_'*100)
    
## Plotting Autocorrelation
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats import diagnostic as diag

sm.graphics.tsa.plot_acf(est.resid)
plt.show()


# In[28]:


## Check for Linearity of the residuals using the Q-Q plot
import pylab
sm.qqplot(est.resid, line = 's')
pylab.show()

## Checking that mean of the residuals is approximately zero
mean_residuals = sum(est.resid)/len(est.resid)
mean_residuals


# In[29]:


import math
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
get_ipython().run_line_magic('matplotlib', 'inline')

## Calculating the mean square Error
model_mse = mean_squared_error(y_test, y_predict)


## Calculate the mean absolute error
model_mae = mean_absolute_error(y_test, y_predict)

## Calculate the root mean squared error
model_rmse = math.sqrt(model_mse)


## Display the output
print("MSE {:.3}".format(model_mse))
print("MAE {:.3}".format(model_mae))
print("RMSE {:.3}".format(model_rmse))


# In[30]:


## Model summary
print(est.summary())


# In[ ]:





# In[31]:


#### Building the model ####
## considering Demographic Variables on GDPCAPITA as our dependent Variable ##

## define our input variable and our output variable where ###

x = MICDATA_after.drop(['GDPCAPITA','EODB','NETIPC','GNIPC','INFLAT','OPOE','REALIR','GDPOMT','PROPR','BUSFRD','INFREED','GOVSIZE','FRAGSTAT','RULELAW','LABOURFRE'], axis = 1)
y = MICDATA_after[['GDPCAPITA']]


# In[32]:


## Split dataset into training and testing portion
from sklearn.model_selection import train_test_split
import numpy as np

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 1)


# In[33]:


## Scale the independent variables gives
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
import numpy as np

min_max_scaler= preprocessing.MinMaxScaler()
x_train_minmax = min_max_scaler.fit_transform(x_train)
x_test_minmax = min_max_scaler.fit_transform(x_test)


# In[34]:


x_train = x_train_minmax
x_test= x_test_minmax 


# In[35]:


## Create an instance of our model
regression_model1 = LinearRegression()

## Fit the model
regression_model1.fit(x_train, y_train)


# In[36]:


##Grab the intercept and the coeffitients
intercept = regression_model1.intercept_[0]
coef = regression_model1.coef_[0]

print("The intercept for our model is {:.4}".format(intercept))
print('_'*100)

## Loop through dictionary and print the data
for cf in zip(x.columns, coef):
    print("The Coeffitient for {} is {:.4}".format(cf[0],cf[1]))


# In[37]:


## Getting multiple prediction
y_predict = regression_model1.predict(x_test)

## Show the first five
y_predict[:5]


# In[38]:


## Evaluating the model
import statsmodels.api as sm
from statsmodels.stats import diagnostic as diag
from statsmodels.stats.outliers_influence import variance_inflation_factor

## Define our input variable
x2 = sm.add_constant(x)

## Create an OLS model
model = sm.OLS(y, x2)

## fit the data
est = model.fit()


# In[39]:


import math
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
get_ipython().run_line_magic('matplotlib', 'inline')

## Calculating the mean square Error
model_mse = mean_squared_error(y_test, y_predict)


## Calculate the mean absolute error
model_mae = mean_absolute_error(y_test, y_predict)

## Calculate the root mean squared error
model_rmse = math.sqrt(model_mse)


## Display the output
print("MSE {:.3}".format(model_mse))
print("MAE {:.3}".format(model_mae))
print("RMSE {:.3}".format(model_rmse))


# In[40]:


## Model summary
print(est.summary())


# In[ ]:





# In[41]:


## MODEL 2 Economic Variables on GDPCAPITA


# In[42]:


#### Building the model ####

## define our input variable and our output variable where ###
x = MICDATA.drop(['GDPCAPITA', 'EODB','PPLRA','PGR','ADR','PSIZE','PPOOR','PIUI', 'FISFRED','BUSFRD','OPOE','FINSF','PROPR','INFREED', 'FRAGSTAT','LABOURFRE','GOVSIZE' ], axis = 1)
y = MICDATA['GDPCAPITA']


# In[43]:


## Split dataset into training and testing portion
from sklearn.model_selection import train_test_split
import numpy as np

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 1)

 


# In[44]:


## Scale the independent variables gives
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
import numpy as np

min_max_scaler= preprocessing.MinMaxScaler()
x_train_minmax = min_max_scaler.fit_transform(x_train)
x_test_minmax = min_max_scaler.fit_transform(x_test)


# In[45]:


x_train = x_train_minmax
x_test= x_test_minmax 


# In[46]:


## Create an instance of our model
regression_model2 = LinearRegression()

## Fit the model
regression_model2.fit(x_train, y_train)


# In[47]:


## Getting multiple prediction
y_predict = regression_model2.predict(x_test)

## Show the first five
y_predict[:5]


# In[48]:


## Evaluating the model
import statsmodels.api as sm
from statsmodels.stats import diagnostic as diag
from statsmodels.stats.outliers_influence import variance_inflation_factor

## Define our input variable
x2 = sm.add_constant(x)

## Create an OLS model
model = sm.OLS(y, x2)

## fit the data
est = model.fit()


# In[49]:


import math
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
get_ipython().run_line_magic('matplotlib', 'inline')

## Calculating the mean square Error
model_mse = mean_squared_error(y_test, y_predict)


## Calculate the mean absolute error
model_mae = mean_absolute_error(y_test, y_predict)

## Calculate the root mean squared error
model_rmse = math.sqrt(model_mse)


## Display the output
print("MSE {:.3}".format(model_mse))
print("MAE {:.3}".format(model_mae))
print("RMSE {:.3}".format(model_rmse))


# In[50]:


## Model summary
print(est.summary())


# In[ ]:





# In[51]:


## MODEL 3 Institutional Variables on GDPCAPITA


# In[52]:


#### Building the model ####

## define our input variable and our output variable where ###
x = MICDATA.drop(['GDPCAPITA','ADR', 'PSIZE','PGR', 'PPOOR','MTSUBS','PIUI','PPLRA','EODB','NETIPC','GNIPC','INFLAT','REALIR','GDPOMT','INSP','INSDENS','RULELAW'], axis = 1)
y = MICDATA['GDPCAPITA']


# In[53]:


## Split dataset into training and testing portion
from sklearn.model_selection import train_test_split
import numpy as np

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 1)


# In[54]:


## Scale the independent variables gives
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
import numpy as np

min_max_scaler= preprocessing.MinMaxScaler()
x_train_minmax = min_max_scaler.fit_transform(x_train)
x_test_minmax = min_max_scaler.fit_transform(x_test)


# In[55]:


x_train = x_train_minmax
x_test= x_test_minmax


# In[56]:


## Create an instance of our model
regression_model3 = LinearRegression()

## Fit the model
regression_model3.fit(x_train, y_train)


# In[57]:


## Getting multiple prediction
y_predict = regression_model3.predict(x_test)

## Show the first five
y_predict[:5]


# In[58]:


## Evaluating the model
import statsmodels.api as sm
from statsmodels.stats import diagnostic as diag
from statsmodels.stats.outliers_influence import variance_inflation_factor

## Define our input variable
x2 = sm.add_constant(x)

## Create an OLS model
model = sm.OLS(y, x2)

## fit the data
est = model.fit()


# In[59]:


import math
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
get_ipython().run_line_magic('matplotlib', 'inline')

## Calculating the mean square Error
model_mse = mean_squared_error(y_test, y_predict)


## Calculate the mean absolute error
model_mae = mean_absolute_error(y_test, y_predict)

## Calculate the root mean squared error
model_rmse = math.sqrt(model_mse)


## Display the output
print("MSE {:.3}".format(model_mse))
print("MAE {:.3}".format(model_mae))
print("RMSE {:.3}".format(model_rmse))


# In[60]:


## Model summary
print(est.summary())


# In[ ]:





# In[ ]:





# In[61]:


## Ease of Doing Business as Dependent Variable


# In[62]:


#### Building the model ####
## ## Full Model, MODEL 4
## define our input variable and our output variable where ###
x = MICDATA.drop(['GDPCAPITA', 'EODB'], axis = 1)
y = MICDATA['EODB']


# In[63]:


## Split dataset into training and testing portion
from sklearn.model_selection import train_test_split
import numpy as np

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 1)

## Scale the independent variables gives
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
import numpy as np

min_max_scaler= preprocessing.MinMaxScaler()
x_train_minmax = min_max_scaler.fit_transform(x_train)
x_test_minmax = min_max_scaler.fit_transform(x_test)


# In[64]:


x_train = x_train_minmax
x_test= x_test_minmax


# In[65]:


## Create an instance of our model
regression_model = LinearRegression()

## Full Model
## Fit the model
regression_model.fit(x_train, y_train)


# In[66]:


## Getting multiple prediction
y_predict = regression_model.predict(x_test)

## Show the first five
y_predict[:5]


# In[67]:


## Evaluating the model
import statsmodels.api as sm
from statsmodels.stats import diagnostic as diag
from statsmodels.stats.outliers_influence import variance_inflation_factor

## Define our input variable
x2 = sm.add_constant(x)

## Create an OLS model
model = sm.OLS(y, x2)

## fit the data
est = model.fit()


# In[68]:


## Testing the Model Assumptions
# Heteroscedasticity using the Breusch-Pegan test

#H0:σ2=σ2
#H1:σ2!=σ2

## Grab the p-values 
_, pval, _, f_pval = diag.het_breuschpagan(est.resid, est.model.exog)
print(pval, f_pval)
print('_'*100)
          
if pval > 0.05:
    print("For the Breusch Pagan's Test")
    print("The p-value was {:.4}".format(pval))
    print("we fail to reject the null hypothesis, and conclude that there is no heteroscedasticity.")
else:
    print("For the Breusch Pagan's Test")
    print("The p-value was {:.4}".format(pval))
    print("we reject the null hypothesis, and conclude that there is heteroscedasticity.")


# In[69]:


### Checking for Autocorrelation using the Ljungbox test
#H0: The data are random
#H1: The data are not random

## Calculate the lag
lag = min(10, (len(x)//5))
print('The number of lags will be {}'.format(lag))
print('_'*100)

## Perform the test          
test_results = diag.acorr_ljungbox(est.resid, lags = lag)
 
## print the result for the test
print(test_results)

## Grab the P-Value and the test statistics
ibvalue, p_val = test_results          
          

## print the result for the test
if min(p_val) > 0.05:
    print("The lowest p-value found was {:.4}".format(min(p_val)))
    print("we fail to reject the null hypothesis, and conclude that there is no Autocorrelation.")
    print('_'*100)
else:
    print("The lowest p-value found was {:.4}".format(min(p_val)))
    print("we reject the null hypothesis, and conclude that there is Autocorrelation.")
    print('_'*100)
    
## Plotting Autocorrelation
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats import diagnostic as diag

sm.graphics.tsa.plot_acf(est.resid)
plt.show()


# In[70]:


## Check for Linearity of the residuals using the Q-Q plot
import pylab
sm.qqplot(est.resid, line = 's')
pylab.show()

## Checking that mean of the residuals is approximately zero
mean_residuals = sum(est.resid)/len(est.resid)
mean_residuals


# In[71]:


import math
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
get_ipython().run_line_magic('matplotlib', 'inline')

## Calculating the mean square Error
model_mse = mean_squared_error(y_test, y_predict)


## Calculate the mean absolute error
model_mae = mean_absolute_error(y_test, y_predict)

## Calculate the root mean squared error
model_rmse = math.sqrt(model_mse)


## Display the output
print("MSE {:.3}".format(model_mse))
print("MAE {:.3}".format(model_mae))
print("RMSE {:.3}".format(model_rmse))


# In[72]:


## Model summary
print(est.summary())


# In[ ]:





# In[73]:


## Model one(1) with Demographic variables on EODB
rt


# In[74]:


#### Building the model ####

## define our input variable and our output variable where ###
x = MICDATA_after.drop(['GDPCAPITA','EODB','NETIPC','GNIPC','INFLAT','OPOE','REALIR','GDPOMT','PROPR','BUSFRD','INFREED','GOVSIZE','FRAGSTAT','RULELAW','LABOURFRE'], axis = 1)
y = MICDATA_after[['EODB']]


# In[75]:


## Split dataset into training and testing portion
from sklearn.model_selection import train_test_split
import numpy as np

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 1)


# In[76]:


## Scale the independent variables gives
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
import numpy as np

min_max_scaler= preprocessing.MinMaxScaler()
x_train_minmax = min_max_scaler.fit_transform(x_train)
x_test_minmax = min_max_scaler.fit_transform(x_test)


# In[77]:


x_train = x_train_minmax
x_test= x_test_minmax


# In[78]:


## Create an instance of our model
regression_model1 = LinearRegression()

## Fit the model
regression_model1.fit(x_train, y_train)


# In[79]:


##Grab the intercept and the coeffitients
intercept = regression_model1.intercept_[0]
coef = regression_model1.coef_[0]

print("The intercept for our model is {:.4}".format(intercept))
print('_'*100)

## Loop through dictionary and print the data
for cf in zip(x.columns, coef):
    print("The Coeffitient for {} is {:.4}".format(cf[0],cf[1]))


# In[80]:


## Getting multiple prediction
y_predict = regression_model1.predict(x_test)

## Show the first five
y_predict[:5]


# In[81]:


## Evaluating the model
import statsmodels.api as sm
from statsmodels.stats import diagnostic as diag
from statsmodels.stats.outliers_influence import variance_inflation_factor

## Define our input variable
x2 = sm.add_constant(x)

## Create an OLS model
model = sm.OLS(y, x2)

## fit the data
est = model.fit()


# In[82]:


import math
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
get_ipython().run_line_magic('matplotlib', 'inline')

## Calculating the mean square Error
model_mse = mean_squared_error(y_test, y_predict)


## Calculate the mean absolute error
model_mae = mean_absolute_error(y_test, y_predict)

## Calculate the root mean squared error
model_rmse = math.sqrt(model_mse)


## Display the output
print("MSE {:.3}".format(model_mse))
print("MAE {:.3}".format(model_mae))
print("RMSE {:.3}".format(model_rmse))


# In[83]:


## Model summary
print(est.summary())


# In[ ]:





# In[84]:


## MODEL 2 ECONOMIC variables on EODB as the Dependent Variables


# In[85]:


#### Building the model ####

## define our input variable and our output variable where ###
x = MICDATA.drop(['GDPCAPITA', 'EODB','PPLRA','PGR','ADR','PSIZE','PPOOR','PIUI', 'FISFRED','BUSFRD','OPOE','PROPR','INFREED', 'FRAGSTAT','LABOURFRE','GOVSIZE' ], axis = 1)
y = MICDATA['EODB']


# In[86]:


## Split dataset into training and testing portion
from sklearn.model_selection import train_test_split
import numpy as np

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 1)


# In[87]:


## Scale the independent variables gives
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
import numpy as np

min_max_scaler= preprocessing.MinMaxScaler()
x_train_minmax = min_max_scaler.fit_transform(x_train)
x_test_minmax = min_max_scaler.fit_transform(x_test)


# In[88]:


x_train = x_train_minmax
x_test= x_test_minmax


# In[89]:


## Create an instance of our model
regression_model2 = LinearRegression()

## Fit the model
regression_model2.fit(x_train, y_train)


# In[90]:


## Getting multiple prediction
y_predict = regression_model2.predict(x_test)

## Show the first five
y_predict[:5]


# In[91]:


## Evaluating the model
import statsmodels.api as sm
from statsmodels.stats import diagnostic as diag
from statsmodels.stats.outliers_influence import variance_inflation_factor

## Define our input variable
x2 = sm.add_constant(x)

## Create an OLS model
model = sm.OLS(y, x2)

## fit the data
est = model.fit()


# In[92]:


import math
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
get_ipython().run_line_magic('matplotlib', 'inline')

## Calculating the mean square Error
model_mse = mean_squared_error(y_test, y_predict)


## Calculate the mean absolute error
model_mae = mean_absolute_error(y_test, y_predict)

## Calculate the root mean squared error
model_rmse = math.sqrt(model_mse)


## Display the output
print("MSE {:.3}".format(model_mse))
print("MAE {:.3}".format(model_mae))
print("RMSE {:.3}".format(model_rmse))


# In[93]:


## Model summary
print(est.summary())


# In[ ]:





# In[94]:


## MODEL 3 Institutional Variables on EODB


# In[95]:


#### Building the model ####

## define our input variable and our output variable where ###
x = MICDATA.drop(['GDPCAPITA','EODB','ADR', 'PSIZE','PGR', 'PPOOR','MTSUBS','PIUI','PPLRA','NETIPC','GNIPC','INFLAT','REALIR','GDPOMT','INSP','INSDENS','RULELAW'], axis = 1)
y = MICDATA['EODB']


# In[96]:


## Split dataset into training and testing portion
from sklearn.model_selection import train_test_split
import numpy as np

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 1)


# In[97]:


## Scale the independent variables gives
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
import numpy as np

min_max_scaler= preprocessing.MinMaxScaler()
x_train_minmax = min_max_scaler.fit_transform(x_train)
x_test_minmax = min_max_scaler.fit_transform(x_test)


# In[98]:


x_train = x_train_minmax
x_test= x_test_minmax


# In[99]:


## Create an instance of our model
regression_model3 = LinearRegression()

## Fit the model
regression_model3.fit(x_train, y_train)


# In[100]:


## Getting multiple prediction
y_predict = regression_model3.predict(x_test)

## Show the first five
y_predict[:5]


# In[101]:


## Evaluating the model
import statsmodels.api as sm
from statsmodels.stats import diagnostic as diag
from statsmodels.stats.outliers_influence import variance_inflation_factor

## Define our input variable
x2 = sm.add_constant(x)

## Create an OLS model
model = sm.OLS(y, x2)

## fit the data
est = model.fit()


# In[102]:


import math
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
get_ipython().run_line_magic('matplotlib', 'inline')

## Calculating the mean square Error
model_mse = mean_squared_error(y_test, y_predict)


## Calculate the mean absolute error
model_mae = mean_absolute_error(y_test, y_predict)

## Calculate the root mean squared error
model_rmse = math.sqrt(model_mse)


## Display the output
print("MSE {:.3}".format(model_mse))
print("MAE {:.3}".format(model_mae))
print("RMSE {:.3}".format(model_rmse))


# In[103]:


## Model summary
print(est.summary())


# In[ ]:





# In[ ]:





# In[ ]:





# In[1]:


## Microinsurance Density the Dependent Variable


# In[2]:


## Modules required
import pandas as pd
import seaborn as sns
import numpy as np
import pylab
import math
import matplotlib.pyplot as plt


# In[3]:


from scipy import stats
import statsmodels.api as sm
from statsmodels.stats import diagnostic as diag
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


## Load the dataset into pandas
MICDATA=pd.read_excel('INSDENSITY.xlsx')

## set the index equal to the year column
MICDATA.index = MICDATA['Year']
MICDATA = MICDATA.drop(['Year', 'CCODE','Premiums'], axis = 1)
MICDATA.head()


# In[5]:


## Get the summary of our original data set
desc_MICDATA = MICDATA.describe()

## Add the standard deviation metric
desc_MICDATA.loc['+3_std']=desc_MICDATA.loc['mean']+(desc_MICDATA.loc['std']*3)
desc_MICDATA.loc['-3_std']=desc_MICDATA.loc['mean']-(desc_MICDATA.loc['std']*3)
desc_MICDATA


# In[6]:


## Data preprocessing ##
## How is the distribution of the dependent variables?


# In[7]:


## Condisder GDPCAPITA
MICROIDENSITY = MICDATA.MICROIDENSITY

pd.Series(MICROIDENSITY).hist()
plt.show()


stats.probplot(MICROIDENSITY, dist="norm", plot=pylab)
pylab.show()


# In[8]:


## Performing data transformation on this variable for normality
MICROIDENSITY_bc, lmda = stats.boxcox(MICROIDENSITY)
pd.Series(MICROIDENSITY_bc).hist()
plt.show()

stats.probplot(MICROIDENSITY_bc, dist = "norm", plot=pylab)
pylab.show()
print("lambda parameter for Box-Cox Transformation is {}".format(lmda))


# In[9]:


MICDATA["MICROIDENSITY"] = MICROIDENSITY_bc


# In[10]:


## Checking the Model Assumptions
######## Multicolinearity #################
## printing out correlation matrix of the data frame
corr=MICDATA.corr()

## Display the correlation matrix
display(corr)


# In[11]:


## plot a heatmap
sns.heatmap(corr, xticklabels = corr.columns, yticklabels = corr.columns, cmap="RdBu")


# In[12]:


### Using the VIF to measure to detect the above and dropping all variable with greater than 10 VIF
MICDATA_before = MICDATA
MICDATA_after = MICDATA.drop(['GNIPC','INSP','FISFRED','FINSF','PROPR','OPOE','INSDENS'], axis = 1)


x1 = sm.tools.add_constant(MICDATA_before)
x2 = sm.tools.add_constant(MICDATA_after)

#Create a series for both

series_before = pd.Series([variance_inflation_factor(x1.values, i) for i in range(x1.shape[1])], index = x1.columns)
series_after = pd.Series([variance_inflation_factor(x2.values, i) for i in range(x2.shape[1])], index = x2.columns)

## dispay the series
print('DATA BEFORE')
print('-'*100)
display(series_before)


print('DATA AFTER')
print('-'*100)
display(series_after)


# In[13]:


MICDATA_after.head()


# In[14]:


#### Building the model ####
## ## Full Model, MODEL 4
## define our input variable and our output variable where ###
x = MICDATA_after.drop(['GDPCAPITA', 'EODB','MICROIDENSITY'], axis = 1)
y = MICDATA_after['MICROIDENSITY']


# In[15]:


## Split dataset into training and testing portion
from sklearn.model_selection import train_test_split
import numpy as np

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 1)

## Scale the independent variables gives
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
import numpy as np

min_max_scaler= preprocessing.MinMaxScaler()
x_train_minmax = min_max_scaler.fit_transform(x_train)
x_test_minmax = min_max_scaler.fit_transform(x_test)


# In[16]:


x_train = x_train_minmax
x_test= x_test_minmax


# In[17]:


## Create an instance of our model
regression_model = LinearRegression()

## Full Model
## Fit the model
regression_model.fit(x_train, y_train)


# In[18]:


## Getting multiple prediction
y_predict = regression_model.predict(x_test)

## Show the first five
y_predict[:5]


# In[19]:


## Evaluating the model
import statsmodels.api as sm
from statsmodels.stats import diagnostic as diag
from statsmodels.stats.outliers_influence import variance_inflation_factor

## Define our input variable
x2 = sm.add_constant(x)

## Create an OLS model
model = sm.OLS(y, x2)

## fit the data
est = model.fit()


# In[20]:


## Testing the Model Assumptions
# Heteroscedasticity using the Breusch-Pegan test

#H0:σ2=σ2
#H1:σ2!=σ2

## Grab the p-values 
_, pval, _, f_pval = diag.het_breuschpagan(est.resid, est.model.exog)
print(pval, f_pval)
print('_'*100)
          
if pval > 0.05:
    print("For the Breusch Pagan's Test")
    print("The p-value was {:.4}".format(pval))
    print("we fail to reject the null hypothesis, and conclude that there is no heteroscedasticity.")
else:
    print("For the Breusch Pagan's Test")
    print("The p-value was {:.4}".format(pval))
    print("we reject the null hypothesis, and conclude that there is heteroscedasticity.")


# In[21]:


### Checking for Autocorrelation using the Ljungbox test
#H0: The data are random
#H1: The data are not random

## Calculate the lag
lag = min(10, (len(x)//5))
print('The number of lags will be {}'.format(lag))
print('_'*100)

## Perform the test          
test_results = diag.acorr_ljungbox(est.resid, lags = lag)
 
## print the result for the test
print(test_results)

## Grab the P-Value and the test statistics
ibvalue, p_val = test_results          
          

## print the result for the test
if min(p_val) > 0.05:
    print("The lowest p-value found was {:.4}".format(min(p_val)))
    print("we fail to reject the null hypothesis, and conclude that there is no Autocorrelation.")
    print('_'*100)
else:
    print("The lowest p-value found was {:.4}".format(min(p_val)))
    print("we reject the null hypothesis, and conclude that there is Autocorrelation.")
    print('_'*100)
    
## Plotting Autocorrelation
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats import diagnostic as diag

sm.graphics.tsa.plot_acf(est.resid)
plt.show()


# In[22]:


## Check for Linearity of the residuals using the Q-Q plot
import pylab
sm.qqplot(est.resid, line = 's')
pylab.show()

## Checking that mean of the residuals is approximately zero
mean_residuals = sum(est.resid)/len(est.resid)
mean_residuals


# In[23]:


import math
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
get_ipython().run_line_magic('matplotlib', 'inline')

## Calculating the mean square Error
model_mse = mean_squared_error(y_test, y_predict)


## Calculate the mean absolute error
model_mae = mean_absolute_error(y_test, y_predict)

## Calculate the root mean squared error
model_rmse = math.sqrt(model_mse)


## Display the output
print("MSE {:.3}".format(model_mse))
print("MAE {:.3}".format(model_mae))
print("RMSE {:.3}".format(model_rmse))


# In[24]:


## Model summary
print(est.summary())


# In[ ]:





# In[25]:


## Model one(1) with Demographic variables on MICROIDENSITY


# In[26]:


#### Building the model ####

## define our input variable and our output variable where ###
x = MICDATA_after.drop(['MICROIDENSITY','GDPCAPITA','EODB','NETIPC','INFREED','INFLAT','REALIR','GDPOMT','BUSFRD','GOVSIZE','FRAGSTAT','RULELAW','LABOURFRE'], axis = 1)
y = MICDATA_after[['MICROIDENSITY']]


# In[27]:


## Split dataset into training and testing portion
from sklearn.model_selection import train_test_split
import numpy as np

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 1)


# In[28]:


## Scale the independent variables gives
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
import numpy as np

min_max_scaler= preprocessing.MinMaxScaler()
x_train_minmax = min_max_scaler.fit_transform(x_train)
x_test_minmax = min_max_scaler.fit_transform(x_test)


# In[29]:


x_train = x_train_minmax
x_test= x_test_minmax


# In[30]:


## Create an instance of our model
regression_model1 = LinearRegression()

## Fit the model
regression_model1.fit(x_train, y_train)


# In[31]:


#Grab the intercept and the coeffitients
intercept = regression_model1.intercept_[0]
coef = regression_model1.coef_[0]

print("The intercept for our model is {:.4}".format(intercept))
print('_'*100)

## Loop through dictionary and print the data
for cf in zip(x.columns, coef):
    print("The Coeffitient for {} is {:.4}".format(cf[0],cf[1]))


# In[32]:


## Getting multiple prediction
y_predict = regression_model1.predict(x_test)

## Show the first five
y_predict[:5]


# In[33]:


## Evaluating the model
import statsmodels.api as sm
from statsmodels.stats import diagnostic as diag
from statsmodels.stats.outliers_influence import variance_inflation_factor

## Define our input variable
x2 = sm.add_constant(x)

## Create an OLS model
model = sm.OLS(y, x2)

## fit the data
est = model.fit()


# In[34]:


import math
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
get_ipython().run_line_magic('matplotlib', 'inline')

## Calculating the mean square Error
model_mse = mean_squared_error(y_test, y_predict)


## Calculate the mean absolute error
model_mae = mean_absolute_error(y_test, y_predict)

## Calculate the root mean squared error
model_rmse = math.sqrt(model_mse)


## Display the output
print("MSE {:.3}".format(model_mse))
print("MAE {:.3}".format(model_mae))
print("RMSE {:.3}".format(model_rmse))


# In[35]:


## Model summary
print(est.summary())


# In[ ]:





# In[36]:


## MODEL 2 Economic Variables on MICROIDENSITY


# In[37]:


#### Building the model ####

## define our input variable and our output variable where ###
x = MICDATA.drop(['GDPCAPITA','MICROIDENSITY','EODB','MTSUBS','PPLRA','PGR','ADR','PSIZE','PPOOR','PIUI','FISFRED','FINSF','GNIPC','INSP','OPOE','INSDENS','INSP', 'FISFRED','BUSFRD','PROPR','INFREED','RULELAW', 'FRAGSTAT','LABOURFRE','GOVSIZE' ], axis = 1)
y = MICDATA['MICROIDENSITY']


# In[38]:


## Split dataset into training and testing portion
from sklearn.model_selection import train_test_split
import numpy as np

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 1)


# In[39]:


## Scale the independent variables gives
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
import numpy as np

min_max_scaler= preprocessing.MinMaxScaler()
x_train_minmax = min_max_scaler.fit_transform(x_train)
x_test_minmax = min_max_scaler.fit_transform(x_test)


# In[40]:


x_train = x_train_minmax
x_test= x_test_minmax


# In[41]:


## Create an instance of our model
regression_model2 = LinearRegression()

## Fit the model
regression_model2.fit(x_train, y_train)


# In[42]:


## Getting multiple prediction
y_predict = regression_model2.predict(x_test)

## Show the first five
y_predict[:5]


# In[43]:


## Evaluating the model
import statsmodels.api as sm
from statsmodels.stats import diagnostic as diag
from statsmodels.stats.outliers_influence import variance_inflation_factor

## Define our input variable
x2 = sm.add_constant(x)

## Create an OLS model
model = sm.OLS(y, x2)

## fit the data
est = model.fit()


# In[44]:


import math
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
get_ipython().run_line_magic('matplotlib', 'inline')

## Calculating the mean square Error
model_mse = mean_squared_error(y_test, y_predict)


## Calculate the mean absolute error
model_mae = mean_absolute_error(y_test, y_predict)

## Calculate the root mean squared error
model_rmse = math.sqrt(model_mse)


## Display the output
print("MSE {:.3}".format(model_mse))
print("MAE {:.3}".format(model_mae))
print("RMSE {:.3}".format(model_rmse))


# In[45]:


## Model summary
print(est.summary())


# In[ ]:





# In[46]:


## MODEL 3 Institutional Variables on MICROIDENSITY


# In[47]:


#### Building the model ####

## define our input variable and our output variable where ###
x = MICDATA.drop(['GDPCAPITA','MICROIDENSITY', 'EODB','ADR','PPOOR', 'PSIZE','PGR','PROPR','MTSUBS','PIUI','PPLRA','NETIPC','GNIPC','OPOE','INFLAT','REALIR','GDPOMT','INSP','INSDENS','FISFRED','FINSF'], axis = 1)
y = MICDATA['MICROIDENSITY']


# In[48]:


## Split dataset into training and testing portion
from sklearn.model_selection import train_test_split
import numpy as np

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 1)


# In[49]:


## Scale the independent variables gives
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
import numpy as np

min_max_scaler= preprocessing.MinMaxScaler()
x_train_minmax = min_max_scaler.fit_transform(x_train)
x_test_minmax = min_max_scaler.fit_transform(x_test)


# In[50]:


x_train = x_train_minmax
x_test= x_test_minmax


# In[51]:


## Create an instance of our model
regression_model3 = LinearRegression()

## Fit the model
regression_model3.fit(x_train, y_train)


# In[52]:


## Getting multiple prediction
y_predict = regression_model3.predict(x_test)

## Show the first five
y_predict[:5]


# In[53]:


## Evaluating the model
import statsmodels.api as sm
from statsmodels.stats import diagnostic as diag
from statsmodels.stats.outliers_influence import variance_inflation_factor

## Define our input variable
x2 = sm.add_constant(x)

## Create an OLS model
model = sm.OLS(y, x2)

## fit the data
est = model.fit()


# In[54]:


import math
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
get_ipython().run_line_magic('matplotlib', 'inline')

## Calculating the mean square Error
model_mse = mean_squared_error(y_test, y_predict)


## Calculate the mean absolute error
model_mae = mean_absolute_error(y_test, y_predict)

## Calculate the root mean squared error
model_rmse = math.sqrt(model_mse)


## Display the output
print("MSE {:.3}".format(model_mse))
print("MAE {:.3}".format(model_mae))
print("RMSE {:.3}".format(model_rmse))


# In[55]:


## Model summary
print(est.summary())


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




