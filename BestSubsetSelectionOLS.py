import numpy as np
import pandas as pd
import statsmodels.api as sm
import random
import itertools
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.metrics import euclidean_distances

np.random.seed(2020)
random.seed(2020)

# %%pixie_debugger
class BestSubsetSelectionOLS(BaseEstimator):
    def __init__ (self, fK=3):
        self.fK = fK              #number of predictors
        
    def myBic(self, n, mse, k):
        if k<=0:
            return np.nan
        else:
            return n*np.log(mse) + k*np.log(n)
        
    ############   Criteria   ##################################
    def processSubset(self, X,y,feature_set):
        regr = sm.OLS(y, X[list(feature_set)]).fit()

        rsq_adj = regr.rsquared_adj

        bic = self.myBic(X.shape[0], regr.mse_resid, len(feature_set))
        rsq = regr.rsquared
        return{"model": regr, "rsq_adj":rsq_adj, "bic":bic, "rsq":rsq, "best_predictors": feature_set}  
    
 ############## bext subset selection######################
    def getBest(self, X,y,fK):
        results = [] #fill results in a list
        #get X variable's all combinations(X.columns,k):
        X = pd.DataFrame(X)
        for combo in itertools.combinations(X.columns, fK):
            results.append(self.processSubset(X,y,combo))
        # Wrap everything up in a nice dataframe
        models = pd.DataFrame(results)
        # Choose the model with the highest rsq_adj
        best_model =  models.loc[models["rsq"].idxmax(), 'model']
#         best_model = 1
        #Return best_model
        best_predictors =  models.loc[models["rsq"].idxmax(), 'best_predictors']
        print(fK, best_predictors)
        return best_model, best_predictors #later add feature_set
    
# forwardK should be not applicable to best selection since best only choose a specific fK

    def fit(self, X, y):
        X, y = check_X_y(X, y, accept_sparse = True)
        
        self.best_model, self.best_predictors = self.getBest(X,y, self.fK) #later add return predictors
        
        # hexin
        self.is_fitted_ = True
        # print(self.best_predictors)
        
        return self
    
    def predict(self, X):
        X = check_array(X, accept_sparse = True)
        
        # hexin
        y_pred = self.best_model.predict(X[:, list(self.best_predictors)]) #later add returning the feature_set
        # hexin
        
        check_is_fitted(self, 'is_fitted_')
        return y_pred
    
    def get_params(self, deep = True):
        return {"fK": self.fK}
    
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
    
    def score(self, X, y_true):
        return r2_score(y_true, self.predict(X))
 
if __name__ == '__main__': 
    N = 1000
    P = 10 # Total number of inputs

    N_true_inputs = 5
    N_false_inputs = P - N_true_inputs
    n_obs = N/2
    n_pred = N/2
    error_sd = 1

    # True inputs have coefficient 1
    beta = np.matrix(np.zeros((P,1)))
    beta[:N_true_inputs, :] = 1

    # stimulate the data
    X = np.matrix(np.random.rand(N,P))
    epsilon = np.matrix(error_sd*np.random.normal(0, size= (N,1)))
    y = X*beta + epsilon

    # Pack the data into a dataframe
    DF = pd.concat([pd.DataFrame(X), pd.DataFrame(y)], axis = 1)
    new_names_true = ['x_true_'+str(i) for i in range(1, N_true_inputs + 1)]
    new_names_false = ['x_true_' +str(i) for i in range (1, N_false_inputs +1)]
    names = new_names_true + new_names_false + ['y']
    DF.columns = names

    # Now we split the data into an estimation and prediction sample. # Randomly draw n_obs obervations
    train_index = random.sample(range(0,N), np.int(n_obs))
    train_index.sort()
    DF_estimation = DF.loc[train_index, :]
    DF_prediction = DF.drop(index = train_index) 

    ####### Algorithm ####################
    bfit = BestSubsetSelectionOLS(fK=5)
    bfit.fit(DF_estimation.drop('y', 1), DF_estimation['y'])
    bfit.predict(DF_prediction.drop('y',1))
    print(bfit.score(DF_prediction.drop('y',1), DF_prediction['y']))
