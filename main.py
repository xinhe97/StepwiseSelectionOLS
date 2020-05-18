from ForwardStepwiseOLS import *
from BackwardStepwiseOLS import *
from BestSubsetSelectionOLS import *

# DGP
# Spare signals
N = 1000
P = 10 # Total number of inputs

N_true_inputs = 5
N_false_inputs = P - N_true_inputs
n_obs = N/2
n_pred = N/2
error_sd = 1

#True inputs have coefficient 1
beta = np.matrix(np.zeros((P,1)))
beta[:N_true_inputs,:] = 1

#Simulate the data
X = np.matrix(np.random.rand(N,P))
epsilon = np.matrix(error_sd*np.random.normal(0,size=(N,1)))
y = X*beta + epsilon
y_label = y>1

# Pack the data into a dataframe
#DF = pd.concat([pd.DataFrame(X),pd.DataFrame(y)],axis=1)
DF = pd.concat([pd.DataFrame(X),pd.DataFrame(y_label)],axis=1)
new_names_true = ['x_true_'+str(i) for i in range(1,N_true_inputs+1)]
new_names_false = ['x_false_'+str(i) for i in range(1,N_false_inputs+1)]
names = new_names_true + new_names_false + ['y']
DF.columns = names

# Now we split the data into an estimation and prediction sample. # Randomly draw n_obs observations
train_index = random.sample(range(0,N),np.int(n_obs))
train_index.sort()
DF_estimation = DF.loc[train_index,:]
DF_prediction = DF.drop(index=train_index)

###### GridSearchCV ######

############
# # forward
# param_grid_pipe_fwd = {
#     'fwd__fK': [1,2,3,4,5,6,7,8,9,10]
# }
# pipe = Pipeline(steps=[('fwd', ForwardStepwiseOLS())])
# search = GridSearchCV(estimator=pipe, cv=5, param_grid=param_grid_pipe_fwd, n_jobs=-1)
# search.fit(DF_estimation.drop('y',1), DF_estimation['y'])

# print(search.best_params_)
# print(search.cv_results_)

############
# # backward
# # the hyperparameter fK: is the least number of features included
# param_grid_pipe_bwd = {
#     'bwd__fK': [1,2,3,4,5,6,7,8,9,10]
# }
# pipe = Pipeline(steps=[('bwd', BackwardStepwiseOLS())])
# search = GridSearchCV(estimator=pipe, cv=5, param_grid=param_grid_pipe_bwd, n_jobs=-1)
# search.fit(DF_estimation.drop('y',1), DF_estimation['y'])

# print(search.best_params_)
# print(search.cv_results_)

###########
# best subset
# the hyperparameter fK: is the number of features included
param_grid_pipe_bst = {
    'bst__fK': [1,2,3,4,5,6,7,8,9,10]
}
pipe = Pipeline(steps=[('bst', BestSubsetSelectionOLS())])
search = GridSearchCV(estimator=pipe, cv=5, param_grid=param_grid_pipe_bst, n_jobs=-1)
search.fit(DF_estimation.drop('y',1), DF_estimation['y'])

print(search.best_params_)
print(search.cv_results_)
