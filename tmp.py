################### Criteria ###################
def processSubset(X,y,feature_set):
    # Fit model on feature_set and calculate rsq_adj
    regr = sm.OLS(y,X[list(feature_set)]).fit()
    rsq_adj = regr.rsquared_adj
    return {"model":regr, "rsq_adj":rsq_adj}

################### Best Subset ###################
def getBest(X,y,k):
    results = [] #fill results in a list
    #get X variable's all combinations
    for combo in itertools.combinations(X.columns, k):
        results.append(processSubset(X,y,combo))
    # Wrap everything up in a nice dataframe
    models = pd.DataFrame(results)
    # Choose the model with the highest rsq_adj
    best_model = models.loc[models['rsq_adj'].idxmax()]
    #Return the best model
    return best_model

################### Best Subset ###################
# Choosing Among Models
def best(X_train,y_train,X_test,y_test,mav):
    index_list = []
    coefi = {}
    prediction = []
    val_errors=[]
    models_best = pd.DataFrame(columns=["model", "rsq_adj"])
    #Here decide the MAX number of predictors
    for i in range(1,mav+1):
        models_best.loc[i] = getBest(X_train,y_train,i).values
        mod = models_best.loc[i,'model']
        index = list(mod.params.index)
        index_list.append(index)
        coefi[str(i)+'_variable_model'] = mod.params
        # select the selected model's required variable from X_prediction
        pred = mod.predict(X_test[index])
        prediction.append(pred)
        val_errors.append(np.mean((y_test-pred)**2))
    k_val = np.array(val_errors).argmin()+1 # find the best model
    return index_list, coefi, prediction, val_errors



################### forward ################
def forward(predictors,X,y):
    # Pull out predictors we still need to process
    remaining_predictors = [p for p in X.columns
                            if p not in predictors]
    results = []
    for p in remaining_predictors:
        results.append(processSubset(X,y,predictors+[p]))
        # Wrap everything up in a nice dataframe
    models = pd.DataFrame(results)
    # Choose the model with the highest rsq_adj
    best_model = models.loc[models['rsq_adj'].idxmax()]
    # Return the best model, along with model's other  information
    return best_model


##################### backward ################
def backward(predictors,X,y):
    results = []
    for combo in itertools.combinations(predictors, len(predictors)-1):
        results.append(processSubset(X,y,combo))
    # Wrap everything up in a nice dataframe
    models = pd.DataFrame(results)
    # Choose the model with the maximum rsq_adj
    best_model = models.loc[models['rsq_adj'].idxmax()]
    # Return the best model, along with some other useful information about the model
    return best_model






#### loop ####
def ForwardStepWise(X_est,y_est):
    models_fwd = pd.DataFrame(columns=["rsq_adj", "model"])
    predictors = []
    bic = []

    for i in range(1,len(X_est.columns)+1):
        models_fwd.loc[i] = forward(predictors,X_est,y_est)
        predictors = models_fwd.loc[i]["model"].model.exog_names
        print(predictors)

    best_model_fwd = models_fwd.loc[models_fwd['rsq_adj'].idxmax(),'model']
    print(best_model_fwd.params)
    return best_model_fwd

m = ForwardStepWise(DF_estimation.drop('y',1), DF_estimation['y'])