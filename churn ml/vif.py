from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd
def cal_vif(X):

    vif=pd.DataFrame()
    vif["variables"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    return(vif)
