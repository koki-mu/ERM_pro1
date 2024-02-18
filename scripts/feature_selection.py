import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

def vif_check(x):
    columns_array = []
    vif_res = pd.DataFrame()
    # 許容値=10
    perm_level = 10

    while True:
        vif = pd.DataFrame()
        vif['VIF Factor'] = [variance_inflation_factor(x.values, i) for i in range(x.shape[1])]
        vif['features'] = x.columns
        vif['drop'] = ['' for i in range(len(x.columns))]

        if vif['VIF Factor'].max() <= perm_level:
            vif_res = pd.concat([vif_res, pd.DataFrame(vif)], axis = 1)
            break

        drop_idx = vif['VIF Factor'].idxmax()
        drop_col = vif.loc[drop_idx, 'features']
        x.drop(drop_col, axis = 1, inplace = True)
        vif['drop'].iloc[0] = drop_col

        vif_res = pd.concat([vif_res, pd.DataFrame(vif)], axis = 1)

        if len(x.columns) == 1:
            break

    columns_array = x.columns.tolist()

    return x, columns_array, vif_res