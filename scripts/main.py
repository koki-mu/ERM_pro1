import os
import re
import shutil
import time
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from statistics import mean
import warnings

import numpy as np
import pandas as pd
import seaborn as sns
import shap
import lightgbm as lgb
import optuna
import matplotlib.pyplot as plt
from optuna.samplers import TPESampler
import optuna.integration.lightgbm as lgb_o
from sklearn.model_selection import KFold
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
)

import feature_selection as fs

# for LightGBM Regression using Optuna

def main():
    """
    Main function to run the LightGBM regression model with Optuna tuning.
    """
    warnings.simplefilter('ignore')
    start = time.time()

    timestamp= datetime.now().strftime("%Y%m%d%H%M")
    DATASET_PATH_LIST = ['<dataset path>']
    FEATURE_SELECT_TYPE = 'full'
    MONTHES = [
        '1M',
        '3M',
        '6M',
        ]
    PREDICT_TYPE = 'normal'
    EXP_COUNT = 100

    ## Normal model
    for dataset in DATASET_PATH_LIST:
        DATASET_MONTH = re.search(r"_(\d+)$", dataset).group(1)
        for month in MONTHES:
            exp(month, FEATURE_SELECT_TYPE, PREDICT_TYPE, EXP_COUNT, dataset)

        OLD_FILE_PATH = '../result/LightGBM_o'
        NEW_FILE_PATH = f'../result/LightGBM_o_{EXP_COUNT}_{PREDICT_TYPE}_{DATASET_MONTH}_{FEATURE_SELECT_TYPE}_{timestamp}'
        os.rename(OLD_FILE_PATH, NEW_FILE_PATH)

    ## Chronological model
    PREDICT_TYPE = 'chrono'
    for dataset in DATASET_PATH_LIST:
        DATASET_MONTH = re.search(r"_(\d+)$", dataset).group(1)
        for month in MONTHES:
            exp(month, FEATURE_SELECT_TYPE, PREDICT_TYPE, EXP_COUNT, dataset)

        OLD_FILE_PATH = '../result/LightGBM_o'
        NEW_FILE_PATH = f'../result/LightGBM_o_{EXP_COUNT}_{PREDICT_TYPE}_{DATASET_MONTH}_{FEATURE_SELECT_TYPE}_{timestamp}'
        os.rename(OLD_FILE_PATH, NEW_FILE_PATH)

    end = time.time() - start
    q, mod = divmod(end, 60)
    print(f"{round(q)}mins {round(mod)}secs elapsed")

def exp(ob_name, feature_pattern, exp_pattern, EXP_COUNT, dataset):
    """
    Function to run the LightGBM regression model with Optuna tuning for a given set of parameters.

    Args:
        ob_name (str): The name of the objective variable.
        feature_pattern (str): The type of features to include ('patient_only', 'OCT_only', or 'full').
        exp_pattern (str): The type of experiment ('normal' or 'chrono').
        EXP_COUNT (int): The number of experiments to run.
        dataset (str): The path to the dataset file.
    """
    warnings.simplefilter('ignore')

    model = 'LightGBM_o'
    tuner_name = 'Optuna'

    # Create directories
    file_path_m = f'../result/{model}/{ob_name}'
    if os.path.isdir(file_path_m):
        shutil.rmtree(file_path_m)
    if not os.path.isdir(file_path_m):
        os.makedirs(file_path_m)

    OBJECTIVE_VARIABLE = f'{ob_name}_BCVA_logMAR'

    # Read dataset
    dataset_ERM = pd.read_csv(f'{dataset}.csv', index_col=0, header=0, dtype=float)

    # Select features based on feature_pattern
    if feature_pattern == 'patient_only':
        dataset_ERM = dataset_ERM[['AGE', 'Gender', 'Affected_eye', 'PRE_BCVA_logMAR', 'Axl', '1M_BCVA_logMAR', '3M_BCVA_logMAR', '6M_BCVA_logMAR']].copy(deep=True)
    elif feature_pattern == 'OCT_only':
        dataset_ERM = dataset_ERM.drop(['AGE', 'Gender', 'Affected_eye', 'PRE_BCVA_logMAR', 'Axl', '1M_BCVA_logMAR', '3M_BCVA_logMAR', '6M_BCVA_logMAR'], axis=1).copy(deep=True)
    elif feature_pattern == 'full':
        pass

    # Select the target variable index based on exp_pattern
    if exp_pattern == 'normal':
        idx = dataset_ERM.columns.tolist().index('1M_BCVA_logMAR')
    elif exp_pattern == 'chrono':
        idx = dataset_ERM.columns.tolist().index(f'{ob_name}_BCVA_logMAR')

    USE_EXPLANATORY = dataset_ERM.columns[:idx].tolist()

    dataset_ERM = pd.concat([dataset_ERM[USE_EXPLANATORY], dataset_ERM[OBJECTIVE_VARIABLE]], axis = 1)
    dataset_ERM = dataset_ERM.dropna()

    y = pd.DataFrame(dataset_ERM[OBJECTIVE_VARIABLE].values.reshape(-1, 1), index=dataset_ERM.index, columns=[OBJECTIVE_VARIABLE])
    x = dataset_ERM[USE_EXPLANATORY]

    model_dict = {'model_columns': None, 'model_score': None, 'model_params': None, 'model_vif_select': None}
    df_importance_list_exp = pd.DataFrame()

    for seed in range(EXP_COUNT):
        random_state = seed

        # Make directories
        file_path_o = f'../result/{model}/{ob_name}'
        if not os.path.isdir(file_path_o):
            os.makedirs(file_path_o)

        # Define variables to store results
        op_scores, best_models, model_scores = [], [], []
        df_y_preds, df_y_observes, df_train_result_mean, df_vali_result_mean, df_e_true, df_e_false = [pd.DataFrame() for _ in range(6)]
        break_count = None

        fold_count = 1
        score = 'rmse'
        num_leaves = 31
        max_depth = -1
        feature_fraction = 1.0
        min_data_in_leaf = 20
        min_child_samples = 20
        lambda_l1 = 0
        lambda_l2 = 0
        bagging_fraction = 1.0
        bagging_freq = 0

        params = {
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'metric': score,
            'random_seed': random_state,
            'verbosity': -1,
            'deterministic': True,
            'force_row_wise': True,
            'num_leaves': num_leaves,
            'max_depth': max_depth,
            'feature_fraction': feature_fraction,
            'min_data_in_leaf': min_data_in_leaf,
            'min_child_samples': min_child_samples,
            'lambda_l1': lambda_l1,
            'lambda_l2': lambda_l2,
            'bagging_fraction': bagging_fraction,
            'bagging_freq': bagging_freq,
            'n_jobs': -1,
        }

        # Define the number of splits for outer and inner cross-validation
        outer_cv = 5
        inner_cv = 4
        outer_kfold = KFold(n_splits=outer_cv, shuffle=True, random_state=random_state)
        inner_kfold = KFold(n_splits=inner_cv, shuffle=True, random_state=random_state)

        for train, test in outer_kfold.split(x, y):
            fold = 'fold{0}'.format(fold_count)
            file_path_f = f'../result/{model}/{ob_name}/{fold}'
            if not os.path.isdir(file_path_f):
                os.makedirs(file_path_f)

            x_train = x.iloc[train]
            columns_tmp_x = x_train.columns
            y_train = y.iloc[train]
            x_test = x.iloc[test]
            y_test = y.iloc[test]

            x_train = pd.DataFrame(x_train, columns = columns_tmp_x, dtype='float')
            x_test = pd.DataFrame(x_test, columns = columns_tmp_x, dtype='float')
            y_train = pd.DataFrame(y_train, columns = [OBJECTIVE_VARIABLE], dtype='float')
            y_test = pd.DataFrame(y_test, columns = [OBJECTIVE_VARIABLE], dtype='float')

            columns_tmp_x = x_train.columns

            # Remove features with VIF > 10
            x_train, x_train_columns_array, vif_res = fs.vif_check(x=x_train)
            vif_res.to_csv(f'../result/{model}/{ob_name}/{fold}/vif_res_{ob_name}_{fold}.csv')
            x_test = x_test.loc[:, x_train_columns_array]
            columns_tmp_x = x_train.columns

            df_feature_result_train = pd.concat([x_train, y_train], axis=1)
            df_feature_result_train.index = x_train.index
            df_feature_result_train.to_csv(f'../result/{model}/{ob_name}/{fold}/dataset_train_{ob_name}_{fold}.csv')
            df_feature_result_test = pd.concat([x_test, y_test], axis=1)
            df_feature_result_test.index = x_test.index
            df_feature_result_test.to_csv(f'../result/{model}/{ob_name}/{fold}/dataset_test_{ob_name}_{fold}.csv')

            train = lgb_o.Dataset(x_train, y_train)
            sampler = TPESampler(seed = random_state)
            study = optuna.create_study(sampler = sampler)

            def objective(trial):
                return

            study.optimize(objective, n_trials=1)
            sampler = TPESampler(seed=random_state)
            study = optuna.create_study(sampler=sampler)

            for i in range(7):
                study.enqueue_trial({'feature_fraction': 1.0})
            for i in range(20):
                study.enqueue_trial({'num_leaves': num_leaves})
            for i in range(15):
                study.enqueue_trial({'feature_fraction': 1.0})

            tuner = lgb_o.LightGBMTunerCV(
                params,
                train,
                study=study,
                optuna_seed=random_state,
                num_boost_round=1000,
                folds=inner_kfold,
                return_cvbooster=True,
                show_progress_bar=True,
                early_stopping_rounds=50,
            )

            # Inner CV
            tuner.run()
            df_trials = study.trials_dataframe()
            df_trials.to_csv(f'../result/{model}/{ob_name}/{fold}/trials_dataframe_{ob_name}_{fold}.csv')

            # Define the optimal score and model obtained from Bayesian optimization
            op_score = tuner.best_score
            op_scores.append(op_score)

            # Define the optimal model
            best_model = tuner.best_params
            best_models.append(best_model)

            # Obtain the booster and save it as the best model
            cv_booster = tuner.get_best_booster()
            best_estimater = cv_booster

            # Predict using the optimal model
            y_pred = best_estimater.predict(x_test)
            y_pred = np.array(y_pred).mean(axis=0)
            df_y_preds = pd.concat([df_y_preds, pd.DataFrame(y_pred)])
            df_y_observes = pd.concat([df_y_observes, y_test])

            # Calculate the model's score
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            model_scores.append(rmse)

            # Update the model_dict with the best parameters and score
            dec_rmse = Decimal(str(rmse)).quantize(Decimal('0.000001'), rounding=ROUND_HALF_UP)
            if model_dict['model_score'] is None:
                model_dict.update({'model_columns': columns_tmp_x, 'model_score': dec_rmse, 'model_params': best_model, 'model_vif_select': vif_res})
            elif model_dict['model_score'] > dec_rmse:
                model_dict.update({'model_columns': columns_tmp_x, 'model_score': rmse, 'model_params': best_model, 'model_vif_select': vif_res})

            # Plot the learning curve from the tuner's trial results
            rmse_mean = df_trials['value']
            round_r = np.arange(len(rmse_mean))
            plt.clf()
            plt.xlabel('round')
            plt.ylabel('rmse')
            plt.plot(round_r, rmse_mean)
            plt.savefig(f'../result/{model}/{ob_name}/{fold}/learning_curve_{ob_name}_{fold}.png', dpi=300)
            plt.clf()
            plt.close()

            # Calculate feature importance and plot
            importance_gain = best_estimater.feature_importance(importance_type='gain')
            df_importance_gain = pd.DataFrame(importance_gain, columns=columns_tmp_x)

            df_importance_list_exp = pd.concat([df_importance_list_exp, df_importance_gain], axis=0)

            df_importance_gain_mean = pd.DataFrame(np.array(df_importance_gain.mean()).reshape(1, len(columns_tmp_x)), columns=columns_tmp_x, index=['mean'])
            df_importance_gain = pd.concat([df_importance_gain, df_importance_gain_mean], axis=0)
            df_importance_gain.to_csv(f'../result/{model}/{ob_name}/{fold}/importance_gain_{ob_name}_{fold}.csv')

            df_importance_gain_mean.sort_values(by='mean', ascending=False, inplace=True, axis=1)

            plt.clf()
            plt.figure(figsize=(10, 10))
            sns.barplot(data=df_importance_gain_mean, orient='h', color='royalblue')
            plt.xlabel('importance')
            plt.ylabel('feature')
            plt.title(f'importance_gain:train_{score}={rmse}')
            plt.tight_layout()
            plt.savefig(f'../result/{model}/{ob_name}/{fold}/feature_importance_gain_{ob_name}_{fold}.png', dpi=300)
            plt.clf()
            plt.close()

            fold_count += 1

        if break_count == 1:
            continue

        # Calculate and save model scores
        df_result = pd.concat([pd.DataFrame(op_scores), pd.DataFrame(model_scores)], axis=1)
        df_result.columns = [f"train_score:{score}", f"test_score:{score}"]
        df_result_mean = pd.DataFrame(np.array(df_result.mean()).reshape(1, 2), index=[f'try{random_state}'], columns=df_result.columns)
        df_result = pd.concat([df_result, df_result_mean], axis=0)
        df_result.index = ['fold1', 'fold2', 'fold3', 'fold4', 'fold5', 'mean']
        df_result.to_csv(f'../result/{model}/{ob_name}/{tuner_name}_{model}_train_result_{ob_name}.csv')

        # Save mean train results
        data_file_path = f'../result/{model}/{ob_name}/{tuner_name}_{model}_train_result_mean_{ob_name}.csv'
        if os.path.exists(data_file_path):
            df_train_result_mean = pd.read_csv(data_file_path, index_col=0, header=0)
            df_train_result_mean = pd.concat([df_train_result_mean, df_result_mean], axis=0)
        elif not os.path.isdir(data_file_path):
            df_train_result_mean = df_result_mean
        df_train_result_mean.to_csv(data_file_path)

        # Reset indices for observed and predicted dataframes
        df_y_observes = df_y_observes.reset_index(drop=True)
        df_y_preds = df_y_preds.reset_index(drop=True)

        # Calculate evaluation metrics
        mse = mean_squared_error(y_true=df_y_observes.values, y_pred=df_y_preds.values)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true=df_y_observes.values, y_pred=df_y_preds.values)
        rmse_mae = rmse / mae
        print("RMSE:" + '{:.2f}'.format(rmse))
        print("MAE:" + '{:.2f}'.format(mae))
        print("RMSE / MAE:" + '{:.2f}'.format(rmse_mae))

        # Save validation results
        vali_result = pd.DataFrame({'RMSE': rmse, 'MAE': mae, 'RMSE/MAE': rmse_mae}, index=[f'scores{random_state}'])
        vali_result.to_csv(f'../result/{model}/{ob_name}/{tuner_name}_{model}_vali_result_scores_{ob_name}.csv')

        # Save mean validation results
        data_file_path = f'../result/{model}/{ob_name}/{tuner_name}_{model}_vali_result_scores_mean_{ob_name}.csv'
        if os.path.exists(data_file_path):
            df_vali_result_mean = pd.read_csv(data_file_path, index_col=0, header=0)
            df_vali_result_mean = pd.concat([df_vali_result_mean, vali_result], axis=0)
        elif not os.path.isdir(data_file_path):
            df_vali_result_mean = vali_result
        df_vali_result_mean.to_csv(data_file_path)

        print(f"{ob_name}_try{random_state};fin")
#<--------------------------------------------------------------------------------------------------------------------------------------------->#
#<--------------------------------------------------------------------------------------------------------------------------------------------->#
    # After completing all experiments
    exp_best_params = model_dict['model_params']
    exp_best_columns = model_dict['model_columns']
    x = x.loc[:, exp_best_columns]

    # Save the best parameters
    fin_df_best_params = pd.DataFrame(data={'model_best_params': [exp_best_params], 'model_best_columns': [np.array(exp_best_columns.values)], 'model_best_score': model_dict['model_score'], 'model_vif_select': [np.array(model_dict['model_vif_select'])]})
    fin_df_best_params.to_csv(f'../result/{model}/{ob_name}/{tuner_name}_{model}_fin_best_params_{ob_name}.csv')

    fin_result_scores, df_shap_all, fin_df_feature_importance_exp = [pd.DataFrame() for _ in range(3)]

    # Re-train with the best parameters and interpret using SHAP
    fold_count = 1
    fin_fold = KFold(n_splits=5, shuffle=True, random_state=42)
    for train, test in fin_fold.split(x, y):
        fin_fold = f'fin_fold{fold_count}'

        # Make directories (inner)
        file_path_f = f'../result/{model}/{ob_name}/{fin_fold}'
        if not os.path.isdir(file_path_f):
            os.makedirs(file_path_f)

        x_train = x.iloc[train]
        columns_tmp_x = x_train.columns
        y_train = y.iloc[train]
        x_test = x.iloc[test]
        y_test = y.iloc[test]

        x_train = pd.DataFrame(x_train, columns=columns_tmp_x, dtype='float')
        x_test = pd.DataFrame(x_test, columns=columns_tmp_x, dtype='float')
        y_train = pd.DataFrame(y_train, columns=[OBJECTIVE_VARIABLE], dtype='float')
        y_test = pd.DataFrame(y_test, columns=[OBJECTIVE_VARIABLE], dtype='float')

        # Re-train with the optimal model
        base_train = lgb.Dataset(x_train, y_train)
        shap_base_model = lgb.train(exp_best_params, base_train)

        # Plot feature importance
        fin_df_feature_importance = pd.DataFrame(shap_base_model.feature_importance(importance_type='gain'), index=exp_best_columns, columns=['Importance'])
        fin_df_feature_importance.sort_values('Importance', ascending=False, inplace=True, axis=0)

        fin_df_feature_importance_exp = pd.concat([fin_df_feature_importance_exp, fin_df_feature_importance], axis=1)

        plt.clf()
        plt.figure(figsize=(10, 10))
        sns.barplot(data=fin_df_feature_importance.T, orient='h', color='royalblue')
        plt.xlabel('importance')
        plt.ylabel('feature')
        plt.title(f'feature_importance_gain_subset{fold_count}')
        plt.tight_layout()
        plt.savefig(f'../result/{model}/{ob_name}/{fin_fold}/{tuner_name}_{model}_importance_plot_{ob_name}.png', dpi=300)
        plt.clf()
        plt.close()

        # Predict using the model
        y_pred = shap_base_model.predict(x_test)
        fin_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        fin_mae = mean_absolute_error(y_test, y_pred)
        fin_rmse_mae = fin_rmse / fin_mae

        # Calculate SHAP base score (using the best model)
        explainer = shap.TreeExplainer(model=shap_base_model, feature_names=columns_tmp_x)
        x_test_shap = x_test.copy().reset_index(drop=True)
        shap_values = explainer(X=x_test_shap)

        # Summarize results
        df_fin_scores = pd.DataFrame({'fin_RMSE': fin_rmse, 'fin_MAE': fin_mae, 'fin_RMSE/MAE': fin_rmse_mae, 'base_SHAP_value': explainer.expected_value}, index=[f'{fin_fold}'])
        fin_result_scores = pd.concat([fin_result_scores, df_fin_scores], axis=0)
        fin_result_scores.to_csv(f'../result/{model}/{ob_name}/{fin_fold}/{tuner_name}_{model}_fin_scores_{ob_name}.csv')

        # Plot SHAP values
        plt.figure()
        sns.set(style='white')
        plt.subplots_adjust(left=0.35, right=0.95, bottom=0.1, top=0.9)

        for i in range(len(x_test_shap)):
            plt.clf()
            shap.waterfall_plot(shap_values[i])
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.savefig(f'../result/{model}/{ob_name}/{fin_fold}/shap_waterfall_plot_{ob_name}_{fin_fold}_{i}.png', dpi=300)
            plt.clf()
            plt.close()

            plt.clf()
            shap.decision_plot(base_value=shap_values.base_values[i], shap_values=shap_values.values[i], feature_names=shap_values.feature_names)
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.savefig(f'../result/{model}/{ob_name}/{fin_fold}/shap_decision_plot_{ob_name}_{fin_fold}_{i}.png', dpi=300)
            plt.clf()
            plt.close()

        plt.clf()
        shap.summary_plot(shap_values, feature_names=columns_tmp_x)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(f'../result/{model}/{ob_name}/{fin_fold}/shap_summary_plot_{ob_name}_{fin_fold}.png', dpi=300)
        plt.clf()
        plt.close()

        plt.clf()
        shap.summary_plot(shap_values, feature_names=columns_tmp_x, plot_type='bar')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(f'../result/{model}/{ob_name}/{fin_fold}/shap_summary_bar_plot_{ob_name}_{fin_fold}.png', dpi=300)
        plt.clf()
        plt.close()

        try:
            shap.summary_plot(shap_values, feature_names=columns_tmp_x, plot_type='violin')
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.savefig(f'../result/{model}/{ob_name}/{fin_fold}/shap_summary_violin_plot_{ob_name}_{fin_fold}.png', dpi=300)
            plt.clf()
            plt.close()
        except:
            pass

        plt.clf()
        df_shap = pd.DataFrame(shap_values.values, columns=columns_tmp_x)
        df_shap.to_csv(f'../result/{model}/{ob_name}/{fin_fold}/df_shap_summary_bar_{ob_name}_{fin_fold}.csv')

        df_shap_all = pd.concat([df_shap_all, df_shap], axis=0)

        fold_count += 1

    # Calculate the mean of SHAP values across all folds
    df_shap_all_mean_abs = pd.DataFrame(np.array(df_shap_all.abs().mean()).reshape(1, -1), columns=df_shap_all.columns, index=['mean'])
    df_shap_all_mean_abs.sort_values('mean', ascending=False, inplace=True, axis=1)
    df_shap_all_mean_abs.to_csv(f'../result/{model}/{ob_name}/{tuner_name}_{model}_df_shap_all_mean_summary_bar_{ob_name}.csv')

    plt.clf()
    plt.figure(figsize=(10, 10))
    sns.barplot(data=df_shap_all_mean_abs, orient='h', color='royalblue')
    plt.xlabel('importance')
    plt.ylabel('feature')
    plt.yticks(fontsize=30)
    plt.tight_layout()
    plt.savefig(f'../result/{model}/{ob_name}/{tuner_name}_{model}_df_shap_all_mean_summary_bar_{ob_name}.png', dpi=300)
    plt.clf()
    plt.close()
    return None

if __name__ == "__main__":
    main()