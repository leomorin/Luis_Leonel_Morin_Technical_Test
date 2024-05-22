import pandas as pd
import numpy as np

## model and stats libraries
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import catboost as cb
from sklearn.metrics import mean_absolute_error,  mean_absolute_percentage_error, mean_squared_error, r2_score

import matplotlib.pyplot as plt
import seaborn as sns
from pandas import concat
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import catboost as cb
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
import warnings

warnings.filterwarnings('ignore')

## ABT Functions
#FUNCTION THAT TRANSFORMS DATA INTO WINDOWS
def series_to_supervised(data, columns, n_in=1, n_out=1, dropnan=True):
          
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [(columns[j]+'(t-%d)' % (i)) for j in range(n_vars)]
         # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [columns[j]+'(t)' for j in range(n_vars)]
        else:
            names += [(columns[j]+'(t+%d)' % (i)) for j in range(n_vars)]
     # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

#FUNCTION THAT RETURNS THE DATAFRAME OF WINDOWS
def windowed_df(df,columns, n_windows):

    raw = pd.DataFrame()
    
    for col in columns:
        
        raw[col] = [x for x in df[col]]
    
    values = raw.values
    df_result = pd.DataFrame()
    df_result  = series_to_supervised(values,columns,  n_windows -1, 1 )
    df_result  = df_result.reset_index(drop=True)

    return df_result 

## Training Functions
# CROSS-VALIDATION IS PERFORMED FOR SEVERAL MODELS TO SELECT THE BEST ONE
def cross_validation(df, n_splits,test_size, modelo, target, training_final, n_columns) :

    # Inicializacion de los fold para la serie de tiempo
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
    
    #Variables donde se van a guardar las metricas de cada particion
    mape_scores = []
    r2_scores = []
    
    #Este caso aplica cuando se realiza el training y se esta buscando el mejor modelo para construir la base predictiva de cada variable     
    if training_final== False :
    
        #columnas de las variable de estudio target (modelos univariados)
        columns = [x for x in  df.columns if x[:len(target)].find(target)!= -1 ]
        print(columns)
        #Ciclo que hace la evaluacion del modelo por fold

        for train_index, test_index in tscv.split(df):

            #Se eligen las particiones del cv
            df_train, df_test = df.iloc[train_index], df.iloc[test_index]

            #Elegimos las variables 
            df_train = df_train[columns]
            df_test = df_test[columns]

            # Variables independientes (X) y variable dependiente (y) para entrenamiento y prueba
            X_train = np.array(df_train.iloc[:, :-1])
            y_train = np.array(df_train.iloc[:, -1])
            X_test = np.array(df_test.iloc[:, :-1])
            y_test = np.array(df_test.iloc[:, -1])

            # Ajuste modelo y prediccion
            model = modelo
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Metricas
            mape = mean_absolute_percentage_error(y_test, y_pred)
            mape_scores.append(mape)
            R2 = r2_score(y_test, y_pred)
            r2_scores.append(R2)


        # Promedio de las metricas (promedio folds)
        prom_mape = np.mean(mape_scores)
        prom_R2 = np.mean(r2_scores)
        
    # Esta caso aplica cuando ya se va a realizar el entrenamiento final para luego hacer las predicciones     
    else :

        for train_index, test_index in tscv.split(df):

            #Se eligen las particiones del cv
            df_train, df_test = df.iloc[train_index], df.iloc[test_index]
            
            # Variables independientes (X) y variable dependiente (y) para entrenamiento y prueba
            X_train = np.array(df_train.iloc[:, :-n_columns])
            y_train = np.array(df_train[target + '(t)'])
            X_test = np.array(df_test.iloc[:, :-n_columns])
            y_test = np.array(df_test[target + '(t)'])

            # Ajuste modelo y prediccion
            model = modelo
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Metricas
            mape = mean_absolute_percentage_error(y_test, y_pred)
            mape_scores.append(mape)
            R2 = r2_score(y_test, y_pred)
            r2_scores.append(R2)


        # Promedio de las metricas (promedio folds)
        prom_mape = np.mean(mape_scores)
        prom_R2 = np.mean(r2_scores)
    
    return  prom_R2 , prom_mape

#Function that calculates the best model per variable to make future predictions
def winner_model(df, columns_final, training_final,n_columns) :
    random_seed = 42
    # Se definen los modelo a probar
    LR_modelo = LinearRegression()
    RF_modelo = RandomForestRegressor(random_state=random_seed)
    CB_modelo = cb.CatBoostRegressor(random_state=random_seed, loss_function='RMSE')
    lgbm_model = LGBMRegressor(random_state=random_seed)
    gb_model = GradientBoostingRegressor(random_state=random_seed)
    
    #Se definen 
    n_splits = 4
    test_size = None
    
    
    #Inicializamos las variables que contendran los resultados del mejor modelo
    r2_result = []
    mape_result= []
    modelo_ganador = []
    status = []
    model_name = []
    
    #Inicializacion base resultante
    df_result = pd.DataFrame()
    
    #Ciclo para obtener el mejor modelo por variable
    for col in columns_final :
        
        #Se las metricas R2 y Mape del analisis de cross_validation (prom folds)
        r2_LR ,mape_LR = cross_validation(df, n_splits,test_size, LR_modelo, col, training_final, n_columns)
        r2_RF ,mape_RF = cross_validation(df, n_splits,test_size, RF_modelo, col, training_final, n_columns)
        r2_CB ,mape_CB = cross_validation(df, n_splits,test_size, CB_modelo, col, training_final, n_columns)
        r2_LGBM ,mape_LGBM = cross_validation(df, n_splits,test_size, lgbm_model, col, training_final, n_columns)
        r2_GB ,mape_GB = cross_validation(df, n_splits,test_size, gb_model, col, training_final, n_columns)
        
        #Se encuentra el modelo con mejor MAPE
        metricas_MAPE = [mape_LR, mape_RF, mape_CB, mape_LGBM, mape_GB ]
        minimo_MAPE = min(metricas_MAPE)  
        posicion_minimo = metricas_MAPE.index(minimo_MAPE)
        
        #Condicional para guardar las metricas del mejor modelo
        if posicion_minimo == 0 :
            
            modelo_ganador_aux = LR_modelo
            r2_result_aux = r2_LR
            mape_result_aux = mape_LR
            model_name_aux = 'lr'
            
        elif posicion_minimo == 1 :
            
            modelo_ganador_aux = RF_modelo
            r2_result_aux = r2_RF
            mape_result_aux = mape_RF
            model_name_aux = 'rf'

        elif posicion_minimo == 2 :
            
            modelo_ganador_aux = CB_modelo
            r2_result_aux = r2_CB
            mape_result_aux = mape_CB
            model_name_aux = 'cat'

        elif posicion_minimo == 3 :
            
            modelo_ganador_aux = lgbm_model
            r2_result_aux = r2_LGBM
            mape_result_aux = mape_LGBM 
            model_name_aux = 'lgbm'
        
        else :
            
            modelo_ganador_aux = gb_model
            r2_result_aux = r2_GB
            mape_result_aux = mape_GB
            model_name_aux = 'gb'
            
        
        #CONDICIONAL PARA CLASIFICAR EL STATUS DE LA METRICA R2
        if mape_result_aux <= 0.03 :
            status_aux = 'Excelente'
            
        elif mape_result_aux> 0.03 and mape_result_aux < 0.1 :
            status_aux = 'Bueno'
            
        elif mape_result_aux>= 0.1 and mape_result_aux < 0.3 :
            status_aux = 'Regular'
            
        else:
             status_aux = 'Malo'
            
        # Se guardar las metricas del mejor modelo por cada columna
        modelo_ganador.append(modelo_ganador_aux)
        r2_result.append(r2_result_aux)
        mape_result.append(mape_result_aux)
        status.append(status_aux)
        model_name.append(model_name_aux)
        
    #Construccion del df final
    df_result['Variables'] = columns_final
    df_result['Modelo_Ganador'] = modelo_ganador
    df_result['R2'] = r2_result
    df_result['MAPE'] = mape_result
    #df_result['Status_MAPE'] = status
    df_result['model_name'] = model_name
    
    return df_result

## Predicction functions

# CREATES THE BASE WITH THE DATA FOR MAKING PREDICTIONS

def df_predict(df, modelos_x_vars, n_periodos ) :

    # reentrenar las series univariadas por variable con el modelo ganador y predecir
    
    var_cont = 1
    
    for var in modelos_x_vars['Variables'].values :
        print(var)
        columns = [x for x in  df.columns if x[:len(var)].find(var)!= -1 ]

        df_train = df[columns]
        X_train = np.array(df_train.iloc[:, :-1])
        y_train = np.array(df_train.iloc[:, -1])

        modelo = modelos_x_vars[modelos_x_vars['Variables'] == var]['Modelo_Ganador'].sum()
        modelo.fit(X_train, y_train)
        
        columns_names =  list(df_train.iloc[:, :-1].columns)
        
        # Nueva ultima columna
        last_row_new = df_train.tail(1).reset_index(drop=True).iloc[:,1:]
        last_row_new.columns = columns_names
        
        df_result_var = last_row_new
        
        for i in range(0,n_periodos):
            
            last_row_new_aux = pd.DataFrame()
            last_row_new_aux['new_val'] = modelo.predict(np.array(last_row_new))
            
            aux = last_row_new.tail(1).reset_index(drop=True).iloc[:,1:]
            last_row_new = pd.concat([aux, last_row_new_aux], axis=1)
            last_row_new.columns = columns_names
            
            df_result_var = pd.concat([df_result_var, last_row_new], axis = 0)
            
            
        if var_cont == 1 :
            
            df_result_final = df_result_var
            var_cont += 1
        
        else:
            df_result_final = pd.concat([df_result_final,df_result_var], axis =1)
            var_cont +=1
        
    df_result_final = df_result_final.reset_index(drop=True)  
    return df_result_final

#Given the chosen model in the model_winner function, we optimize it using RandomSearch to obtain a robust model for production
def optimizar_best_model (X, y, name_best_model):

    if name_best_model == 'gb':
                
        param_gb = {
        'n_estimators': [100, 200, 300, 400],
        'learning_rate': [0.1, 0.05, 0.01, 0.005, 0.001],
        'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
        'min_samples_split': [2, 3, 4, 5],
        'min_samples_leaf': [1, 2, 3, 4, 5]}
                
        modelo = RandomizedSearchCV(
        GradientBoostingRegressor(random_state=117),
        param_distributions=param_gb,
        n_iter=50, 
        cv=3)
                
        modelo.fit(X, y)
        mejor_modelo = modelo.best_estimator_

        # Obtener el MAPE del mejor modelo
        best_index = modelo.best_index_
        best_mape = -modelo.cv_results_['mean_test_score'][best_index]
        #break
        
    if name_best_model == 'lgbm':   
                
        param_lgbm = {
            'boosting_type': ['gbdt', 'dart', 'goss'],  # Explora otros tipos de boosting
            'num_leaves': [10, 20, 30, 40, 50],
            'min_child_samples': [1, 5, 10, 20],
            'max_depth': [3, 4, 5, 6, 7],
            'learning_rate': [0.001, 0.01, 0.05, 0.1],  # Incluye una tasa de aprendizaje intermedia
            'reg_alpha': [0, 0.1, 0.5, 1],
            'feature_fraction': [0.6, 0.8, 1.0],  # Agrega fracción de características
            'bagging_fraction': [0.6, 0.8, 1.0],  # Agrega fracción de muestreo
            'bagging_freq': [0, 1, 5]  # Frecuencia de muestreo
        }

        modelo = RandomizedSearchCV(
        LGBMRegressor(random_state=117),
        param_distributions=param_lgbm,
        n_iter=50, 
        cv=3)
                
        modelo.fit(X, y)
        mejor_modelo = modelo.best_estimator_
        
        # Obtener el MAPE del mejor modelo
        best_index = modelo.best_index_
        best_mape = -modelo.cv_results_['mean_test_score'][best_index]
        #break

    if name_best_model == 'cat':   
                
        param_cat = {
        'depth': [4, 6, 8],
        'learning_rate': [0.01, 0.05],
        'iterations': [100, 200]}

        modelo = RandomizedSearchCV(
        CatBoostRegressor(random_state=117),
        param_distributions=param_cat,
        n_iter=50,
        cv=3)
                
        modelo.fit(X, y)
        mejor_modelo = modelo.best_estimator_

        # Obtener el MAPE del mejor modelo
        best_index = modelo.best_index_
        best_mape = -modelo.cv_results_['mean_test_score'][best_index]
        #break

    if name_best_model == 'rf':   

        param_rf = {
        'n_estimators': [100],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5, 10]}

        modelo = RandomizedSearchCV(
        RandomForestRegressor(random_state=117),
        param_distributions=param_rf,
        n_iter=50, 
        cv=3)
                
        modelo.fit(X, y)
        mejor_modelo = modelo.best_estimator_

        # Obtener el MAPE del mejor modelo
        best_index = modelo.best_index_
        best_mape = -modelo.cv_results_['mean_test_score'][best_index]
        #break

    if name_best_model == 'lr':
                
        modelo = LinearRegression()
                
        modelo.fit(X, y)
        mejor_modelo = modelo

        # Obtener el MAPE del mejor modelo
        best_index = modelo.best_index_
        best_mape = -modelo.cv_results_['mean_test_score'][best_index]
        #break          

    return mejor_modelo, best_mape


