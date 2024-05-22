Explanation step by step for the utils and main codes

General explication: Sliding windows are applied as a training method for the time series, aiming to predict the variable y. Additionally, various ML models such as LGBM, Gradient Boosting Regressor, Linear Regressor, Random Forest Regressor, and CatBoost Regressor are trained to obtain the best possible model for the provided data. The model is also optimized using a predefined hyperparameter grid to ensure its robustness in a production environment.

utils.py

Function's explication:

-series_to_supervised : FUNCTION THAT TRANSFORMS DATA INTO WINDOWS
-windowed_df : FUNCTION THAT RETURNS THE DATAFRAME OF WINDOWS
-cross_validation : CROSS-VALIDATION IS PERFORMED FOR SEVERAL MODELS TO SELECT THE BEST ONE
-winner_model : Function that calculates the best model per variable to make future predictions
-df_predict: CREATES THE BASE WITH THE DATA FOR MAKING PREDICTIONS
-optimizar_best_model : Given the chosen model in the model_winner function, we optimize it using RandomSearch to obtain a robust model for production

MAIN.py
- Libraries: import out útil.py that contains all de functions we need to make the predicctions and train our time series model
- Data Reading: reed the data provide and change the name the first colums as date and put it in the correct format
- ABT: build in this stage our analytical base table using the functions in utils.py (series_to_supervised and windowed_df)
- Predictions: generate the precictions using our winner model.
- Output: in this stage we concatenate the real data with teh predicctions, and generate the document with the real data and the predictions for 12 months

You only need to run the code MAIN without change anything. 
