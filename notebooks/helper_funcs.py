import sqlite3
from pathlib import Path 
import sys
import os
from datetime import datetime
import pandas as pd
import re
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC, SVR

from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
from sklearn.metrics import mean_squared_error as MSE, mean_absolute_error as MAE

# sys.path.append(str(Path(__file__).parent.parent.absolute()))


# DB_PATH = os.environ.get('DB_PATH')
# PARENT_PATH = os.path.realpath(f'./{DB_PATH}')
def get_data(symbols, DB_PATH):
    "Given a list of ticker_ids, returns historical data. sep_data returns multiple "

    "Get historical data given a list of symbols"
    if isinstance(symbols, list):
        str_list = ""
        for index, symbol in enumerate(symbols):
            if index != 0:
                str_list += ", " + '\'' + symbol + '\'' 
            else:
                str_list += '\'' + symbol + '\'' 
    else:
        str_list = '\'' + symbols + '\'' 
    
    conn = sqlite3.connect(DB_PATH)
    print(DB_PATH)
    cur = conn.cursor()

    cur.execute(f"""SELECT * FROM historical_prices WHERE ticker_id IN ({str_list})""")
    rows = cur.fetchall()
    return rows

def convert_unix_to_datetime(date_col):
    'Converts the unix dates into YYYY-MM-DD'
    int_list = list((map(int,date_col)))
    date_list = list(map(datetime.utcfromtimestamp, int_list))
    converted_dates = [date_list[i].strftime('%Y-%m-%d') for i in range(len(date_list))]
    return converted_dates

def separate_symbols(df):
    "Returns an individual df for each symbol. Sorted by alphabetical values of the symbols"
    list_dfs = []
    for symbol in sorted(df['symbol'].unique()):
        current_df = df[df['symbol'] == symbol]
        current_df = current_df.sort_values(by='date')
        list_dfs.append(current_df)
    return list_dfs

def plot_crypto(symbol):
    'Plot the graph given a symbol name'
    df_symbol = df[df['symbol'] == f'{symbol}']
    fig = go.Figure(data=go.Ohlc(x=df_symbol['date'],
                        open=df_symbol['open'],
                        high=df_symbol['high'],
                        low=df_symbol['low'],
                        close=df_symbol['close']))
    fig.update_layout(
    title=f'{symbol} currency')
    return fig.show()

# Taken from ML approaches to crypto
def _get_coin_cols(coin):
    """
    Used in preprocess function. Returns relevant columns for a given coin
    """
    cols = []
    for col in coinmetric_df.columns:
        if re.match(coin, col):
            cols.append(col)
    time_df = pd.DataFrame(coinmetric_df['Time'])
    time_df.rename(columns={"Time": "date"}, inplace=True)
    return time_df.join(coinmetric_df[cols])

def _take_diff(column_list, df):
    """
    Used in preprocess function. Returns the difference for a given list of columns and dataframe.
    """
    for col in column_list:
        df[col] = df[col].diff()
    return df

def preprocess(df, symbol):
    """
    Given a dataframe and a symbol (i.e. 'BTC'), returns a clean dataset with the relevant columns.
    Computes relative price change, parkinson volatility and adds 7 lags to the closing price and volatility.
    
    Returns a dataframe
    """
    # Predicting todays price given yesterdays information, so we need to shift close by -1
    df['close'] = df['close'].shift(-1)
    
    # Natural log of closing price is taken
    df['close'] = np.log(df['close'])
    
    # Need to use current data to predict 1 step ahead
    # df['close'] is shifted one step back to achieve this
    
    df.loc['close'] = df['close'].shift(-1)
    df['rel_price_change'] = 2 * (df['high'] - df['low']) / (df['high'] + df['low'])
    df['parkinson_vol'] = np.sqrt((np.log(df['high']/df['low'])**2)/4*np.log(2))
    df = df[['date', 'close', 'volumeto', 'volumefor', 'rel_price_change', 'parkinson_vol']]
    
    for i in range(1,8):
        lag_close_col = df['close'].shift(i)
        lag_park_col = df['parkinson_vol'].shift(i)
        df['close_lag'+str(i)] = lag_close_col
        df['parkinson_lag'+str(i)] = lag_park_col
    
    # df = df.merge(_get_coin_cols(symbol), on='date')
    df = df.set_index('date')
    # df.columns =[re.sub(symbol+' / ', '', col) for col in df.columns]
    
    # column_list = ['Market Cap (USD)', 'Tx Cnt', 'Active Addr Cnt', 
    #            'Mean Difficulty', 'Block Cnt', 'Xfer Cnt']
    # btc_df = _take_diff(column_list, df=df)
    return df

def split_data(df, return_test = False):
    if isinstance(df, tuple):
        df = df[0]
    df = df.dropna()
    train = df[:int(len(df) * 0.75)]
    test = df[int(len(df) * 0.75):]
    if return_test:
        return test

    y = train['close']
    X = train.drop('close', axis=1)

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, shuffle=False)
    return X_train, X_valid, y_train, y_valid

class model_pipeline:
    """
    Full model pipeline.
    Requires cleaned dataframe from preprocess and the following imports:
    MAE, MSE, f1_score, precision_score, recall_score, accuracy_score
    as well as the models
    """
    def __init__(self, df, verbose=False):
        self.df = df,
        self.verbose=verbose,
        self.models = ['LogisticRegression', 'RandomForestClassifier', 'SVC', 'EnsembleClf',
          'LinearRegression', 'RandomForestRegressor', 'SVR', 'EnsembleReg'],
        self.error = ['MAE', 'MSE', 'f1', 'precision', 'recall', 'accuracy'],
        self.model_list_clf = ['RandomForestClassifier', 'SVC', 'LogisticRegression'],
        self.model_list_reg = ['LinearRegression', 'RandomForestRegressor', 'SVR']

    def _class_or_reg(self, mod_type, sk_model, X_train, y_train, X_valid, y_valid):
        """
        Helper function for fit_model
        """
        sk_model.fit(X_train, y_train)
        y_pred = sk_model.predict(X_valid)
        if mod_type == 'regression':
            errors = MAE(y_valid, y_pred), MSE(y_valid, y_pred)
            return sk_model, y_pred, errors

        elif mod_type == 'classification':
            errors = [f1_score(y_valid, y_pred), precision_score(y_valid, y_pred),
                    recall_score(y_valid, y_pred), accuracy_score(y_valid, y_pred)]
            return sk_model, y_pred, errors
    
    def fit_model(self, model_type):
        """

        Parameters
        ----------

        df : dataframe
            Full cleaned dataframe for a given coin

        model_type: str
            'RandomForestClassifier',
            'RandomForestRegressor',
            'SVC',
            'SVR',
            'LinearRegression',
            'LogisticRegression'

        Returns
        -------
        trained_model : sklearn model
            The model trained on the training set and validated on validation set. Unseen to test set

        y_prediction: np.array
            A 1-D array that the model has predicted on the validation set.

        errors : list
            if regression model:
                returns [MAE, MSE]
            if classification mode:
                returns [f1, precision, recall, accuracy]

        """
        X_train, X_valid, y_train, y_valid = split_data(self.df)

        #Convert to binary dependent variables for classification models
        clf_y_train = y_train.diff() > 0
        clf_y_valid = y_valid.diff() > 0

        if model_type == 'RandomForestClassifier':
            clf = RandomForestClassifier()
            return self._class_or_reg(mod_type = 'classification', sk_model=clf, X_train=X_train, y_train = clf_y_train,
                         X_valid=X_valid, y_valid=clf_y_valid)

        elif model_type == 'RandomForestRegressor':
            reg = RandomForestRegressor()
            return self._class_or_reg(mod_type = 'regression', sk_model=reg, X_train=X_train, y_train=y_train, 
                                X_valid=X_valid, y_valid=y_valid)

        elif model_type == 'SVC':
            clf = SVC()
            return self._class_or_reg(mod_type = 'classification', sk_model=clf, X_train=X_train, y_train = clf_y_train,
                         X_valid=X_valid, y_valid=clf_y_valid)

        elif model_type == 'SVR':
            reg = SVR()
            return self._class_or_reg(mod_type = 'regression', sk_model=reg, X_train=X_train, y_train=y_train, 
                                X_valid=X_valid, y_valid=y_valid)


        elif model_type == 'LinearRegression':
            reg = LinearRegression()
            return self._class_or_reg(mod_type = 'regression', sk_model=reg, X_train=X_train, y_train=y_train, 
                                X_valid=X_valid, y_valid=y_valid)

        elif model_type == 'LogisticRegression':
            clf = LogisticRegression()
            return self._class_or_reg(mod_type = 'classification', sk_model=clf, X_train=X_train, y_train = clf_y_train,
                         X_valid=X_valid, y_valid=clf_y_valid)

    def make_dataframe(self, ensemble=True):
        """
        Summarizes the errors of all used models.
        Ensemble adds a combination model for both classification and regression
        Returns a dataframe with all relevant errors in self.error
        """
        df_summary = pd.DataFrame(index = self.models[0], columns = self.error[0])
        classify_models = []
        self.model_list_clf = self.model_list_clf[0]
        for i in self.model_list_clf:
            clf, clf_pred, [f1, precision, recall, accuracy] = self.fit_model(i)
            classify_models.append((f1, precision, recall, accuracy))
        
        for i, vals in zip(self.model_list_clf, classify_models):
            df_summary.loc[i][2:] =  vals

        reg_models = []
        for i in self.model_list_reg:
            reg, reg_pred, [mae, mse] = self.fit_model(i)
            reg_models.append([mae, mse])

        for i, vals in zip(self.model_list_reg, reg_models):
            df_summary.loc[i][:2] = vals
            
        if ensemble:
            clf_errors, reg_errors = self.ensemble()
            df_summary.loc['EnsembleClf'][2:] = clf_errors
            df_summary.loc['EnsembleReg'][:2] = reg_errors

        return df_summary
    
    def ensemble(self):
        """
        Combines all classification methods listed in self.model_list_clf and
        regression methods in self.model_list_reg and returns the average result.
        """
        ensemble_clf = []
        if isinstance(self.model_list_clf, tuple):
            self.model_list_clf = self.model_list_clf[0]
        for i in self.model_list_clf:
            clf, clf_pred, [f1, precision, recall, accuracy] = self.fit_model(i)
            ensemble_clf.append(clf_pred)
        mapping = {True: 1, False: -1}
        mapped_data = []
                                                  
        for i in ensemble_clf:
            mapped_data.append([mapping[x] for x in i])
        model_1, model_2, model_3 = np.array(mapped_data)
        y_pred = model_1+model_2+model_3
        _, _, _, y_valid = split_data(self.df)
        y_valid_clf = y_valid.diff() > 0
        y_pred = y_pred > 0
        clf_errors = [f1_score(y_valid_clf, y_pred), precision_score(y_valid_clf, y_pred),
                    recall_score(y_valid_clf, y_pred), accuracy_score(y_valid_clf, y_pred)]
        
        ensemble_reg = []
        for i in self.model_list_reg:
            reg, reg_pred, [mae, mse] = self.fit_model(i)
            ensemble_reg.append(reg_pred)
        model_1, model_2, model_3 = ensemble_reg
        ensemble_reg = (np.array(model_1) + np.array(model_2) + np.array(model_3))/3
        reg_errors = [MAE(y_valid, ensemble_reg), MSE(y_valid, ensemble_reg)]
        
        return clf_errors, reg_errors
if __name__ == '__main__':
    rows = get_data('BTC')
    rows = get_data(['BTC'])
    rows = get_data(['BTC', 'ETH'])