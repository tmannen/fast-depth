 1/1: sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
 1/2: improt tensorflow as tf
 1/3: import tensorflow as tf
 2/1: import tensorflow as tf
 3/1: import tensorflow as tf
 3/2: sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
 4/1:
import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))
 5/1:
import pandas
import numpy
import xgboost as xgb
 5/2:
import pandas as pd
import numpy as np
import xgboost as xgb
 5/3: data = pd.read_csv("data/cons_training.csv")
 5/4: data.head()
 5/5: data.describe()
 5/6: data.dtypes()
 5/7: data.dtype()
 5/8: data.dtypes
 5/9: data = pd.read_csv("data/cons_training.csv", parse_dates=[1,2])
5/10: data.dtypes
5/11: data.head()
5/12: data = pd.read_csv("data/cons_training.csv", parse_dates=[0,1])
5/13: data.head()
5/14: data.dtypes
5/15: data.head()
5/16: data.sort_values("start_time_utc").head()
5/17: data.sort_values("start_time_utc", ascending=False).head()
5/18: data.sort_values("start_time_utc", inplace=True).head()
5/19: data.sort_values("start_time_utc", inplace=True)
5/20: data.head()
5/21: data.isnull().sum()
5/22: data.isnull()
5/23: data[data.isnull()]
5/24: data.isnull().sum()
5/25: data[data['s101042'].isnull()]
5/26: data.isnull().sum()
5/27: data.columns()
5/28: data.columns
5/29:
targets = ['cons_actual_plus_umm', 'cons_actual_excl_umm']
feature_columns_not_used = ['start_time_utc', 'start_time_local', 'cons_fcast_fingrid_excl_umm']
feature_columns_to_use = [c for c in data.columns if c not in targets + feature_columns_not_used]
5/30: feature_columns_to_use
5/31: data.dropna().describe()
5/32: data.describe()
5/33:
data_na_dropped = data.dropna()
X = data_na_dropped[feature_columns_to_use].drop(targets, axis=1)
y = data_na_dropped[targets]
5/34: data_na_dropped.columns
5/35:
data_na_dropped = data.dropna()
X = data_na_dropped[feature_columns_to_use].drop(targets)
y = data_na_dropped[targets]
5/36:
data_na_dropped = data.dropna()
X = data_na_dropped.drop(targets)[feature_columns_to_use]
y = data_na_dropped.cons_actual_excl_umm
5/37: data_na_dropped.columns
5/38: data_na_dropped.drop(targets, axis=1)
5/39:
data_na_dropped = data.dropna()
X = data_na_dropped.drop(targets, axis=1)[feature_columns_to_use]
y = data_na_dropped.cons_actual_excl_umm
5/40:
targets = ['cons_actual_plus_umm', 'cons_actual_excl_umm']
feature_columns_not_used = ['start_time_utc', 'start_time_local', 'cons_fcast_fingrid_excl_umm']
feature_columns_to_use = [c for c in data.columns if c not in targets + feature_columns_not_used]
feature_columns_to_use
5/41:
data_na_dropped = data.dropna()
X = data_na_dropped.drop(targets, axis=1)[feature_columns_to_use]
y = data_na_dropped.cons_actual_excl_umm
5/42: X.columns
5/43: y.columns
5/44: y.head()
5/45: y.count
5/46: y.size()
5/47: y.length
5/48:
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
5/49:
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
5/50: data_na_dropped.dtypes
5/51: X.dtypes
5/52: train_X, test_X, train_y, test_y = train_test_split(X.as_matrix(), y.as_matrix(), test_size=0.2)
5/53: train_X, test_X, train_y, test_y = train_test_split(X.values, y.values, test_size=0.2)
5/54: train_X.head()
5/55: train_X[:5]
5/56: train_X.shape
5/57:
my_model = XGBRegressor()
# Add silent=True to avoid printing out updates with each cycle
my_model.fit(train_X, train_y, verbose=False)
5/58:
predictions = my_model.predict(test_X)

from sklearn.metrics import mean_absolute_error
print("Mean Absolute Error : " + str(mean_absolute_error(predictions, test_y)))
5/59:
targets = ['cons_actual_plus_umm', 'cons_actual_excl_umm']
feature_columns_not_used = ['start_time_utc', 'start_time_local', 'cons_fcast_fingrid_excl_umm']
feature_columns_to_use = [c for c in data.columns if c not in targets + feature_columns_not_used]
5/60:
predictions = my_model.predict(test_data[feature_columns_to_use].values)
print("Mean Absolute Error : " + str(mean_absolute_error(predictions, test_data.cons_actual_excl_umm.values)))
5/61:
data = pd.read_csv("data/cons_training.csv", parse_dates=[0,1])
test_data = pd.read_csv("data/cons_testing.csv", parse_dates=[0,1])
5/62:
predictions = my_model.predict(test_data[feature_columns_to_use].values)
print("Mean Absolute Error : " + str(mean_absolute_error(predictions, test_data.cons_actual_excl_umm.values)))
5/63:
my_model = XGBRegressor(n_estimators=1000)
# Add silent=True to avoid printing out updates with each cycle
my_model.fit(train_X, train_y, early_stopping_rounds=5, 
             eval_set=[(test_X, test_y)], verbose=False)
5/64:
predictions = my_model.predict(test_X[feature_columns_to_use].values)

from sklearn.metrics import mean_absolute_error
print("Mean Absolute Error : " + str(mean_absolute_error(predictions, test_y)))
5/65:
predictions = my_model.predict(test_X.values)

from sklearn.metrics import mean_absolute_error
print("Mean Absolute Error : " + str(mean_absolute_error(predictions, test_y)))
5/66:
predictions = my_model.predict(test_X)

from sklearn.metrics import mean_absolute_error
print("Mean Absolute Error : " + str(mean_absolute_error(predictions, test_y)))
5/67:
predictions = my_model.predict(test_data[feature_columns_to_use].values)
print("Mean Absolute Error : " + str(mean_absolute_error(predictions, test_data.cons_actual_excl_umm.values)))
5/68:
my_model = XGBRegressor(n_estimators=1000, max_depth=6)
# Add silent=True to avoid printing out updates with each cycle
my_model.fit(train_X, train_y, early_stopping_rounds=5, 
             eval_set=[(test_X, test_y)], verbose=False)
5/69:
predictions = my_model.predict(test_X)

from sklearn.metrics import mean_absolute_error
print("Mean Absolute Error : " + str(mean_absolute_error(predictions, test_y)))
5/70:
predictions = my_model.predict(test_data[feature_columns_to_use].values)
print("Mean Absolute Error : " + str(mean_absolute_error(predictions, test_data.cons_actual_excl_umm.values)))
5/71:
my_model = XGBRegressor(n_estimators=1000, max_depth=6, n_jobs=4)
# Add silent=True to avoid printing out updates with each cycle
my_model.fit(train_X, train_y, early_stopping_rounds=5, 
             eval_set=[(test_X, test_y)], verbose=False)
5/72:
predictions = my_model.predict(test_X)

from sklearn.metrics import mean_absolute_error
print("Mean Absolute Error : " + str(mean_absolute_error(predictions, test_y)))
5/73:
predictions = my_model.predict(test_data[feature_columns_to_use].values)
print("Mean Absolute Error : " + str(mean_absolute_error(predictions, test_data.cons_actual_excl_umm.values)))
5/74:
my_model = XGBRegressor(n_estimators=1000, max_depth=6, n_jobs=4, learning_rate=0.05)
# Add silent=True to avoid printing out updates with each cycle
my_model.fit(train_X, train_y, early_stopping_rounds=5, 
             eval_set=[(test_X, test_y)], verbose=False)
5/75:
predictions = my_model.predict(test_X)

from sklearn.metrics import mean_absolute_error
print("Mean Absolute Error : " + str(mean_absolute_error(predictions, test_y)))
5/76:
predictions = my_model.predict(test_data[feature_columns_to_use].values)
print("Mean Absolute Error : " + str(mean_absolute_error(predictions, test_data.cons_actual_excl_umm.values)))
5/77:
data.sort_values("start_time_utc", inplace=True)
data.dtypes
5/78:
data = pd.read_csv("data/cons_training.csv", parse_dates=[0,1])
test_data = pd.read_csv("data/cons_testing.csv", parse_dates=[0,1])
test_data.head()
5/79:
#TODO: add previous days/hours values, impute missing values for temps with last value, keras lstm in other notebook
#Add day of week as feature? Cross validation
test_X.head()
5/80:
#TODO: add previous days/hours values, impute missing values for temps with last value, keras lstm in other notebook
#Add day of week as feature? Cross validation
test_X[:5]
5/81: print("Mean Absolute Error : " + str(mean_absolute_error(X['cons_actual_24h_ago'].values, test_y)))
5/82: print("Mean Absolute Error : " + str(mean_absolute_error(test_X['cons_actual_24h_ago'].values, test_y)))
5/83: X.columns
5/84: X.columns[17]
5/85: X.columns[18]
5/86: X.columns[20]
5/87: X.columns[19]
5/88: test_X[:,19]
5/89: print("Mean Absolute Error : " + str(mean_absolute_error(test_X[:,19], test_y)))
 6/1:
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
 6/2:
data = pd.read_csv("data/cons_training.csv", parse_dates=[0,1])
data.sort_values("start_time_utc", inplace=True)
cons_test_data = pd.read_csv("data/cons_testing.csv", parse_dates=[0,1])
data.dtypes
 6/3: data.start_of_day_utc[0]
 6/4: data.start_time_utc[0]
 6/5: date.today(data.start_time_utc[0]).weekday()
 6/6:
import pandas as pd
import numpy as np
import xgboost as xgb
import datetime.datetime as date
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
 6/7: datetime.datetime.today(data.start_time_utc[0]).weekday()
 6/8:
import pandas as pd
import numpy as np
import xgboost as xgb
import datetime
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
 6/9: datetime.datetime.today(data.start_time_utc[0]).weekday()
6/10: datetime.datetime(data.start_time_utc[0])
6/11:
#datetime.datetime(data.start_time_utc[0])
datetime.datetime.today()
6/12: data.start_time_utc[0]
6/13: datetime.date.fromtimestamp(data.start_time_utc[0])
6/14: timestamp()
6/15:
import time
time.time()
6/16: data.start_time_utc[0]
6/17:
data.start_time_utc[0]
datetime.datetime.strptime(data.start_time_utc[0], "%Y-%m-%d %H:%M:%S")
6/18: daate = data.start_time_utc[0]
6/19: daate.dayofweek
6/20: daate = data.start_time_utc[0]
6/21: data.start_time_utc[0]
6/22:
def day_of_week(df):
    df['day_of_week'] = df.apply(lambda row: row.dayofweek, axis=1)
6/23:
def add_day_of_week(df):
    df['day_of_week'] = df.apply(lambda row: row.dayofweek, axis=1)
6/24:
def add_day_of_week(df):
    df['day_of_week'] = df.apply(lambda row: row.dayofweek, axis=1)
    return df
6/25: new_df = add_day_of_week(data)
6/26:
def add_day_of_week(df):
    df['day_of_week'] = df.apply(lambda row: row.dayofweek(), axis=1)
    return df
6/27: add_day_of_week(data)
6/28:
def add_day_of_week(df):
    df['day_of_week'] = df.apply(lambda row: row.start_time_local.dayofweek(), axis=1)
    return df
6/29: add_day_of_week(data)
6/30:
def add_day_of_week(df):
    df['day_of_week'] = df.apply(lambda row: row.start_time_local.dayofweek, axis=1)
    return df
6/31: add_day_of_week(data)
 7/1:
import pandas as pd
import numpy as np
import xgboost as xgb
import datetime
from xgboost.sklearn import XGBRegressor
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
 7/2:
import pandas as pd
import numpy as np
import datetime
import matplotlib.pylab as plt
%matplotlib inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4

import xgboost as xgb
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import Imputer
 7/3:
data = pd.read_csv("data/cons_training.csv", parse_dates=[0,1])
data.sort_values("start_time_utc", inplace=True)
cons_test_data = pd.read_csv("data/cons_testing.csv", parse_dates=[0,1])
data.dtypes
 7/4:
def pre_process(df):
    df['day_of_week'] = df.apply(lambda row: row.start_time_local.dayofweek, axis=1)
    return df
 7/5: data = pre_process(data)
 7/6: data.describe()
 7/7:
targets = ['cons_actual_plus_umm', 'cons_actual_excl_umm']
feature_columns_not_used = ['start_time_utc', 'start_time_local', 'cons_fcast_fingrid_excl_umm']
feature_columns_to_use = [c for c in data.columns if c not in targets + feature_columns_not_used]
 7/8:
data_na_dropped = data.dropna()
X = data_na_dropped.drop(targets, axis=1)[feature_columns_to_use]
y = data_na_dropped.cons_actual_excl_umm
#y2 = data_na_dropped.cons_actual_plus_umm
 7/9:
tuning_model = XGBRegressor(n_estimators=800, max_depth=6, n_jobs=1, learning_rate=0.05)

params = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5, 7]
        }

folds = 5
param_comb = 5

skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 1001)

random_search = RandomizedSearchCV(tuning_model, param_distributions=params, n_iter=param_comb, 
                                   scoring='mae', n_jobs=4, cv=skf.split(X, y), verbose=3)
7/10:
#Tuning from https://www.kaggle.com/tilii7/hyperparameter-grid-search-with-xgboost/notebook
def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))
7/11:
tuning_model = XGBRegressor(n_estimators=800, max_depth=6, n_jobs=1, learning_rate=0.05)

params = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5, 7]
        }

folds = 5
param_comb = 5

skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 1001)

random_search = RandomizedSearchCV(tuning_model, param_distributions=params, n_iter=param_comb, 
                                   scoring='mae', n_jobs=4, cv=skf.split(X, y), verbose=3)

start_time = timer(None) # timing starts from this point for "start_time" variable
random_search.fit(X, Y)
timer(start_time) # timing ends here for "start_time" variable
7/12:
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pylab as plt
%matplotlib inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4

import xgboost as xgb
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import Imputer
7/13:
tuning_model = XGBRegressor(n_estimators=800, max_depth=6, n_jobs=1, learning_rate=0.05)

params = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5, 7]
        }

folds = 5
param_comb = 5

skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 1001)

random_search = RandomizedSearchCV(tuning_model, param_distributions=params, n_iter=param_comb, 
                                   scoring='mae', n_jobs=4, cv=skf.split(X, y), verbose=3)

start_time = timer(None) # timing starts from this point for "start_time" variable
random_search.fit(X, Y)
timer(start_time) # timing ends here for "start_time" variable
7/14:
tuning_model = XGBRegressor(n_estimators=800, max_depth=6, n_jobs=1, learning_rate=0.05)

params = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5, 7]
        }

folds = 5
param_comb = 5

skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 1001)

random_search = RandomizedSearchCV(tuning_model, param_distributions=params, n_iter=param_comb, 
                                   scoring='mae', n_jobs=4, cv=skf.split(X, y), verbose=3)

start_time = timer(None) # timing starts from this point for "start_time" variable
random_search.fit(X, y)
timer(start_time) # timing ends here for "start_time" variable
7/15:
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pylab as plt
%matplotlib inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4

import xgboost as xgb
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import Imputer
7/16:
tuning_model = XGBRegressor(n_estimators=800, max_depth=6, n_jobs=1, learning_rate=0.05)

params = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5, 7]
        }

folds = 5
param_comb = 5

skf = KFold(n_splits=folds, shuffle = True, random_state = 1001)

random_search = RandomizedSearchCV(tuning_model, param_distributions=params, n_iter=param_comb, 
                                   scoring='mae', n_jobs=4, cv=skf.split(X, y), verbose=3)

start_time = timer(None) # timing starts from this point for "start_time" variable
random_search.fit(X, y)
timer(start_time) # timing ends here for "start_time" variable
7/17:
tuning_model = XGBRegressor(n_estimators=800, max_depth=6, n_jobs=1, learning_rate=0.05)

params = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5, 7]
        }

folds = 5
param_comb = 5

skf = KFold(n_splits=folds, shuffle = True, random_state = 1001)

random_search = RandomizedSearchCV(tuning_model, param_distributions=params, n_iter=param_comb, 
                                   scoring='neg_mean_absolute_error', n_jobs=4, cv=skf.split(X, y), verbose=3)

start_time = timer(None) # timing starts from this point for "start_time" variable
random_search.fit(X, y)
timer(start_time) # timing ends here for "start_time" variable
7/18:
print('Best MAE:')
print(random_search.best_score_)
print('\n Best hyperparameters:')
print(random_search.best_params_)
7/19:
train_X, test_X, train_y, test_y = train_test_split(X.values, y.values, test_size=0.2)

my_model = XGBRegressor(n_estimators=800, max_depth=5, n_jobs=4, gamma=0.5, colsample_bytree=0.6, 
                        min_child_weight=5, subsample=0.6)
# Add silent=True to avoid printing out updates with each cycle
my_model.fit(train_X, train_y, early_stopping_rounds=5, 
             eval_set=[(test_X, test_y)], verbose=False)

predictions = my_model.predict(test_X)

from sklearn.metrics import mean_absolute_error
print("Mean Absolute Error : " + str(mean_absolute_error(predictions, test_y)))
7/20:
train_X, test_X, train_y, test_y = train_test_split(X.values, y.values, test_size=0.2)

my_model = XGBRegressor(n_estimators=800, max_depth=3, n_jobs=4, gamma=2, colsample_bytree=0.8, 
                        min_child_weight=1, subsample=0.6)
# Add silent=True to avoid printing out updates with each cycle
my_model.fit(train_X, train_y, early_stopping_rounds=5, 
             eval_set=[(test_X, test_y)], verbose=False)

predictions = my_model.predict(test_X)

from sklearn.metrics import mean_absolute_error
print("Mean Absolute Error : " + str(mean_absolute_error(predictions, test_y)))
7/21:
tuning_model = XGBRegressor(n_estimators=800, n_jobs=1, learning_rate=0.1)

params = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 5, 7, 9]
        }

folds = 5
param_comb = 25

skf = KFold(n_splits=folds, shuffle = True)

random_search = RandomizedSearchCV(tuning_model, param_distributions=params, n_iter=param_comb, 
                                   scoring='neg_mean_absolute_error', n_jobs=4, cv=skf.split(X, y), verbose=3)

start_time = timer(None) # timing starts from this point for "start_time" variable
random_search.fit(X, y)
timer(start_time) # timing ends here for "start_time" variable
7/22:
print('Best negative MAE:')
print(random_search.best_score_)
print('\n Best hyperparameters:')
print(random_search.best_params_)
7/23:
train_X, test_X, train_y, test_y = train_test_split(X.values, y.values, test_size=0.2)

my_model = XGBRegressor(n_estimators=800, max_depth=9, n_jobs=4, gamma=1, colsample_bytree=0.6, 
                        min_child_weight=10, subsample=0.8)
# Add silent=True to avoid printing out updates with each cycle
my_model.fit(train_X, train_y, early_stopping_rounds=5, 
             eval_set=[(test_X, test_y)], eval_metric='mae', verbose=False)

predictions = my_model.predict(test_X)

from sklearn.metrics import mean_absolute_error
print("Mean Absolute Error : " + str(mean_absolute_error(predictions, test_y)))
7/24:
predictions = my_model.predict(cons_test_data[feature_columns_to_use].values)
cons_test_y = cons_test_data.cons_actual_excl_umm.values

from sklearn.metrics import mean_absolute_error
print("Mean Absolute Error : " + str(mean_absolute_error(predictions, cons_test_y)))
7/25:
cons_test_data = pre_process(cons_test_data)
predictions = my_model.predict(cons_test_data[feature_columns_to_use].values)
cons_test_y = cons_test_data.cons_actual_excl_umm.values

from sklearn.metrics import mean_absolute_error
print("Mean Absolute Error : " + str(mean_absolute_error(predictions, cons_test_y)))
7/26:
train_X, test_X, train_y, test_y = train_test_split(X.values, y.values, test_size=0.2)

my_model = XGBRegressor(n_estimators=800, max_depth=9, n_jobs=4, gamma=3, colsample_bytree=0.6, 
                        min_child_weight=10, subsample=0.8)
# Add silent=True to avoid printing out updates with each cycle
my_model.fit(train_X, train_y, early_stopping_rounds=5, 
             eval_set=[(test_X, test_y)], eval_metric='mae', verbose=False)

predictions = my_model.predict(test_X)

from sklearn.metrics import mean_absolute_error
print("Mean Absolute Error : " + str(mean_absolute_error(predictions, test_y)))
7/27:
cons_test_data = pre_process(cons_test_data)
predictions = my_model.predict(cons_test_data[feature_columns_to_use].values)
cons_test_y = cons_test_data.cons_actual_excl_umm.values

from sklearn.metrics import mean_absolute_error
print("Mean Absolute Error : " + str(mean_absolute_error(predictions, cons_test_y)))
7/28:
train_X, test_X, train_y, test_y = train_test_split(X.values, y.values, test_size=0.2)

my_model = XGBRegressor(n_estimators=800, max_depth=9, n_jobs=4, gamma=5, colsample_bytree=0.6, 
                        min_child_weight=10, subsample=0.8)
# Add silent=True to avoid printing out updates with each cycle
my_model.fit(train_X, train_y, early_stopping_rounds=5, 
             eval_set=[(test_X, test_y)], eval_metric='mae', verbose=False)

predictions = my_model.predict(test_X)

from sklearn.metrics import mean_absolute_error
print("Mean Absolute Error : " + str(mean_absolute_error(predictions, test_y)))
7/29:
cons_test_data = pre_process(cons_test_data)
predictions = my_model.predict(cons_test_data[feature_columns_to_use].values)
cons_test_y = cons_test_data.cons_actual_excl_umm.values

from sklearn.metrics import mean_absolute_error
print("Mean Absolute Error : " + str(mean_absolute_error(predictions, cons_test_y)))
7/30:
train_X, test_X, train_y, test_y = train_test_split(X.values, y.values, test_size=0.2)

my_model = XGBRegressor(n_estimators=800, max_depth=9, n_jobs=4, gamma=3, colsample_bytree=0.6, 
                        min_child_weight=10, subsample=0.8)
# Add silent=True to avoid printing out updates with each cycle
my_model.fit(train_X, train_y, early_stopping_rounds=5, 
             eval_set=[(test_X, test_y)], eval_metric='mae', verbose=False)

predictions = my_model.predict(test_X)

from sklearn.metrics import mean_absolute_error
print("Mean Absolute Error : " + str(mean_absolute_error(predictions, test_y)))
7/31:
cons_test_data = pre_process(cons_test_data)
predictions = my_model.predict(cons_test_data[feature_columns_to_use].values)
cons_test_y = cons_test_data.cons_actual_excl_umm.values

from sklearn.metrics import mean_absolute_error
print("Mean Absolute Error : " + str(mean_absolute_error(predictions, cons_test_y)))
7/32:
train_X, test_X, train_y, test_y = train_test_split(X.values, y.values, test_size=0.2)

my_model = XGBRegressor(n_estimators=800, max_depth=6, n_jobs=4, gamma=1, colsample_bytree=0.6, 
                        min_child_weight=10, subsample=0.8)
# Add silent=True to avoid printing out updates with each cycle
my_model.fit(train_X, train_y, early_stopping_rounds=5, 
             eval_set=[(test_X, test_y)], eval_metric='mae', verbose=False)

predictions = my_model.predict(test_X)

from sklearn.metrics import mean_absolute_error
print("Mean Absolute Error : " + str(mean_absolute_error(predictions, test_y)))
7/33:
cons_test_data = pre_process(cons_test_data)
predictions = my_model.predict(cons_test_data[feature_columns_to_use].values)
cons_test_y = cons_test_data.cons_actual_excl_umm.values

from sklearn.metrics import mean_absolute_error
print("Mean Absolute Error : " + str(mean_absolute_error(predictions, cons_test_y)))
7/34:
train_X, test_X, train_y, test_y = train_test_split(X.values, y.values, test_size=0.2)

my_model = XGBRegressor(n_estimators=800, random_search.best_params_)
# Add silent=True to avoid printing out updates with each cycle
my_model.fit(train_X, train_y, early_stopping_rounds=5, 
             eval_set=[(test_X, test_y)], eval_metric='mae', verbose=False)

predictions = my_model.predict(test_X)

from sklearn.metrics import mean_absolute_error
print("Mean Absolute Error : " + str(mean_absolute_error(predictions, test_y)))
7/35:
tuning_model = XGBRegressor(n_estimators=800, n_jobs=1, learning_rate=0.1)

params = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 5, 7, 9]
        }

folds = 8
param_comb = 25

skf = KFold(n_splits=folds, shuffle = True)

random_search = RandomizedSearchCV(tuning_model, param_distributions=params, n_iter=param_comb, 
                                   scoring='neg_mean_absolute_error', n_jobs=4, cv=skf.split(X, y), verbose=3)

start_time = timer(None) # timing starts from this point for "start_time" variable
random_search.fit(X, y, early_stopping_rounds=10)
timer(start_time) # timing ends here for "start_time" variable
7/36:
tuning_model = XGBRegressor(n_estimators=800, n_jobs=1, learning_rate=0.1, early_stopping_rounds=10)

params = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 5, 7, 9]
        }

folds = 8
param_comb = 25

skf = KFold(n_splits=folds, shuffle = True)

random_search = RandomizedSearchCV(tuning_model, param_distributions=params, n_iter=param_comb, 
                                   scoring='neg_mean_absolute_error', n_jobs=4, cv=skf.split(X, y), verbose=3)

start_time = timer(None) # timing starts from this point for "start_time" variable
random_search.fit(X, y)
timer(start_time) # timing ends here for "start_time" variable
7/37:
tuning_model = XGBRegressor(n_estimators=1000, n_jobs=1, learning_rate=0.07, early_stopping_rounds=10)

params = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 5, 7, 9, 13]
        }

folds = 5
param_comb = 25

skf = KFold(n_splits=folds, shuffle = True)

random_search = RandomizedSearchCV(tuning_model, param_distributions=params, n_iter=param_comb, 
                                   scoring='neg_mean_absolute_error', n_jobs=4, cv=skf.split(X, y), verbose=3)

start_time = timer(None) # timing starts from this point for "start_time" variable
random_search.fit(X, y)
timer(start_time) # timing ends here for "start_time" variable
7/38:
print('Best negative MAE:')
print(random_search.best_score_)
print('\n Best hyperparameters:')
print(random_search.best_params_)
7/39: random_search.best_estimator_
7/40:
train_X, test_X, train_y, test_y = train_test_split(X.values, y.values, test_size=0.2)

my_model = random_search.best_estimator_
# Add silent=True to avoid printing out updates with each cycle

predictions = my_model.predict(test_X)

from sklearn.metrics import mean_absolute_error
print("Mean Absolute Error : " + str(mean_absolute_error(predictions, test_y)))
7/41:
train_X, test_X, train_y, test_y = train_test_split(X.values, y.values, test_size=0.2)

my_model = XGBRegressor(n_estimators=1000, max_depth=13, n_jobs=4, gamma=0.5, colsample_bytree=0.8, 
                        min_child_weight=10, subsample=0.6, learning_rate=0.07)
# Add silent=True to avoid printing out updates with each cycle
my_model.fit(train_X, train_y, early_stopping_rounds=15, 
             eval_set=[(test_X, test_y)], verbose=False)

predictions = my_model.predict(test_X)

print("Mean Absolute Error : " + str(mean_absolute_error(predictions, test_y)))
7/42:
my_model.fit(train_X, train_y, early_stopping_rounds=10, 
             eval_set=[(test_X, test_y)], eval_metric='mae', verbose=False)
cons_test_data = pre_process(cons_test_data)
predictions = my_model.predict(cons_test_data[feature_columns_to_use].values)
cons_test_y = cons_test_data.cons_actual_excl_umm.values

print("Mean Absolute Error : " + str(mean_absolute_error(predictions, cons_test_y)))
7/43:
train_X, test_X, train_y, test_y = train_test_split(X.values, y.values, test_size=0.2)

my_model = XGBRegressor(n_estimators=1000, max_depth=13, n_jobs=4, gamma=0.5, colsample_bytree=0.8, 
                        min_child_weight=10, subsample=0.6, learning_rate=0.07)
# Add silent=True to avoid printing out updates with each cycle
my_model.fit(train_X, train_y, early_stopping_rounds=15, 
             eval_set=[(test_X, test_y)], verbose=False)

predictions = my_model.predict(test_X)

print("Mean Absolute Error : " + str(mean_absolute_error(predictions, test_y)))
7/44:
cons_test_data = pre_process(cons_test_data)
predictions = my_model.predict(cons_test_data[feature_columns_to_use].values)
cons_test_y = cons_test_data.cons_actual_excl_umm.values

print("Mean Absolute Error : " + str(mean_absolute_error(predictions, cons_test_y)))
7/45:
train_X, test_X, train_y, test_y = train_test_split(X.values, y.values, test_size=0.2)

my_model = XGBRegressor(n_estimators=1000, max_depth=13, n_jobs=4, gamma=2, colsample_bytree=0.8, 
                        min_child_weight=10, subsample=0.6, learning_rate=0.07)
# Add silent=True to avoid printing out updates with each cycle
my_model.fit(train_X, train_y, early_stopping_rounds=15, 
             eval_set=[(test_X, test_y)], verbose=False)

predictions = my_model.predict(test_X)

print("Mean Absolute Error : " + str(mean_absolute_error(predictions, test_y)))
7/46:
cons_test_data = pre_process(cons_test_data)
predictions = my_model.predict(cons_test_data[feature_columns_to_use].values)
cons_test_y = cons_test_data.cons_actual_excl_umm.values

print("Mean Absolute Error : " + str(mean_absolute_error(predictions, cons_test_y)))
7/47:
train_X, test_X, train_y, test_y = train_test_split(X.values, y.values, test_size=0.2)

my_model = XGBRegressor(n_estimators=1000, max_depth=13, n_jobs=4, gamma=4, colsample_bytree=0.8, 
                        min_child_weight=10, subsample=0.6, learning_rate=0.07)
# Add silent=True to avoid printing out updates with each cycle
my_model.fit(train_X, train_y, early_stopping_rounds=15, 
             eval_set=[(test_X, test_y)], verbose=False)

predictions = my_model.predict(test_X)

print("Mean Absolute Error : " + str(mean_absolute_error(predictions, test_y)))
7/48:
cons_test_data = pre_process(cons_test_data)
predictions = my_model.predict(cons_test_data[feature_columns_to_use].values)
cons_test_y = cons_test_data.cons_actual_excl_umm.values

print("Mean Absolute Error : " + str(mean_absolute_error(predictions, cons_test_y)))
7/49:
train_X, test_X, train_y, test_y = train_test_split(X.values, y.values, test_size=0.2)

my_model = XGBRegressor(n_estimators=1000, max_depth=11, n_jobs=4, gamma=3, colsample_bytree=0.6, 
                        min_child_weight=10, subsample=0.8, learning_rate=0.07)
# Add silent=True to avoid printing out updates with each cycle
my_model.fit(train_X, train_y, early_stopping_rounds=15, 
             eval_set=[(test_X, test_y)], verbose=False)

predictions = my_model.predict(test_X)

print("Mean Absolute Error : " + str(mean_absolute_error(predictions, test_y)))
7/50:
cons_test_data = pre_process(cons_test_data)
predictions = my_model.predict(cons_test_data[feature_columns_to_use].values)
cons_test_y = cons_test_data.cons_actual_excl_umm.values

print("Mean Absolute Error : " + str(mean_absolute_error(predictions, cons_test_y)))
7/51:
train_X, test_X, train_y, test_y = train_test_split(X.values, y.values, test_size=0.2)

my_model = XGBRegressor(n_estimators=1000, max_depth=11, n_jobs=4, gamma=3, colsample_bytree=0.6, 
                        min_child_weight=10, subsample=0.8, learning_rate=0.15)
# Add silent=True to avoid printing out updates with each cycle
my_model.fit(train_X, train_y, early_stopping_rounds=15, 
             eval_set=[(test_X, test_y)], verbose=False)

predictions = my_model.predict(test_X)

print("Mean Absolute Error : " + str(mean_absolute_error(predictions, test_y)))
7/52:
cons_test_data = pre_process(cons_test_data)
predictions = my_model.predict(cons_test_data[feature_columns_to_use].values)
cons_test_y = cons_test_data.cons_actual_excl_umm.values

print("Mean Absolute Error : " + str(mean_absolute_error(predictions, cons_test_y)))
7/53:
train_X, test_X, train_y, test_y = train_test_split(X.values, y.values, test_size=0.2)

my_model = XGBRegressor(n_estimators=1000, max_depth=11, n_jobs=4, gamma=3, colsample_bytree=0.6, 
                        min_child_weight=10, subsample=0.8, learning_rate=0.1)
# Add silent=True to avoid printing out updates with each cycle
my_model.fit(train_X, train_y, early_stopping_rounds=15, 
             eval_set=[(test_X, test_y)], verbose=False)

predictions = my_model.predict(test_X)

print("Mean Absolute Error : " + str(mean_absolute_error(predictions, test_y)))
7/54:
cons_test_data = pre_process(cons_test_data)
predictions = my_model.predict(cons_test_data[feature_columns_to_use].values)
cons_test_y = cons_test_data.cons_actual_excl_umm.values

print("Mean Absolute Error : " + str(mean_absolute_error(predictions, cons_test_y)))
7/55:
train_X, test_X, train_y, test_y = train_test_split(X.values, y.values, test_size=0.2)

my_model = XGBRegressor(n_estimators=1000, max_depth=9, n_jobs=4, gamma=3, colsample_bytree=0.6, 
                        min_child_weight=10, subsample=0.8, learning_rate=0.1)
# Add silent=True to avoid printing out updates with each cycle
my_model.fit(train_X, train_y, early_stopping_rounds=15, 
             eval_set=[(test_X, test_y)], verbose=False)

predictions = my_model.predict(test_X)

print("Mean Absolute Error : " + str(mean_absolute_error(predictions, test_y)))
7/56:
cons_test_data = pre_process(cons_test_data)
predictions = my_model.predict(cons_test_data[feature_columns_to_use].values)
cons_test_y = cons_test_data.cons_actual_excl_umm.values

print("Mean Absolute Error : " + str(mean_absolute_error(predictions, cons_test_y)))
7/57:
train_X, test_X, train_y, test_y = train_test_split(X.values, y.values, test_size=0.2)

my_model = XGBRegressor(n_estimators=1000, max_depth=9, n_jobs=4, gamma=3, colsample_bytree=0.6, 
                        min_child_weight=10, subsample=0.8, learning_rate=0.1)
# Add silent=True to avoid printing out updates with each cycle
my_model.fit(train_X, train_y, early_stopping_rounds=5, 
             eval_set=[(test_X, test_y)], verbose=False)

predictions = my_model.predict(test_X)

print("Mean Absolute Error : " + str(mean_absolute_error(predictions, test_y)))
7/58:
cons_test_data = pre_process(cons_test_data)
predictions = my_model.predict(cons_test_data[feature_columns_to_use].values)
cons_test_y = cons_test_data.cons_actual_excl_umm.values

print("Mean Absolute Error : " + str(mean_absolute_error(predictions, cons_test_y)))
7/59:
train_X, test_X, train_y, test_y = train_test_split(X.values, y.values, test_size=0.2)

my_model = XGBRegressor(n_estimators=1000, max_depth=3, n_jobs=4, gamma=1.5, colsample_bytree=1, 
                        min_child_weight=1, subsample=1, learning_rate=0.07)
# Add silent=True to avoid printing out updates with each cycle
my_model.fit(train_X, train_y, early_stopping_rounds=5, 
             eval_set=[(test_X, test_y)], verbose=False)

predictions = my_model.predict(test_X)

print("Mean Absolute Error : " + str(mean_absolute_error(predictions, test_y)))
7/60:
cons_test_data = pre_process(cons_test_data)
predictions = my_model.predict(cons_test_data[feature_columns_to_use].values)
cons_test_y = cons_test_data.cons_actual_excl_umm.values

print("Mean Absolute Error : " + str(mean_absolute_error(predictions, cons_test_y)))
7/61:
train_X, test_X, train_y, test_y = train_test_split(X.values, y.values, test_size=0.2)

my_model = XGBRegressor(n_estimators=10000, max_depth=9, n_jobs=4, gamma=3, colsample_bytree=0.6, 
                        min_child_weight=10, subsample=0.8, learning_rate=0.1)
# Add silent=True to avoid printing out updates with each cycle
my_model.fit(train_X, train_y, early_stopping_rounds=10, 
             eval_set=[(test_X, test_y)], verbose=False)

predictions = my_model.predict(test_X)

print("Mean Absolute Error : " + str(mean_absolute_error(predictions, test_y)))
7/62:
cons_test_data = pre_process(cons_test_data)
predictions = my_model.predict(cons_test_data[feature_columns_to_use].values)
cons_test_y = cons_test_data.cons_actual_excl_umm.values

print("Mean Absolute Error : " + str(mean_absolute_error(predictions, cons_test_y)))
7/63:
train_X, test_X, train_y, test_y = train_test_split(X.values, y.values, test_size=0.2)

my_model = XGBRegressor(n_estimators=10000, max_depth=9, n_jobs=4, gamma=3, colsample_bytree=0.6, 
                        min_child_weight=10, subsample=0.8, learning_rate=0.1)
# Add silent=True to avoid printing out updates with each cycle
my_model.fit(train_X, train_y, early_stopping_rounds=10, 
             eval_set=[(test_X, test_y)], eval_metric='mae' verbose=False)

predictions = my_model.predict(test_X)

print("Mean Absolute Error : " + str(mean_absolute_error(predictions, test_y)))
7/64:
train_X, test_X, train_y, test_y = train_test_split(X.values, y.values, test_size=0.2)

my_model = XGBRegressor(n_estimators=10000, max_depth=9, n_jobs=4, gamma=3, colsample_bytree=0.6, 
                        min_child_weight=10, subsample=0.8, learning_rate=0.1)
# Add silent=True to avoid printing out updates with each cycle
my_model.fit(train_X, train_y, early_stopping_rounds=10, 
             eval_set=[(test_X, test_y)], eval_metric='mae', verbose=False)

predictions = my_model.predict(test_X)

print("Mean Absolute Error : " + str(mean_absolute_error(predictions, test_y)))
7/65:
train_X, test_X, train_y, test_y = train_test_split(X.values, y.values, test_size=0.2)

my_model = XGBRegressor(n_estimators=10000, max_depth=9, n_jobs=4, gamma=3, colsample_bytree=0.6, 
                        min_child_weight=10, subsample=0.8, learning_rate=0.1)
# Add silent=True to avoid printing out updates with each cycle
my_model.fit(train_X, train_y, early_stopping_rounds=10, 
             eval_set=[(test_X, test_y)], eval_metric='mae', verbose=True)

predictions = my_model.predict(test_X)

print("Mean Absolute Error : " + str(mean_absolute_error(predictions, test_y)))
7/66:
cons_test_data = pre_process(cons_test_data)
predictions = my_model.predict(cons_test_data[feature_columns_to_use].values)
cons_test_y = cons_test_data.cons_actual_excl_umm.values

print("Mean Absolute Error : " + str(mean_absolute_error(predictions, cons_test_y)))
7/67:
train_X, test_X, train_y, test_y = train_test_split(X.values, y.values, test_size=0.2)

my_model = XGBRegressor(n_estimators=10000, max_depth=9, n_jobs=4, gamma=3, colsample_bytree=0.6, 
                        min_child_weight=10, subsample=0.8, learning_rate=0.05)
# Add silent=True to avoid printing out updates with each cycle
my_model.fit(train_X, train_y, early_stopping_rounds=10, 
             eval_set=[(test_X, test_y)], eval_metric='mae', verbose=True)

predictions = my_model.predict(test_X)

print("Mean Absolute Error : " + str(mean_absolute_error(predictions, test_y)))
7/68:
cons_test_data = pre_process(cons_test_data)
predictions = my_model.predict(cons_test_data[feature_columns_to_use].values)
cons_test_y = cons_test_data.cons_actual_excl_umm.values

print("Mean Absolute Error : " + str(mean_absolute_error(predictions, cons_test_y)))
7/69:
tuning_model = XGBRegressor(n_estimators=1000, n_jobs=1, learning_rate=0.07, early_stopping_rounds=10, verbose=True)

params = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 5, 7, 9, 13]
        }

folds = 5
param_comb = 2

skf = KFold(n_splits=folds, shuffle = True)

random_search = RandomizedSearchCV(tuning_model, param_distributions=params, n_iter=param_comb, 
                                   scoring='neg_mean_absolute_error', n_jobs=4, cv=skf.split(X, y), verbose=3)

start_time = timer(None) # timing starts from this point for "start_time" variable
random_search.fit(X, y)
timer(start_time) # timing ends here for "start_time" variable
7/70:
train_X, test_X, train_y, test_y = train_test_split(X.values, y.values, test_size=0.2)

my_model = XGBRegressor(n_estimators=10000, max_depth=9, n_jobs=4, gamma=0, colsample_bytree=0.6, 
                        min_child_weight=10, subsample=0.8, learning_rate=0.05)
# Add silent=True to avoid printing out updates with each cycle
my_model.fit(train_X, train_y, early_stopping_rounds=10, 
             eval_set=[(test_X, test_y)], eval_metric='mae', verbose=True)

predictions = my_model.predict(test_X)

print("Mean Absolute Error : " + str(mean_absolute_error(predictions, test_y)))
 8/1:
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pylab as plt
%matplotlib inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4

import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import Imputer
 8/2:
data = pd.read_csv("data/cons_training.csv", parse_dates=[0,1])
data.sort_values("start_time_utc", inplace=True)
cons_test_data = pd.read_csv("data/cons_testing.csv", parse_dates=[0,1])
data.dtypes
 8/3:
def pre_process(df):
    df['day_of_week'] = df.apply(lambda row: row.start_time_local.dayofweek, axis=1)
    return df
 8/4: data = pre_process(data)
 8/5:
targets = ['cons_actual_plus_umm', 'cons_actual_excl_umm']
feature_columns_not_used = ['start_time_utc', 'start_time_local', 'cons_fcast_fingrid_excl_umm']
feature_columns_to_use = [c for c in data.columns if c not in targets + feature_columns_not_used]
 8/6:
data_na_dropped = data.dropna()
X = data_na_dropped.drop(targets, axis=1)[feature_columns_to_use]
y = data_na_dropped.cons_actual_excl_umm
#y2 = data_na_dropped.cons_actual_plus_umm
 8/7:
train_X, test_X, train_y, test_y = train_test_split(X.values, y.values, test_size=0.2)

my_model = XGBRegressor(n_estimators=1500, max_depth=9, n_jobs=4, gamma=1, colsample_bytree=0.6, 
                        min_child_weight=10, subsample=0.8)
# Add silent=True to avoid printing out updates with each cycle
my_model.fit(train_X, train_y, early_stopping_rounds=5, 
             eval_set=[(test_X, test_y)], eval_metric='mae', verbose=True)

predictions = my_model.predict(test_X)

from sklearn.metrics import mean_absolute_error
print("Mean Absolute Error : " + str(mean_absolute_error(predictions, test_y)))
 8/8:
train_X, test_X, train_y, test_y = train_test_split(X.values, y.values, test_size=0.2)

my_model = XGBRegressor(n_estimators=1500, max_depth=9, n_jobs=4, gamma=3, colsample_bytree=0.6, 
                        min_child_weight=10, subsample=0.8)
# Add silent=True to avoid printing out updates with each cycle
my_model.fit(train_X, train_y, early_stopping_rounds=5, 
             eval_set=[(test_X, test_y)], eval_metric='mae', verbose=True)

predictions = my_model.predict(test_X)

from sklearn.metrics import mean_absolute_error
print("Mean Absolute Error : " + str(mean_absolute_error(predictions, test_y)))
 9/1:
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pylab as plt
%matplotlib inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4

import xgboost as xgb
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import Imputer
 9/2:
data = pd.read_csv("data/cons_training.csv", parse_dates=[0,1])
data.sort_values("start_time_utc", inplace=True)
cons_test_data = pd.read_csv("data/cons_testing.csv", parse_dates=[0,1])
data.dtypes
 9/3:
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pylab as plt
%matplotlib inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4

import xgboost as xgb
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import Imputer
from sklearn.ensemble import RandomForestRegressor
 9/4:
data = pd.read_csv("data/cons_training.csv", parse_dates=[0,1])
data.sort_values("start_time_utc", inplace=True)
cons_test_data = pd.read_csv("data/cons_testing.csv", parse_dates=[0,1])
data.dtypes
 9/5:
def pre_process(df):
    df['day_of_week'] = df.apply(lambda row: row.start_time_local.dayofweek, axis=1)
    return df
 9/6: data = pre_process(data)
 9/7:
targets = ['cons_actual_plus_umm', 'cons_actual_excl_umm']
feature_columns_not_used = ['start_time_utc', 'start_time_local', 'cons_fcast_fingrid_excl_umm']
feature_columns_to_use = [c for c in data.columns if c not in targets + feature_columns_not_used]
 9/8:
data_na_dropped = data.dropna()
X = data_na_dropped.drop(targets, axis=1)[feature_columns_to_use]
y = data_na_dropped.cons_actual_excl_umm
#y2 = data_na_dropped.cons_actual_plus_umm
 9/9: train_X, test_X, train_y, test_y = train_test_split(X.values, y.values, test_size=0.2)
9/10:
regr = RandomForestRegressor(max_depth=5, random_state=0)
rf_model = regr.fit(train_X, train_y)
9/11:
regr = RandomForestRegressor(max_depth=5, random_state=0)
rf_model = regr.fit(train_X, train_y)
preds = rf_model.predict(test_X)
print("Mean Absolute Error : " + str(mean_absolute_error(preds, test_y)))
9/12:
regr = RandomForestRegressor()
rf_model = regr.fit(train_X, train_y)
preds = rf_model.predict(test_X)
print("Mean Absolute Error : " + str(mean_absolute_error(preds, test_y)))
9/13:
regr = RandomForestRegressor(n_estimators=40)
rf_model = regr.fit(train_X, train_y)
preds = rf_model.predict(test_X)
print("Mean Absolute Error : " + str(mean_absolute_error(preds, test_y)))
9/14: rf_model.feature_importances_
9/15: X.columns
10/1:
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pylab as plt
%matplotlib inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4

import xgboost as xgb
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import Imputer
10/2:
data = pd.read_csv("data/cons_training.csv", parse_dates=[0,1])
data.sort_values("start_time_utc", inplace=True)
cons_test_data = pd.read_csv("data/cons_testing.csv", parse_dates=[0,1])
data.dtypes
10/3:
data = pd.read_csv("data/cons_training.csv", parse_dates=[0,1])
data.sort_values("start_time_utc", inplace=True)
cons_test_data = pd.read_csv("data/cons_testing_without_labels.csv", parse_dates=[0,1])
data.dtypes
10/4:
def pre_process(df):
    df['day_of_week'] = df.apply(lambda row: row.start_time_local.dayofweek, axis=1)
    return df
10/5:
#Baseline: predict the current value from the last day's value: y[i] = y[i-24]?
data = pre_process(data)
data.describe()
10/6:
targets = ['cons_actual_plus_umm', 'cons_actual_excl_umm']
features_not_used = ['start_time_utc', 'start_time_local', 'cons_fcast_fingrid_excl_umm']
features_to_use = [c for c in data.columns if c not in targets + features_not_used]
10/7:
data_na_dropped = data.dropna()
X = data_na_dropped.drop(targets, axis=1)[features_to_use]
y = data_na_dropped.cons_actual_excl_umm
#y2 = data_na_dropped.cons_actual_plus_umm
10/8: train_X, test_X, train_y, test_y = train_test_split(X.values, y.values, test_size=0.2)
10/9:
#Tuning from https://www.kaggle.com/tilii7/hyperparameter-grid-search-with-xgboost/notebook
def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))

tuning_model = XGBRegressor(n_estimators=800, n_jobs=1, learning_rate=0.1)

params = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 5, 7, 9]
        }

folds = 5
param_comb = 25

skf = KFold(n_splits=folds, shuffle = True)

random_search = RandomizedSearchCV(tuning_model, param_distributions=params, n_iter=param_comb, 
                                   scoring='neg_mean_absolute_error', n_jobs=4, cv=skf.split(X, y), verbose=3)

start_time = timer(None) # timing starts from this point for "start_time" variable
random_search.fit(X, y)
timer(start_time) # timing ends here for "start_time" variable
10/10:
#TODO: Try doing own random search because the sklearn doesn't support early stopping? or does it?
iterations = 2
evals = []
for i in range(iterations):
    params_chosen = {}
    params_chosen['objective'] = 'reg:linear'
    params_chosen['eval_metric'] = 'mae'
    params_chosen['min_child_weight'] = np.random.choice(params['min_child_weight'])
    params_chosen['gamma'] = np.random.choice(params['gamma'])
    params_chosen['subsample'] = np.random.choice(params['subsample'])
    params_chosen['colsample_bytree'] = np.random.choice(params['colsample_bytree'])
    params_chosen['max_depth'] = np.random.choice(params['max_depth'])
    print("Parameters for this run: ", params_chosen)
    print("Starting CV: ")
    cv = xgb.cv(params_chosen, early_stopping_rounds=12, metrics=['mae'], nfold=5)
    evals.append(cv)
10/11:
#TODO: Try doing own random search because the sklearn doesn't support early stopping? or does it?
iterations = 2
evals = []
xgb_X = data_na_dropped.(['cons_actual_excl_umm'], axis=1)[features_to_use]
xgb_X = xgb.DMatrix(xgb_X, label='cons_actual_plus_umm')

for i in range(iterations):
    params_chosen = {}
    params_chosen['objective'] = 'reg:linear'
    params_chosen['eval_metric'] = 'mae'
    params_chosen['min_child_weight'] = np.random.choice(params['min_child_weight'])
    params_chosen['gamma'] = np.random.choice(params['gamma'])
    params_chosen['subsample'] = np.random.choice(params['subsample'])
    params_chosen['colsample_bytree'] = np.random.choice(params['colsample_bytree'])
    params_chosen['max_depth'] = np.random.choice(params['max_depth'])
    print("Parameters for this run: ", params_chosen)
    print("Starting CV: ")
    cv = xgb.cv(params_chosen, xgb_X, early_stopping_rounds=12, metrics=['mae'], nfold=5)
    evals.append(cv)
10/12:
#TODO: Try doing own random search because the sklearn doesn't support early stopping? or does it?
iterations = 2
evals = []
xgb_X = data_na_dropped.drop(['cons_actual_excl_umm'], axis=1)[features_to_use]
xgb_X = xgb.DMatrix(xgb_X, label='cons_actual_plus_umm')

for i in range(iterations):
    params_chosen = {}
    params_chosen['objective'] = 'reg:linear'
    params_chosen['eval_metric'] = 'mae'
    params_chosen['min_child_weight'] = np.random.choice(params['min_child_weight'])
    params_chosen['gamma'] = np.random.choice(params['gamma'])
    params_chosen['subsample'] = np.random.choice(params['subsample'])
    params_chosen['colsample_bytree'] = np.random.choice(params['colsample_bytree'])
    params_chosen['max_depth'] = np.random.choice(params['max_depth'])
    print("Parameters for this run: ", params_chosen)
    print("Starting CV: ")
    cv = xgb.cv(params_chosen, xgb_X, early_stopping_rounds=12, metrics=['mae'], nfold=5)
    evals.append(cv)
10/13:
#TODO: Try doing own random search because the sklearn doesn't support early stopping? or does it?
iterations = 2
evals = []
xgb_X = data_na_dropped.drop(['cons_actual_excl_umm'], axis=1)[features_to_use]
xgb_X = xgb.DMatrix(X, label=y.values)

for i in range(iterations):
    params_chosen = {}
    params_chosen['objective'] = 'reg:linear'
    params_chosen['eval_metric'] = 'mae'
    params_chosen['min_child_weight'] = np.random.choice(params['min_child_weight'])
    params_chosen['gamma'] = np.random.choice(params['gamma'])
    params_chosen['subsample'] = np.random.choice(params['subsample'])
    params_chosen['colsample_bytree'] = np.random.choice(params['colsample_bytree'])
    params_chosen['max_depth'] = np.random.choice(params['max_depth'])
    print("Parameters for this run: ", params_chosen)
    print("Starting CV: ")
    cv = xgb.cv(params_chosen, xgb_X, early_stopping_rounds=12, metrics=['mae'], nfold=5)
    evals.append(cv)
10/14: evals[0]
10/15:
#TODO: Try doing own random search because the sklearn doesn't support early stopping? or does it?
iterations = 2
evals = []
xgb_X = data_na_dropped.drop(['cons_actual_excl_umm'], axis=1)[features_to_use]
xgb_X = xgb.DMatrix(X, label=y.values)

for i in range(iterations):
    params_chosen = {}
    params_chosen['objective'] = 'reg:linear'
    params_chosen['eval_metric'] = 'mae'
    params_chosen['min_child_weight'] = np.random.choice(params['min_child_weight'])
    params_chosen['gamma'] = np.random.choice(params['gamma'])
    params_chosen['subsample'] = np.random.choice(params['subsample'])
    params_chosen['colsample_bytree'] = np.random.choice(params['colsample_bytree'])
    params_chosen['max_depth'] = np.random.choice(params['max_depth'])
    print("Parameters for this run: ", params_chosen)
    print("Starting CV: ")
    cv = xgb.cv(params_chosen, num_boost_round=1000, xgb_X, early_stopping_rounds=12, metrics=['mae'], nfold=5)
    evals.append(cv)
10/16:
#TODO: Try doing own random search because the sklearn doesn't support early stopping? or does it?
iterations = 2
evals = []
xgb_X = data_na_dropped.drop(['cons_actual_excl_umm'], axis=1)[features_to_use]
xgb_X = xgb.DMatrix(X, label=y.values)

for i in range(iterations):
    params_chosen = {}
    params_chosen['objective'] = 'reg:linear'
    params_chosen['eval_metric'] = 'mae'
    params_chosen['min_child_weight'] = np.random.choice(params['min_child_weight'])
    params_chosen['gamma'] = np.random.choice(params['gamma'])
    params_chosen['subsample'] = np.random.choice(params['subsample'])
    params_chosen['colsample_bytree'] = np.random.choice(params['colsample_bytree'])
    params_chosen['max_depth'] = np.random.choice(params['max_depth'])
    print("Parameters for this run: ", params_chosen)
    print("Starting CV: ")
    cv = xgb.cv(params_chosen, xgb_X, num_boost_round=1000, early_stopping_rounds=12, metrics=['mae'], nfold=5)
    evals.append(cv)
10/17: len(evals[0])
10/18: evals[999]
10/19: evals[800]
10/20: evals[0][999}
10/21: evals[0][999]
10/22: evals[0]
10/23:
#TODO: Try doing own random search because the sklearn doesn't support early stopping? or does it?
iterations = 2
evals = []
xgb_X = data_na_dropped.drop(['cons_actual_excl_umm'], axis=1)[features_to_use]
xgb_X = xgb.DMatrix(X, label=y.values)

for i in range(iterations):
    np.random.seed()
    params_chosen = {}
    params_chosen['objective'] = 'reg:linear'
    params_chosen['eval_metric'] = 'mae'
    params_chosen['min_child_weight'] = np.random.choice(params['min_child_weight'])
    params_chosen['gamma'] = np.random.choice(params['gamma'])
    params_chosen['subsample'] = np.random.choice(params['subsample'])
    params_chosen['colsample_bytree'] = np.random.choice(params['colsample_bytree'])
    params_chosen['max_depth'] = np.random.choice(params['max_depth'])
    print("Parameters for this run: ", params_chosen)
    print("Starting CV: ")
    cv = xgb.cv(params_chosen, xgb_X, num_boost_round=100, early_stopping_rounds=12, metrics=['mae'], nfold=4)
    evals.append(cv)
10/24:
#TODO: Try doing own random search because the sklearn doesn't support early stopping? or does it?
iterations = 2
evals = []
xgb_X = data_na_dropped.drop(['cons_actual_excl_umm'], axis=1)[features_to_use]
xgb_X = xgb.DMatrix(X, label=y.values)

for i in range(iterations):
    np.random.seed()
    params_chosen = {}
    params_chosen['objective'] = 'reg:linear'
    params_chosen['eval_metric'] = 'mae'
    params_chosen['min_child_weight'] = np.random.choice(params['min_child_weight'])
    params_chosen['gamma'] = np.random.choice(params['gamma'])
    params_chosen['subsample'] = np.random.choice(params['subsample'])
    params_chosen['colsample_bytree'] = np.random.choice(params['colsample_bytree'])
    params_chosen['max_depth'] = np.random.choice(params['max_depth'])
    print("Parameters for this run: ", params_chosen)
    print("Starting CV: ")
    cv = xgb.cv(params_chosen, xgb_X, num_boost_round=2000, early_stopping_rounds=12, metrics=['mae'], nfold=4, n_jobs=4)
    evals.append(cv)
10/25:
#TODO: Try doing own random search because the sklearn doesn't support early stopping? or does it?
iterations = 2
evals = []
xgb_X = data_na_dropped.drop(['cons_actual_excl_umm'], axis=1)[features_to_use]
xgb_X = xgb.DMatrix(X, label=y.values)

for i in range(iterations):
    np.random.seed()
    params_chosen = {}
    params_chosen['objective'] = 'reg:linear'
    params_chosen['n_jobs'] = 4
    params_chosen['eval_metric'] = 'mae'
    params_chosen['min_child_weight'] = np.random.choice(params['min_child_weight'])
    params_chosen['gamma'] = np.random.choice(params['gamma'])
    params_chosen['subsample'] = np.random.choice(params['subsample'])
    params_chosen['colsample_bytree'] = np.random.choice(params['colsample_bytree'])
    params_chosen['max_depth'] = np.random.choice(params['max_depth'])
    print("Parameters for this run: ", params_chosen)
    print("Starting CV: ")
    cv = xgb.cv(params_chosen, xgb_X, num_boost_round=2000, early_stopping_rounds=12, metrics=['mae'], nfold=4)
    evals.append(cv)
10/26: evals[0]
10/27: evals[1
10/28: evals[1]
11/1:
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pylab as plt
%matplotlib inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4

import xgboost as xgb
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import Imputer
11/2:
data = pd.read_csv("data/cons_training.csv", parse_dates=[0,1])
data.sort_values("start_time_utc", inplace=True)
cons_test_data = pd.read_csv("data/cons_testing.csv", parse_dates=[0,1])
data.dtypes
11/3:
data = pd.read_csv("data/cons_training.csv", parse_dates=[0,1])
data.sort_values("start_time_utc", inplace=True)
cons_test_data = pd.read_csv("data/cons_testing_without_labels.csv", parse_dates=[0,1])
data.dtypes
11/4:
def pre_process(df):
    df['day_of_week'] = df.apply(lambda row: row.start_time_local.dayofweek, axis=1)
    return df

#Baseline: predict the current value from the last day's value: y[i] = y[i-24]?
data = pre_process(data)
data.describe()
11/5:
targets = ['cons_actual_plus_umm', 'cons_actual_excl_umm']
features_not_used = ['start_time_utc', 'start_time_local', 'cons_fcast_fingrid_excl_umm']
features_to_use = [c for c in data.columns if c not in targets + features_not_used]
11/6:
data = data.dropna()
X = data.drop(targets, axis=1)[features_to_use]
y = data.cons_actual_excl_umm
#y2 = data.cons_actual_plus_umm
11/7: train_X, test_X, train_y, test_y = train_test_split(X.values, y.values, test_size=0.2)
11/8:
#Tuning from https://www.kaggle.com/tilii7/hyperparameter-grid-search-with-xgboost/notebook
def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))

tuning_model = XGBRegressor(n_estimators=1500, n_jobs=1, learning_rate=0.1)

params = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 5, 7, 9],
        'learning_rate'
        }

folds = 4
param_comb = 25

kf = KFold(n_splits=folds, shuffle = True)

random_search = RandomizedSearchCV(tuning_model, param_distributions=params, n_iter=param_comb, 
                                   scoring='neg_mean_absolute_error', n_jobs=4, cv=kf.split(X, y), verbose=3)

start_time = timer(None) # timing starts from this point for "start_time" variable
random_search.fit(X, y)
timer(start_time) # timing ends here for "start_time" variable
11/9:
#Tuning from https://www.kaggle.com/tilii7/hyperparameter-grid-search-with-xgboost/notebook
def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))

tuning_model = XGBRegressor(n_estimators=1500, n_jobs=1, learning_rate=0.1)

params = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 5, 7, 9]
        }

folds = 4
param_comb = 25

kf = KFold(n_splits=folds, shuffle = True)

random_search = RandomizedSearchCV(tuning_model, param_distributions=params, n_iter=param_comb, 
                                   scoring='neg_mean_absolute_error', n_jobs=4, cv=kf.split(X, y), verbose=3)

start_time = timer(None) # timing starts from this point for "start_time" variable
random_search.fit(X, y)
timer(start_time) # timing ends here for "start_time" variable
11/10:
print('Best negative MAE:')
print(random_search.best_score_)
print('\n Best hyperparameters:')
print(random_search.best_params_)
11/11: random_search.best_index_
11/12: random_search.cv_results_
11/13: len(random_search.cv_results_)
11/14: len(random_search.score)
11/15: len(random_search.score())
11/16: random_search.cv_results_
11/17: random_search.cv_results_['mean_test_score']
11/18: np.argmax(random_search.cv_results_['mean_test_score'])
11/19: random_search.cv_results_['mean_test_score']
11/20: np.argsort(random_search.cv_results_['mean_test_score'])
11/21: random_search.cv_results_['mean_test_score'][24]
11/22: random_search.cv_results_['mean_test_score'][11]
11/23: random_search.cv_results_['mean_test_score'][8]
11/24: random_search.cv_results_['mean_test_score'][16]
11/25: random_search.cv_results_['mean_test_score'][24]
11/26: random_search.cv_results_['params'][24]
11/27:
my_model = XGBRegressor(n_estimators=1500, max_depth=9, n_jobs=4, gamma=1.5, colsample_bytree=0.6, 
                        min_child_weight=10, subsample=0.8, learning_rate=0.1)
# Add silent=True to avoid printing out updates with each cycle
my_model.fit(X, y, verbose=False)

act_test_data = pd.read_csv("~/data/cons_testing.csv", parse_dates=[0,1])
act_test_data = pre_process(act_test_data)
act_X = act_test_data.drop(targets, axis=1)[features_to_use]
act_y = act_test_data.cons_actual_excl_umm

predictions = my_model.predict(act_X)
print("Mean Absolute Error : " + str(mean_absolute_error(predictions, act_y)))
11/28:
my_model = XGBRegressor(n_estimators=1000, max_depth=9, n_jobs=4, gamma=1.5, colsample_bytree=0.6, 
                        min_child_weight=10, subsample=0.8, learning_rate=0.1)
# Add silent=True to avoid printing out updates with each cycle
my_model.fit(X, y, verbose=False)

act_test_data = pd.read_csv("~/data/cons_testing.csv", parse_dates=[0,1])
act_test_data = pre_process(act_test_data)
act_X = act_test_data.drop(targets, axis=1)[features_to_use]
act_y = act_test_data.cons_actual_excl_umm

predictions = my_model.predict(act_X)
print("Mean Absolute Error : " + str(mean_absolute_error(predictions, act_y)))
11/29:
my_model = XGBRegressor(n_estimators=1500, max_depth=9, n_jobs=4, gamma=3, colsample_bytree=0.6, 
                        min_child_weight=10, subsample=0.8, learning_rate=0.1)
# Add silent=True to avoid printing out updates with each cycle
my_model.fit(X, y, verbose=False)

act_test_data = pd.read_csv("~/data/cons_testing.csv", parse_dates=[0,1])
act_test_data = pre_process(act_test_data)
act_X = act_test_data.drop(targets, axis=1)[features_to_use]
act_y = act_test_data.cons_actual_excl_umm

predictions = my_model.predict(act_X)
print("Mean Absolute Error : " + str(mean_absolute_error(predictions, act_y)))
11/30:
my_model = XGBRegressor(n_estimators=1500, max_depth=9, n_jobs=4, gamma=4, colsample_bytree=0.8, 
                        min_child_weight=10, subsample=0.8, learning_rate=0.1)
# Add silent=True to avoid printing out updates with each cycle
my_model.fit(X, y, verbose=False)

act_test_data = pd.read_csv("~/data/cons_testing.csv", parse_dates=[0,1])
act_test_data = pre_process(act_test_data)
act_X = act_test_data.drop(targets, axis=1)[features_to_use]
act_y = act_test_data.cons_actual_excl_umm

predictions = my_model.predict(act_X)
print("Mean Absolute Error : " + str(mean_absolute_error(predictions, act_y)))
11/31:
my_model = XGBRegressor(n_estimators=800, max_depth=9, n_jobs=4, gamma=4, colsample_bytree=0.8, 
                        min_child_weight=10, subsample=0.8, learning_rate=0.1)
# Add silent=True to avoid printing out updates with each cycle
my_model.fit(X, y, verbose=False)

act_test_data = pd.read_csv("~/data/cons_testing.csv", parse_dates=[0,1])
act_test_data = pre_process(act_test_data)
act_X = act_test_data.drop(targets, axis=1)[features_to_use]
act_y = act_test_data.cons_actual_excl_umm

predictions = my_model.predict(act_X)
print("Mean Absolute Error : " + str(mean_absolute_error(predictions, act_y)))
11/32:
my_model = XGBRegressor(n_estimators=200, max_depth=9, n_jobs=4, gamma=4, colsample_bytree=0.8, 
                        min_child_weight=10, subsample=0.8, learning_rate=0.1)
# Add silent=True to avoid printing out updates with each cycle
my_model.fit(X, y, verbose=False)

act_test_data = pd.read_csv("~/data/cons_testing.csv", parse_dates=[0,1])
act_test_data = pre_process(act_test_data)
act_X = act_test_data.drop(targets, axis=1)[features_to_use]
act_y = act_test_data.cons_actual_excl_umm

predictions = my_model.predict(act_X)
print("Mean Absolute Error : " + str(mean_absolute_error(predictions, act_y)))
11/33:
my_model = XGBRegressor(n_estimators=100, max_depth=9, n_jobs=4, gamma=4, colsample_bytree=0.8, 
                        min_child_weight=10, subsample=0.8, learning_rate=0.1)
# Add silent=True to avoid printing out updates with each cycle
my_model.fit(X, y, verbose=False)

act_test_data = pd.read_csv("~/data/cons_testing.csv", parse_dates=[0,1])
act_test_data = pre_process(act_test_data)
act_X = act_test_data.drop(targets, axis=1)[features_to_use]
act_y = act_test_data.cons_actual_excl_umm

predictions = my_model.predict(act_X)
print("Mean Absolute Error : " + str(mean_absolute_error(predictions, act_y)))
11/34:
my_model = XGBRegressor(n_estimators=10, max_depth=9, n_jobs=4, gamma=4, colsample_bytree=0.8, 
                        min_child_weight=10, subsample=0.8, learning_rate=0.1)
# Add silent=True to avoid printing out updates with each cycle
my_model.fit(X, y, verbose=False)

act_test_data = pd.read_csv("~/data/cons_testing.csv", parse_dates=[0,1])
act_test_data = pre_process(act_test_data)
act_X = act_test_data.drop(targets, axis=1)[features_to_use]
act_y = act_test_data.cons_actual_excl_umm

predictions = my_model.predict(act_X)
print("Mean Absolute Error : " + str(mean_absolute_error(predictions, act_y)))
11/35:
my_model = XGBRegressor(n_estimators=1000, max_depth=9, n_jobs=4, gamma=3, colsample_bytree=0.6, 
                        min_child_weight=10, subsample=0.8, learning_rate=0.1)
# Add silent=True to avoid printing out updates with each cycle
my_model.fit(X, y, verbose=False)

act_test_data = pd.read_csv("~/data/cons_testing.csv", parse_dates=[0,1])
act_test_data = pre_process(act_test_data)
act_X = act_test_data.drop(targets, axis=1)[features_to_use]
act_y = act_test_data.cons_actual_excl_umm

predictions = my_model.predict(act_X)
print("Mean Absolute Error : " + str(mean_absolute_error(predictions, act_y)))
11/36:
my_model = XGBRegressor(n_estimators=1000, max_depth=9, n_jobs=4, gamma=3, colsample_bytree=0.6, 
                        min_child_weight=10, subsample=0.8, learning_rate=0.1)
# Add silent=True to avoid printing out updates with each cycle
my_model.fit(text_X, test_y, verbose=False)

act_test_data = pd.read_csv("~/data/cons_testing.csv", parse_dates=[0,1])
act_test_data = pre_process(act_test_data)
act_X = act_test_data.drop(targets, axis=1)[features_to_use]
act_y = act_test_data.cons_actual_excl_umm

predictions = my_model.predict(act_X)
print("Mean Absolute Error : " + str(mean_absolute_error(predictions, act_y)))
11/37:
my_model = XGBRegressor(n_estimators=1000, max_depth=9, n_jobs=4, gamma=3, colsample_bytree=0.6, 
                        min_child_weight=10, subsample=0.8, learning_rate=0.1)
# Add silent=True to avoid printing out updates with each cycle
my_model.fit(train_X, train_y, verbose=False)

act_test_data = pd.read_csv("~/data/cons_testing.csv", parse_dates=[0,1])
act_test_data = pre_process(act_test_data)
act_X = act_test_data.drop(targets, axis=1)[features_to_use]
act_y = act_test_data.cons_actual_excl_umm

predictions = my_model.predict(act_X)
print("Mean Absolute Error : " + str(mean_absolute_error(predictions, act_y)))
11/38:
my_model = XGBRegressor(n_estimators=1000, max_depth=9, n_jobs=4, gamma=3, colsample_bytree=0.6, 
                        min_child_weight=10, subsample=0.8, learning_rate=0.1)
# Add silent=True to avoid printing out updates with each cycle
my_model.fit(train_X, train_y, verbose=False)

act_test_data = pd.read_csv("~/data/cons_testing.csv", parse_dates=[0,1])
act_test_data = pre_process(act_test_data)
act_X = act_test_data.drop(targets, axis=1)[features_to_use]
act_y = act_test_data.cons_actual_excl_umm

predictions = my_model.predict(act_X.values)
print("Mean Absolute Error : " + str(mean_absolute_error(predictions, act_y)))
11/39:
my_model = XGBRegressor(n_estimators=1000, max_depth=9, n_jobs=4, gamma=3, colsample_bytree=0.6, 
                        min_child_weight=10, subsample=0.8, learning_rate=0.2)
# Add silent=True to avoid printing out updates with each cycle
my_model.fit(train_X, train_y, verbose=False)

act_test_data = pd.read_csv("~/data/cons_testing.csv", parse_dates=[0,1])
act_test_data = pre_process(act_test_data)
act_X = act_test_data.drop(targets, axis=1)[features_to_use]
act_y = act_test_data.cons_actual_excl_umm

predictions = my_model.predict(act_X.values)
print("Mean Absolute Error : " + str(mean_absolute_error(predictions, act_y)))
11/40:
my_model = XGBRegressor(n_estimators=2000, max_depth=9, n_jobs=4, gamma=3, colsample_bytree=0.6, 
                        min_child_weight=10, subsample=0.8, learning_rate=0.06)
# Add silent=True to avoid printing out updates with each cycle
my_model.fit(X, y, verbose=False)

act_test_data = pd.read_csv("~/data/cons_testing.csv", parse_dates=[0,1])
act_test_data = pre_process(act_test_data)
act_X = act_test_data.drop(targets, axis=1)[features_to_use]
act_y = act_test_data.cons_actual_excl_umm

predictions = my_model.predict(act_X.values)
print("Mean Absolute Error : " + str(mean_absolute_error(predictions, act_y)))
11/41: random_search.cv_results_['params'][8]
11/42: random_search.cv_results_['params'][16]
11/43:
my_model = XGBRegressor(n_estimators=2000, max_depth=5, n_jobs=4, gamma=3, colsample_bytree=0.6, 
                        min_child_weight=10, subsample=0.8, learning_rate=0.06)
# Add silent=True to avoid printing out updates with each cycle
my_model.fit(X, y, verbose=False)

act_test_data = pd.read_csv("~/data/cons_testing.csv", parse_dates=[0,1])
act_test_data = pre_process(act_test_data)
act_X = act_test_data.drop(targets, axis=1)[features_to_use]
act_y = act_test_data.cons_actual_excl_umm

predictions = my_model.predict(act_X)
print("Mean Absolute Error : " + str(mean_absolute_error(predictions, act_y)))
11/44:
my_model = XGBRegressor(n_estimators=2000, max_depth=3, n_jobs=4, gamma=5, colsample_bytree=0.6, 
                        min_child_weight=10, subsample=0.8, learning_rate=0.06)
# Add silent=True to avoid printing out updates with each cycle
my_model.fit(X, y, verbose=False)

act_test_data = pd.read_csv("~/data/cons_testing.csv", parse_dates=[0,1])
act_test_data = pre_process(act_test_data)
act_X = act_test_data.drop(targets, axis=1)[features_to_use]
act_y = act_test_data.cons_actual_excl_umm

predictions = my_model.predict(act_X)
print("Mean Absolute Error : " + str(mean_absolute_error(predictions, act_y)))
11/45:
my_model = XGBRegressor(n_estimators=2000, max_depth=11, n_jobs=4, gamma=5, colsample_bytree=0.6, 
                        min_child_weight=10, subsample=0.8, learning_rate=0.06)
# Add silent=True to avoid printing out updates with each cycle
my_model.fit(X, y, verbose=False)

act_test_data = pd.read_csv("~/data/cons_testing.csv", parse_dates=[0,1])
act_test_data = pre_process(act_test_data)
act_X = act_test_data.drop(targets, axis=1)[features_to_use]
act_y = act_test_data.cons_actual_excl_umm

predictions = my_model.predict(act_X)
print("Mean Absolute Error : " + str(mean_absolute_error(predictions, act_y)))
11/46:
my_model = XGBRegressor(n_estimators=2000, max_depth=11, n_jobs=4, gamma=7, colsample_bytree=0.6, 
                        min_child_weight=10, subsample=0.8, learning_rate=0.06)
# Add silent=True to avoid printing out updates with each cycle
my_model.fit(X, y, verbose=False)

act_test_data = pd.read_csv("~/data/cons_testing.csv", parse_dates=[0,1])
act_test_data = pre_process(act_test_data)
act_X = act_test_data.drop(targets, axis=1)[features_to_use]
act_y = act_test_data.cons_actual_excl_umm

predictions = my_model.predict(act_X)
print("Mean Absolute Error : " + str(mean_absolute_error(predictions, act_y)))
11/47:
my_model = XGBRegressor(n_estimators=3000, max_depth=11, n_jobs=4, gamma=3, colsample_bytree=0.6, 
                        min_child_weight=10, subsample=0.8, learning_rate=0.06)
# Add silent=True to avoid printing out updates with each cycle
my_model.fit(X, y, verbose=False)

act_test_data = pd.read_csv("~/data/cons_testing.csv", parse_dates=[0,1])
act_test_data = pre_process(act_test_data)
act_X = act_test_data.drop(targets, axis=1)[features_to_use]
act_y = act_test_data.cons_actual_excl_umm

predictions = my_model.predict(act_X)
print("Mean Absolute Error : " + str(mean_absolute_error(predictions, act_y)))
11/48:
my_model = XGBRegressor(n_estimators=1200, max_depth=7, n_jobs=4, gamma=0.5, colsample_bytree=1, 
                        min_child_weight=5, subsample=0.6, learning_rate=0.1)
# Add silent=True to avoid printing out updates with each cycle
my_model.fit(X, y, verbose=False)

act_test_data = pd.read_csv("~/data/cons_testing.csv", parse_dates=[0,1])
act_test_data = pre_process(act_test_data)
act_X = act_test_data.drop(targets, axis=1)[features_to_use]
act_y = act_test_data.cons_actual_excl_umm

predictions = my_model.predict(act_X)
print("Mean Absolute Error : " + str(mean_absolute_error(predictions, act_y)))
11/49:
my_model = XGBRegressor(n_estimators=1200, max_depth=3, n_jobs=4, gamma=0.5, colsample_bytree=1, 
                        min_child_weight=5, subsample=0.6, learning_rate=0.1)
# Add silent=True to avoid printing out updates with each cycle
my_model.fit(X, y, verbose=False)

act_test_data = pd.read_csv("~/data/cons_testing.csv", parse_dates=[0,1])
act_test_data = pre_process(act_test_data)
act_X = act_test_data.drop(targets, axis=1)[features_to_use]
act_y = act_test_data.cons_actual_excl_umm

predictions = my_model.predict(act_X)
print("Mean Absolute Error : " + str(mean_absolute_error(predictions, act_y)))
11/50:
my_model = XGBRegressor(n_estimators=1000, max_depth=9, n_jobs=4, gamma=0.5, colsample_bytree=0.6, 
                        min_child_weight=10, subsample=0.8, learning_rate=0.1)
# Add silent=True to avoid printing out updates with each cycle
my_model.fit(X, y, verbose=False)

act_test_data = pd.read_csv("~/data/cons_testing.csv", parse_dates=[0,1])
act_test_data = pre_process(act_test_data)
act_X = act_test_data.drop(targets, axis=1)[features_to_use]
act_y = act_test_data.cons_actual_excl_umm

predictions = my_model.predict(act_X)
print("Mean Absolute Error : " + str(mean_absolute_error(predictions, act_y)))
11/51:
my_model = XGBRegressor(n_estimators=800, max_depth=9, n_jobs=4, gamma=0.5, colsample_bytree=0.6, 
                        min_child_weight=10, subsample=0.8, learning_rate=0.1)
# Add silent=True to avoid printing out updates with each cycle
my_model.fit(X, y, verbose=False)

act_test_data = pd.read_csv("~/data/cons_testing.csv", parse_dates=[0,1])
act_test_data = pre_process(act_test_data)
act_X = act_test_data.drop(targets, axis=1)[features_to_use]
act_y = act_test_data.cons_actual_excl_umm

predictions = my_model.predict(act_X)
print("Mean Absolute Error : " + str(mean_absolute_error(predictions, act_y)))
11/52:
train_X, test_X, train_y, test_y = train_test_split(X.values, y.values, test_size=0.2)

my_model = XGBRegressor(n_estimators=1000, max_depth=9, n_jobs=4, gamma=3, colsample_bytree=0.6, 
                        min_child_weight=10, subsample=0.8)
# Add silent=True to avoid printing out updates with each cycle
my_model.fit(train_X, train_y, early_stopping_rounds=5, 
             eval_set=[(test_X, test_y)], eval_metric='mae', verbose=False)

predictions = my_model.predict(test_X)
print("Mean Absolute Error : " + str(mean_absolute_error(predictions, test_y)))
11/53:
act_test_data = pd.read_csv("~/data/cons_testing.csv", parse_dates=[0,1])
act_test_data = pre_process(act_test_data)
act_X = act_test_data.drop(targets, axis=1)[features_to_use]
act_y = act_test_data.cons_actual_excl_umm

predictions = my_model.predict(act_X)
print("Mean Absolute Error : " + str(mean_absolute_error(predictions, act_y)))
11/54:
act_test_data = pd.read_csv("~/data/cons_testing.csv", parse_dates=[0,1])
act_test_data = pre_process(act_test_data)
act_X = act_test_data.drop(targets, axis=1)[features_to_use]
act_y = act_test_data.cons_actual_excl_umm

predictions = my_model.predict(act_X.values)
print("Mean Absolute Error : " + str(mean_absolute_error(predictions, act_y)))
11/55: my_model
11/56: my_model.evals_result
11/57: my_model.feature_importances_
11/58: my_model.score
11/59: my_model.score()
11/60:
my_model = XGBRegressor(n_estimators=1000, max_depth=9, n_jobs=4, gamma=3, colsample_bytree=0.6, 
                        min_child_weight=10, subsample=0.8, learning_rate=0.1)
# Add silent=True to avoid printing out updates with each cycle
my_model.fit(X, y, verbose=False)
11/61:


act_test_data = pd.read_csv("~/data/cons_testing.csv", parse_dates=[0,1])
act_test_data = pre_process(act_test_data)
act_X = act_test_data.drop(targets, axis=1)[features_to_use]
act_y = act_test_data.cons_actual_excl_umm

predictions = my_model.predict(act_X.values)
print("Mean Absolute Error : " + str(mean_absolute_error(predictions, act_y)))
11/62:


act_test_data = pd.read_csv("~/data/cons_testing.csv", parse_dates=[0,1])
act_test_data = pre_process(act_test_data)
act_X = act_test_data.drop(targets, axis=1)[features_to_use]
act_y = act_test_data.cons_actual_excl_umm

predictions = my_model.predict(act_X)
print("Mean Absolute Error : " + str(mean_absolute_error(predictions, act_y)))
11/63:
train_X, test_X, train_y, test_y = train_test_split(X.values, y.values, test_size=0.2)

my_model = XGBRegressor(n_estimators=1000, max_depth=9, n_jobs=4, gamma=3, colsample_bytree=0.6, 
                        min_child_weight=10, subsample=0.8)
# Add silent=True to avoid printing out updates with each cycle
my_model.fit(train_X, train_y, early_stopping_rounds=5, 
             eval_set=[(test_X, test_y)], eval_metric='mae', verbose=False)

predictions = my_model.predict(test_X)
print("Mean Absolute Error : " + str(mean_absolute_error(predictions, test_y)))
11/64:


act_test_data = pd.read_csv("~/data/cons_testing.csv", parse_dates=[0,1])
act_test_data = pre_process(act_test_data)
act_X = act_test_data.drop(targets, axis=1)[features_to_use]
act_y = act_test_data.cons_actual_excl_umm

predictions = my_model.predict(act_X)
print("Mean Absolute Error : " + str(mean_absolute_error(predictions, act_y)))
11/65:


act_test_data = pd.read_csv("~/data/cons_testing.csv", parse_dates=[0,1])
act_test_data = pre_process(act_test_data)
act_X = act_test_data.drop(targets, axis=1)[features_to_use]
act_y = act_test_data.cons_actual_excl_umm

predictions = my_model.predict(act_X.values)
print("Mean Absolute Error : " + str(mean_absolute_error(predictions, act_y)))
12/1:
import pandas as pd
import numpy as np
from datetime import datetime

import xgboost as xgb
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
12/2:
data = pd.read_csv("data/cons_training.csv", parse_dates=[0,1])
data.sort_values("start_time_utc", inplace=True)
cons_test_data = pd.read_csv("data/cons_testing_without_labels.csv", parse_dates=[0,1])
data.dtypes
12/3:
import pandas as pd
import numpy as np
from datetime import datetime

import xgboost as xgb
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
12/4:
data = pd.read_csv("data/cons_training.csv", parse_dates=[0,1])
data.dtypes
12/5:
def pre_process(df, y_used = 'cons_actual_excl_umm'):
    targets = ['cons_actual_plus_umm', 'cons_actual_excl_umm']
    features_not_used = ['start_time_utc', 'start_time_local', 'cons_fcast_fingrid_excl_umm']
    features_to_use = [c for c in df.columns if c not in targets + features_not_used]
    df = df.dropna() #not many NA, just drop them
    df['day_of_week'] = df.apply(lambda row: row.start_time_local.dayofweek, axis=1)
    X = df.drop(targets, axis=1)[features_to_use]
    return df, X

data, X = pre_process(data)
y1 = data['cons_actual_plus_umm']
y2 = ['cons_actual_excl_umm']
data.describe()
12/6:
def pre_process(df, y_used = 'cons_actual_excl_umm'):
    targets = ['cons_actual_plus_umm', 'cons_actual_excl_umm']
    features_not_used = ['start_time_utc', 'start_time_local', 'cons_fcast_fingrid_excl_umm']
    features_to_use = [c for c in df.columns if c not in targets + features_not_used]
    df = df.dropna() #not many NA, just drop them
    #df['day_of_week'] = df.apply(lambda row: row.start_time_local.dayofweek, axis=1)
    X = df.drop(targets, axis=1)[features_to_use]
    return df, X

data, X = pre_process(data)
y1 = data['cons_actual_plus_umm']
y2 = ['cons_actual_excl_umm']
data.describe()
12/7:
def pre_process(df, y_used = 'cons_actual_excl_umm'):
    targets = ['cons_actual_plus_umm', 'cons_actual_excl_umm']
    features_not_used = ['start_time_utc', 'start_time_local', 'cons_fcast_fingrid_excl_umm']
    features_to_use = [c for c in df.columns if c not in targets + features_not_used]
    df = df.dropna() #not many NA, just drop them
    df['day_of_week'] = df.apply(lambda row: row.start_time_local.dayofweek, axis=1)
    X = df.drop(targets, axis=1)[features_to_use]
    return df, X

data, X = pre_process(data)
y1 = data['cons_actual_plus_umm']
y2 = ['cons_actual_excl_umm']
data.describe()
12/8:
#Use a small test set so early stopping works
train_X, test_X, train_y, test_y = train_test_split(X.values, y.values, test_size=0.15)

#After some manual CV these seem to perform well. Possible add more gamma for regularization?
my_model = XGBRegressor(n_estimators=1000, max_depth=9, n_jobs=4, gamma=3, colsample_bytree=0.6, 
                        min_child_weight=10, subsample=0.8)

my_model.fit(train_X, train_y, early_stopping_rounds=5, 
             eval_set=[(test_X, test_y)], eval_metric='mae', verbose=False)

predictions = my_model.predict(test_X)
print("Mean Absolute Error : " + str(mean_absolute_error(predictions, test_y)))
12/9:
#Use a small test set so early stopping works
train_X, test_X, train_y, test_y = train_test_split(X, y1, test_size=0.15)

#After some manual CV these seem to perform well. Possible add more gamma for regularization?
my_model = XGBRegressor(n_estimators=1000, max_depth=9, n_jobs=4, gamma=3, colsample_bytree=0.6, 
                        min_child_weight=10, subsample=0.8)

my_model.fit(train_X, train_y, early_stopping_rounds=5, 
             eval_set=[(test_X, test_y)], eval_metric='mae', verbose=False)

predictions = my_model.predict(test_X)
print("Mean Absolute Error : " + str(mean_absolute_error(predictions, test_y)))
12/10:
def pre_process(df):
    targets = ['cons_actual_plus_umm', 'cons_actual_excl_umm']
    features_not_used = ['start_time_utc', 'start_time_local', 'cons_fcast_fingrid_excl_umm']
    features_to_use = [c for c in df.columns if c not in targets + features_not_used]
    df = df.dropna() #not many NA, just drop them
    df['day_of_week'] = df.apply(lambda row: row.start_time_local.dayofweek, axis=1)
    X = df.drop(targets, axis=1)[features_to_use]
    return df, X

data, X = pre_process(data)
y1 = data['cons_actual_plus_umm']
y2 = ['cons_actual_excl_umm']
data.describe()
12/11:
test_set = pd.read_csv("data/cons_testing_without_labels.csv", parse_dates=[0,1])
test_set, test_set_X = pre_process(test_set)
12/12:
test_set = pd.read_csv("data/cons_testing_without_labels.csv", parse_dates=[0,1])
#test_set, test_set_X = pre_process(test_set)
12/13: test_set
12/14:
def pre_process(df, drop_na=False):
    targets = ['cons_actual_plus_umm', 'cons_actual_excl_umm']
    features_not_used = ['start_time_utc', 'start_time_local', 'cons_fcast_fingrid_excl_umm']
    features_to_use = [c for c in df.columns if c not in targets + features_not_used]
    if drop_na:
        df = df.dropna() #not many NA, just drop them
    df['day_of_week'] = df.apply(lambda row: row.start_time_local.dayofweek, axis=1)
    X = df.drop(targets, axis=1)[features_to_use]
    return df, X

data, X = pre_process(data, drop_na=True)
y1 = data['cons_actual_plus_umm']
y2 = ['cons_actual_excl_umm']
data.describe()
12/15:
test_set = pd.read_csv("data/cons_testing_without_labels.csv", parse_dates=[0,1])
test_set, test_set_X = pre_process(test_set)
12/16: test_set
12/17: test_set_X
12/18: test_set
12/19: test_set_X
12/20: predictions = my_model.predict(test_set_X)
12/21: test_set_X
12/22:
test_set = pd.read_csv("data/cons_testing_without_labels.csv", parse_dates=[0,1])
test_set, test_set_X = pre_process(test_set)
12/23: test_set_X
12/24: test_set
12/25:
def pre_process(df, drop_na=False):
    targets = ['cons_actual_plus_umm', 'cons_actual_excl_umm']
    features_not_used = ['start_time_utc', 'start_time_local', 'cons_fcast_fingrid_excl_umm']
    features_to_use = [c for c in df.columns if c not in targets + features_not_used]
    if drop_na:
        df = df.dropna() #not many NA, just drop them
    #Add day of week as feature
    df['day_of_week'] = df.apply(lambda row: row.start_time_local.dayofweek, axis=1)
    pre_X = df.drop(targets, axis=1)[features_to_use]
    return df, pre_X

data, X = pre_process(data, drop_na=True)
y1 = data['cons_actual_plus_umm']
y2 = ['cons_actual_excl_umm']
data.describe()
12/26:
test_set = pd.read_csv("data/cons_testing_without_labels.csv", parse_dates=[0,1])
test_set, test_set_X = pre_process(test_set)
12/27: test_set
12/28: test_set_X
12/29:
def pre_process(df, drop_na=False):
    targets = ['cons_actual_plus_umm', 'cons_actual_excl_umm']
    features_not_used = ['start_time_utc', 'start_time_local', 'cons_fcast_fingrid_excl_umm']
    features_to_use = [c for c in df.columns if c not in targets + features_not_used]
    if drop_na:
        df = df.dropna() #not many NA, just drop them
    #Add day of week as feature
    df['day_of_week'] = df.apply(lambda row: row.start_time_local.dayofweek, axis=1)
    print(df.columns)
    pre_X = df.drop(targets, axis=1)[features_to_use]
    return df, pre_X

data, X = pre_process(data, drop_na=True)
y1 = data['cons_actual_plus_umm']
y2 = ['cons_actual_excl_umm']
data.describe()
12/30:
test_set = pd.read_csv("data/cons_testing_without_labels.csv", parse_dates=[0,1])
test_set, test_set_X = pre_process(test_set)
12/31: X
12/32: test_set.null().sum()
12/33: test_set.null().issum()
12/34: test_set.isnull().sum()
11/66:
data = pd.read_csv("data/cons_training.csv", parse_dates=[0,1])
data.sort_values("start_time_utc", inplace=True)
cons_test_data = pd.read_csv("data/cons_testing_without_labels.csv", parse_dates=[0,1])
data.dtypes
data.isnull.sum()
11/67:
data = pd.read_csv("data/cons_training.csv", parse_dates=[0,1])
data.sort_values("start_time_utc", inplace=True)
cons_test_data = pd.read_csv("data/cons_testing_without_labels.csv", parse_dates=[0,1])
data.dtypes
data.isnull().sum()
12/35:
def pre_process(df):
    targets = ['cons_actual_plus_umm', 'cons_actual_excl_umm']
    features_not_used = ['start_time_utc', 'start_time_local', 'cons_fcast_fingrid_excl_umm']
    features_to_use = [c for c in df.columns if c not in targets + features_not_used]
    # Only some temperatures and cons_fcast (which can't be used) are null. Fill them with prev vals
    df = df.fillna(method='ffill')
    #Add day of week as feature
    df['day_of_week'] = df.apply(lambda row: row.start_time_local.dayofweek, axis=1)
    pre_X = df.drop(targets, axis=1)[features_to_use]
    return df, pre_X

data, X = pre_process(data)
y1 = data['cons_actual_plus_umm']
y2 = ['cons_actual_excl_umm']
data.describe()
12/36:
def pre_process(df):
    targets = ['cons_actual_plus_umm', 'cons_actual_excl_umm']
    features_not_used = ['start_time_utc', 'start_time_local', 'cons_fcast_fingrid_excl_umm']
    features_to_use = [c for c in df.columns if c not in targets + features_not_used]
    # Only some temperatures and cons_fcast (which can't be used) are null. Fill them with prev vals
    df = df.fillna(method='ffill')
    #Add day of week as feature
    df['day_of_week'] = df.apply(lambda row: row.start_time_local.dayofweek, axis=1)
    pre_X = df.drop(targets, axis=1)[features_to_use]
    return df, pre_X

data, X = pre_process(data)
y1 = data['cons_actual_plus_umm']
y2 = ['cons_actual_excl_umm']
data.describe()
data.isnull().sum()
13/1:
import pandas as pd
import numpy as np
from datetime import datetime

import xgboost as xgb
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
13/2:
data = pd.read_csv("data/cons_training.csv", parse_dates=[0,1])
data.dtypes
13/3:
def pre_process(df):
    targets = ['cons_actual_plus_umm', 'cons_actual_excl_umm']
    features_not_used = ['start_time_utc', 'start_time_local', 'cons_fcast_fingrid_excl_umm']
    features_to_use = [c for c in df.columns if c not in targets + features_not_used]
    # Only some temperatures and cons_fcast (which can't be used) are null. Fill them with prev vals
    df = df.fillna(method='ffill')
    #Add day of week as feature
    df['day_of_week'] = df.apply(lambda row: row.start_time_local.dayofweek, axis=1)
    pre_X = df.drop(targets, axis=1)[features_to_use]
    return df, pre_X

data, X = pre_process(data)
y1 = data['cons_actual_plus_umm']
y2 = data['cons_actual_excl_umm']
13/4:
test_set = pd.read_csv("data/cons_testing_without_labels.csv", parse_dates=[0,1])
test_set, test_set_X = pre_process(test_set)
13/5: test_set
13/6: test_set_X
13/7: X
13/8:
def pre_process(df):
    targets = ['cons_actual_plus_umm', 'cons_actual_excl_umm']
    features_not_used = ['start_time_utc', 'start_time_local', 'cons_fcast_fingrid_excl_umm']
    features_to_use = [c for c in df.columns if c not in targets + features_not_used] + ['day_of_week']
    # Only some temperatures and cons_fcast (which can't be used) are null. Fill them with prev vals
    df = df.fillna(method='ffill')
    #Add day of week as feature
    df['day_of_week'] = df.apply(lambda row: row.start_time_local.dayofweek, axis=1)
    pre_X = df.drop(targets, axis=1)[features_to_use]
    return df, pre_X

data, X = pre_process(data)
y1 = data['cons_actual_plus_umm']
y2 = data['cons_actual_excl_umm']
13/9:
#Use a small test set so early stopping works
train_X, test_X, train_y, test_y = train_test_split(X, y1, test_size=0.15)

#After some manual CV these seem to perform well. Possible add more gamma for regularization?
my_model = XGBRegressor(n_estimators=1000, max_depth=9, n_jobs=4, gamma=3, colsample_bytree=0.6, 
                        min_child_weight=10, subsample=0.8)

my_model.fit(train_X, train_y, early_stopping_rounds=5, 
             eval_set=[(test_X, test_y)], eval_metric='mae', verbose=False)

predictions = my_model.predict(test_X)
print("Mean Absolute Error : " + str(mean_absolute_error(predictions, test_y)))
13/10: X
13/11:
def pre_process(df):
    targets = ['cons_actual_plus_umm', 'cons_actual_excl_umm']
    features_not_used = ['start_time_utc', 'start_time_local', 'cons_fcast_fingrid_excl_umm']
    features_to_use = [c for c in df.columns if c not in targets + features_not_used]
    # Only some temperatures and cons_fcast (which can't be used) are null. Fill them with prev vals
    df.fillna(method='ffill', inplace=True)
    #Add day of week as feature
    df['day_of_week'] = df.apply(lambda row: row.start_time_local.dayofweek, axis=1)
    df.drop(targets, axis=1, inplace=True)
    pre_X = df[features_to_use]
    return df, pre_X

data, X = pre_process(data)
y1 = data['cons_actual_plus_umm']
y2 = data['cons_actual_excl_umm']
14/1:
import pandas as pd
import numpy as np
from datetime import datetime

import xgboost as xgb
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
14/2:
data = pd.read_csv("data/cons_training.csv", parse_dates=[0,1])
data.dtypes
14/3:
def pre_process(df):
    targets = ['cons_actual_plus_umm', 'cons_actual_excl_umm']
    features_not_used = ['start_time_utc', 'start_time_local', 'cons_fcast_fingrid_excl_umm']
    features_to_use = [c for c in df.columns if c not in targets + features_not_used]
    # Only some temperatures and cons_fcast (which can't be used) are null. Fill them with prev vals
    df = df.fillna(method='ffill')
    #Add day of week as feature
    df['day_of_week'] = df.apply(lambda row: row.start_time_local.dayofweek, axis=1)
    pre_X = df.drop(targets, axis=1)[features_to_use]
    return df, pre_X

data, X = pre_process(data)
y1 = data['cons_actual_plus_umm']
y2 = data['cons_actual_excl_umm']
14/4:
def pre_process(df):
    targets = ['cons_actual_plus_umm', 'cons_actual_excl_umm']
    features_not_used = ['start_time_utc', 'start_time_local', 'cons_fcast_fingrid_excl_umm']
    features_to_use = [c for c in df.columns if c not in targets + features_not_used]
    # Only some temperatures and cons_fcast (which can't be used) are null. Fill them with prev vals
    df = df.fillna(method='ffill')
    #Add day of week as feature
    df['day_of_week'] = df.apply(lambda row: row.start_time_local.dayofweek, axis=1)
    pre_X = df.drop(targets, axis=1)[features_to_use]
    print(pre_X.columns)
    return df, pre_X

data, X = pre_process(data)
y1 = data['cons_actual_plus_umm']
y2 = data['cons_actual_excl_umm']
14/5:
test_set = pd.read_csv("data/cons_testing_without_labels.csv", parse_dates=[0,1])
test_set, test_set_X = pre_process(test_set)
14/6: test_set
14/7: data
14/8: df
14/9:
def pre_process(df):
    targets = ['cons_actual_plus_umm', 'cons_actual_excl_umm']
    features_not_used = ['start_time_utc', 'start_time_local', 'cons_fcast_fingrid_excl_umm']
    features_to_use = [c for c in df.columns if c not in targets + features_not_used + ['day_of_week']]
    # Only some temperatures and cons_fcast (which can't be used) are null. Fill them with prev vals
    df = df.fillna(method='ffill')
    #Add day of week as feature
    df['day_of_week'] = df.apply(lambda row: row.start_time_local.dayofweek, axis=1)
    pre_X = df.drop(targets, axis=1)[features_to_use]
    return df, pre_X

data, X = pre_process(data)
y1 = data['cons_actual_plus_umm']
y2 = data['cons_actual_excl_umm']
14/10: X.columns
14/11:
def pre_process(df):
    targets = ['cons_actual_plus_umm', 'cons_actual_excl_umm']
    features_not_used = ['start_time_utc', 'start_time_local', 'cons_fcast_fingrid_excl_umm']
    features_to_use = [c for c in df.columns if c not in targets + features_not_used] + ['day_of_week']
    # Only some temperatures and cons_fcast (which can't be used) are null. Fill them with prev vals
    df = df.fillna(method='ffill')
    #Add day of week as feature
    df['day_of_week'] = df.apply(lambda row: row.start_time_local.dayofweek, axis=1)
    print(df.columns)
    pre_X = df.drop(targets, axis=1)[features_to_use]
    return df, pre_X

data, X = pre_process(data)
y1 = data['cons_actual_plus_umm']
y2 = data['cons_actual_excl_umm']
14/12: X.columns
14/13:
def pre_process(df):
    targets = ['cons_actual_plus_umm', 'cons_actual_excl_umm']
    features_not_used = ['start_time_utc', 'start_time_local', 'cons_fcast_fingrid_excl_umm']
    features_to_use = [c for c in df.columns if c not in targets + features_not_used] + ['day_of_week']
    # Only some temperatures and cons_fcast (which can't be used) are null. Fill them with prev vals
    df = df.fillna(method='ffill')
    #Add day of week as feature
    df['day_of_week'] = df.apply(lambda row: row.start_time_local.dayofweek, axis=1)
    print(df.columns)
    pre_X = df.drop(targets, axis=1)[features_to_use]
    print(features_to_use)
    return df, pre_X

data, X = pre_process(data)
y1 = data['cons_actual_plus_umm']
y2 = data['cons_actual_excl_umm']
14/14:
def pre_process(df):
    targets = ['cons_actual_plus_umm', 'cons_actual_excl_umm']
    features_not_used = ['start_time_utc', 'start_time_local', 'cons_fcast_fingrid_excl_umm']
    print(df.columns)
    features_to_use = [c for c in df.columns if c not in targets + features_not_used] + ['day_of_week']
    # Only some temperatures and cons_fcast (which can't be used) are null. Fill them with prev vals
    df = df.fillna(method='ffill')
    #Add day of week as feature
    df['day_of_week'] = df.apply(lambda row: row.start_time_local.dayofweek, axis=1)
    print(df.columns)
    pre_X = df.drop(targets, axis=1)[features_to_use]
    print(features_to_use)
    return df, pre_X

data, X = pre_process(data)
y1 = data['cons_actual_plus_umm']
y2 = data['cons_actual_excl_umm']
14/15:
import pandas as pd
import numpy as np
from datetime import datetime

import xgboost as xgb
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
14/16:
train_data = pd.read_csv("data/cons_training.csv", parse_dates=[0,1])
data.dtypes
14/17:
def pre_process(df):
    targets = ['cons_actual_plus_umm', 'cons_actual_excl_umm']
    features_not_used = ['start_time_utc', 'start_time_local', 'cons_fcast_fingrid_excl_umm']
    print(df.columns)
    features_to_use = [c for c in df.columns if c not in targets + features_not_used] + ['day_of_week']
    # Only some temperatures and cons_fcast (which can't be used) are null. Fill them with prev vals
    df = df.fillna(method='ffill')
    #Add day of week as feature
    df['day_of_week'] = df.apply(lambda row: row.start_time_local.dayofweek, axis=1)
    print(df.columns)
    pre_X = df.drop(targets, axis=1)[features_to_use]
    print(features_to_use)
    return df, pre_X

data, X = pre_process(train_data)
y1 = data['cons_actual_plus_umm']
y2 = data['cons_actual_excl_umm']
15/1:
import pandas as pd
import numpy as np
from datetime import datetime

import xgboost as xgb
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
15/2:
train_data = pd.read_csv("data/cons_training.csv", parse_dates=[0,1])
train_data.dtypes
15/3:
def pre_process(df):
    targets = ['cons_actual_plus_umm', 'cons_actual_excl_umm']
    features_not_used = ['start_time_utc', 'start_time_local', 'cons_fcast_fingrid_excl_umm']
    print(df.columns)
    features_to_use = [c for c in df.columns if c not in targets + features_not_used] + ['day_of_week']
    # Only some temperatures and cons_fcast (which can't be used) are null. Fill them with prev vals
    df = df.fillna(method='ffill')
    #Add day of week as feature
    df['day_of_week'] = df.apply(lambda row: row.start_time_local.dayofweek, axis=1)
    print(df.columns)
    pre_X = df.drop(targets, axis=1)[features_to_use]
    print(features_to_use)
    return df, pre_X

data, X = pre_process(train_data)
y1 = data['cons_actual_plus_umm']
y2 = data['cons_actual_excl_umm']
15/4:
test_set_csv = pd.read_csv("data/cons_testing_without_labels.csv", parse_dates=[0,1])
test_set, test_set_X = pre_process(test_set_csv)
15/5:
#Use a small test set so early stopping works
train_X, test_X, train_y, test_y = train_test_split(X, y1, test_size=0.15)

#After some manual CV these seem to perform well. Possible add more gamma for regularization?
my_model = XGBRegressor(n_estimators=1000, max_depth=9, n_jobs=4, gamma=3, colsample_bytree=0.6, 
                        min_child_weight=10, subsample=0.8)

my_model.fit(train_X, train_y, early_stopping_rounds=5, 
             eval_set=[(test_X, test_y)], eval_metric='mae', verbose=False)

predictions = my_model.predict(test_X)
print("Mean Absolute Error : " + str(mean_absolute_error(predictions, test_y)))
15/6: predictions = my_model.predict(test_set_X)
15/7: predictions
16/1:
import pandas as pd
import numpy as np
from datetime import datetime

import xgboost as xgb
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
16/2:
train_data = pd.read_csv("data/cons_training.csv", parse_dates=[0,1])
train_data.dtypes
16/3:
def pre_process(df):
    targets = ['cons_actual_plus_umm', 'cons_actual_excl_umm']
    features_not_used = ['start_time_utc', 'start_time_local', 'cons_fcast_fingrid_excl_umm']
    features_to_use = [c for c in df.columns if c not in targets + features_not_used] + ['day_of_week']
    # Only some temperatures and cons_fcast (which can't be used) are null. Fill them with prev vals
    df = df.fillna(method='ffill')
    #Add day of week as feature
    df['day_of_week'] = df.apply(lambda row: row.start_time_local.dayofweek, axis=1)
    pre_X = df.drop(targets, axis=1)[features_to_use]
    return df, pre_X

data, X = pre_process(train_data)
cons_actual_plus_umm = data['cons_actual_plus_umm']
cons_actual_excl_umm = data['cons_actual_excl_umm']
16/4:
#Use a small test set so early stopping works
train_X, test_X, train_y, test_y = train_test_split(X, cons_actual_excl_umm, test_size=0.15)

#After some manual CV these seem to perform well. Possible add more gamma for regularization?
model_cons_excl_umm = XGBRegressor(n_estimators=1000, max_depth=9, n_jobs=4, gamma=3, colsample_bytree=0.6, 
                        min_child_weight=10, subsample=0.8)

model_cons_excl_umm.fit(train_X, train_y, early_stopping_rounds=5, 
             eval_set=[(test_X, test_y)], eval_metric='mae', verbose=False)

predictions = model_cons_excl_umm.predict(test_X)
print("Mean Absolute Error on held-out test set: " + str(mean_absolute_error(predictions, test_y)))
16/5:
train_X, test_X, train_y, test_y = train_test_split(X, cons_actual_plus_umm, test_size=0.15)

model_plus_umm = XGBRegressor(n_estimators=1000, max_depth=9, n_jobs=4, gamma=3, colsample_bytree=0.6, 
                        min_child_weight=10, subsample=0.8)

model_plus_umm.fit(train_X, train_y, early_stopping_rounds=5, 
             eval_set=[(test_X, test_y)], eval_metric='mae', verbose=False)

predictions = model_plus_umm.predict(test_X)
print("Mean Absolute Error on held-out test set: " + str(mean_absolute_error(predictions, test_y)))
16/6:
#Use a small test set so early stopping works
train_X, test_X, train_y, test_y = train_test_split(X, cons_actual_excl_umm, test_size=0.15)

#After some manual CV these seem to perform well. Possible add more gamma for regularization?
model_cons_excl_umm = XGBRegressor(n_estimators=1200, max_depth=9, n_jobs=4, gamma=3, colsample_bytree=0.6, 
                        min_child_weight=10, subsample=0.8)

model_cons_excl_umm.fit(train_X, train_y, early_stopping_rounds=5, 
             eval_set=[(test_X, test_y)], eval_metric='mae', verbose=False)

predictions = model_cons_excl_umm.predict(test_X)
print("Mean Absolute Error on held-out test set: " + str(mean_absolute_error(predictions, test_y)))
16/7:
#Use a small test set so early stopping works
train_X, test_X, train_y, test_y = train_test_split(X, cons_actual_excl_umm, test_size=0.15)

#After some manual CV these seem to perform well. Possible add more gamma for regularization?
model_cons_excl_umm = XGBRegressor(n_estimators=1200, max_depth=9, n_jobs=4, gamma=3, colsample_bytree=0.6, 
                        min_child_weight=10, subsample=0.8)

model_cons_excl_umm.fit(train_X, train_y, early_stopping_rounds=5, 
             eval_set=[(test_X, test_y)], eval_metric='mae', verbose=False)

predictions = model_cons_excl_umm.predict(test_X)
print("Mean Absolute Error on held-out test set: " + str(mean_absolute_error(predictions, test_y)))
16/8:
#Use a small test set so early stopping works
train_X, test_X, train_y, test_y = train_test_split(X, cons_actual_excl_umm, test_size=0.15)

#After some manual CV these seem to perform well. Possible add more gamma for regularization?
model_cons_excl_umm = XGBRegressor(n_estimators=1200, max_depth=9, n_jobs=4, gamma=3, colsample_bytree=0.6, 
                        min_child_weight=10, subsample=0.8)

model_cons_excl_umm.fit(train_X, train_y, early_stopping_rounds=5, 
             eval_set=[(test_X, test_y)], eval_metric='mae', verbose=False)

predictions = model_cons_excl_umm.predict(test_X)
print("Mean Absolute Error on held-out test set: " + str(mean_absolute_error(predictions, test_y)))
16/9:
train_X, test_X, train_y, test_y = train_test_split(X, cons_actual_plus_umm, test_size=0.15)

model_plus_umm = XGBRegressor(n_estimators=1000, max_depth=9, n_jobs=4, gamma=3, colsample_bytree=0.6, 
                        min_child_weight=10, subsample=0.8)

model_plus_umm.fit(train_X, train_y, early_stopping_rounds=5, 
             eval_set=[(test_X, test_y)], eval_metric='mae', verbose=False)

predictions = model_plus_umm.predict(test_X)
print("Mean Absolute Error on held-out test set: " + str(mean_absolute_error(predictions, test_y)))
16/10:
test_set_csv = pd.read_csv("data/cons_testing_without_labels.csv", parse_dates=[0,1])
test_set, test_set_X = pre_process(test_set_csv)
16/11:
predictions_excl_umm = model_cons_excl_umm.predict(test_set_X)
predictions_plus_umm = model_cons_plus_umm.predict(test_set_X)
16/12:
train_X, test_X, train_y, test_y = train_test_split(X, cons_actual_plus_umm, test_size=0.15)

model_cons_plus_umm = XGBRegressor(n_estimators=1000, max_depth=9, n_jobs=4, gamma=3, colsample_bytree=0.6, 
                        min_child_weight=10, subsample=0.8)

model_cons_plus_umm.fit(train_X, train_y, early_stopping_rounds=5, 
             eval_set=[(test_X, test_y)], eval_metric='mae', verbose=False)

predictions = model_cons_plus_umm.predict(test_X)
print("Mean Absolute Error on held-out test set: " + str(mean_absolute_error(predictions, test_y)))
16/13:
test_set_csv = pd.read_csv("data/cons_testing_without_labels.csv", parse_dates=[0,1])
test_set, test_set_X = pre_process(test_set_csv)
16/14:
predictions_excl_umm = model_cons_excl_umm.predict(test_set_X)
predictions_plus_umm = model_cons_plus_umm.predict(test_set_X)
16/15: predictions_excl_umm
16/16: test_set
16/17: test_set_X
16/18: preds = pd.DataFrame()
16/19:
preds = pd.DataFrame()
preds['start_time_utc'] = test_set['start_time_utc']
preds['predicted_cons_actual_excl_umm'] = predictions_excl_umm
preds['predicted_cons_actual_plus_umm'] = predictions_plus_umm
16/20: preds
16/21: preds.to_csv("data/predictions.csv")
16/22: preds.to_csv("data/predictions.csv", index=False)
11/68: predictions = pd.read_csv("/home/tman/challenge/data/predictions.csv")
11/69: predictions
11/70: print("Mean Absolute Error : " + str(mean_absolute_error(predictions['predicted_cons_actual_excl_umm'], act_y)))
11/71: print("Mean Absolute Error : " + str(mean_absolute_error(predictions['predicted_cons_actual_plus_umm'], act_y)))
11/72: print("Mean Absolute Error : " + str(mean_absolute_error(predictions['predicted_cons_actual_excl_umm'], act_y)))
11/73: print("Mean Absolute Error : " + str(mean_absolute_error(predictions['predicted_cons_actual_plus_umm'], act_y)))
11/74: print("Mean Absolute Error : " + str(mean_absolute_error(predictions['predicted_cons_actual_excl_umm'], act_y)))
11/75: print("Mean Absolute Error : " + str(mean_absolute_error(predictions['predicted_cons_actual_plus_umm'], act_test_data.cons_actual_plus_umm)))
11/76: print("Mean Absolute Error : " + str(mean_absolute_error(predictions['predicted_cons_actual_excl_umm'], act_test_data.cons_actual_excl_umm)))
11/77: print("Mean Absolute Error : " + str(mean_absolute_error(predictions['predicted_cons_actual_excl_umm'], act_test_data.cons_actual_excl_umm)))
11/78: print("Mean Absolute Error : " + str(mean_absolute_error(predictions['predicted_cons_actual_plus_umm'], act_test_data.cons_actual_plus_umm)))
17/1:
import pandas as pd
import numpy as np
from datetime import datetime

import xgboost as xgb
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
17/2:
data = pd.read_csv("data/cons_training.csv", parse_dates=[0,1])
data.sort_values("start_time_utc", inplace=True)
cons_test_data = pd.read_csv("data/cons_testing_without_labels.csv", parse_dates=[0,1])
data.isnull().sum()
17/3:


act_test_data = pd.read_csv("~/data/cons_testing.csv", parse_dates=[0,1])
act_test_data = pre_process(act_test_data)
act_X = act_test_data.drop(targets, axis=1)[features_to_use]
act_y = act_test_data.cons_actual_excl_umm

#predictions = my_model.predict(act_X.values)
#print("Mean Absolute Error : " + str(mean_absolute_error(predictions, act_y)))
17/4:
data = pd.read_csv("data/cons_training.csv", parse_dates=[0,1])
data.sort_values("start_time_utc", inplace=True)
cons_test_data = pd.read_csv("data/cons_testing_without_labels.csv", parse_dates=[0,1])
data.isnull().sum()
17/5:


act_test_data = pd.read_csv("~/data/cons_testing.csv", parse_dates=[0,1])
act_test_data = pre_process(act_test_data)
act_X = act_test_data.drop(targets, axis=1)[features_to_use]
act_y = act_test_data.cons_actual_excl_umm

#predictions = my_model.predict(act_X.values)
#print("Mean Absolute Error : " + str(mean_absolute_error(predictions, act_y)))
17/6:
# Only some temperatures and cons_fcast (which can't be used) are null. Fill them with prev vals
def pre_process(df):
    df['day_of_week'] = df.apply(lambda row: row.start_time_local.dayofweek, axis=1)
    return df

#Baseline: predict the current value from the last day's value: y[i] = y[i-24]?
data = pre_process(data)
data.describe()
17/7:


act_test_data = pd.read_csv("~/data/cons_testing.csv", parse_dates=[0,1])
act_test_data = pre_process(act_test_data)
act_X = act_test_data.drop(targets, axis=1)[features_to_use]
act_y = act_test_data.cons_actual_excl_umm

#predictions = my_model.predict(act_X.values)
#print("Mean Absolute Error : " + str(mean_absolute_error(predictions, act_y)))
18/1:
import pandas as pd
import numpy as np
from datetime import datetime

import xgboost as xgb
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
18/2:
data = pd.read_csv("data/cons_training.csv", parse_dates=[0,1])
data.sort_values("start_time_utc", inplace=True)
cons_test_data = pd.read_csv("data/cons_testing_without_labels.csv", parse_dates=[0,1])
data.isnull().sum()
18/3:
# Only some temperatures and cons_fcast (which can't be used) are null. Fill them with prev vals
def pre_process(df):
    df['day_of_week'] = df.apply(lambda row: row.start_time_local.dayofweek, axis=1)
    return df

#Baseline: predict the current value from the last day's value: y[i] = y[i-24]?
data = pre_process(data)
data.describe()
18/4:
targets = ['cons_actual_plus_umm', 'cons_actual_excl_umm']
features_not_used = ['start_time_utc', 'start_time_local', 'cons_fcast_fingrid_excl_umm']
features_to_use = [c for c in data.columns if c not in targets + features_not_used]
18/5: print("Mean Absolute Error : " + str(mean_absolute_error(predictions['predicted_cons_actual_plus_umm'], act_test_data.cons_actual_plus_umm)))
18/6: predictions = pd.read_csv("/home/tman/Downloads/challenge/data/predictions.csv")
18/7: print("Mean Absolute Error : " + str(mean_absolute_error(predictions['predicted_cons_actual_plus_umm'], act_test_data.cons_actual_plus_umm)))
18/8:


act_test_data = pd.read_csv("~/data/cons_testing.csv", parse_dates=[0,1])
act_test_data = pre_process(act_test_data)
act_X = act_test_data.drop(targets, axis=1)[features_to_use]
act_y = act_test_data.cons_actual_excl_umm

#predictions = my_model.predict(act_X.values)
#print("Mean Absolute Error : " + str(mean_absolute_error(predictions, act_y)))
18/9: print("Mean Absolute Error : " + str(mean_absolute_error(predictions['predicted_cons_actual_plus_umm'], act_test_data.cons_actual_plus_umm)))
18/10: print("Mean Absolute Error : " + str(mean_absolute_error(predictions['predicted_cons_actual_excl_umm'], act_test_data.cons_actual_excl_umm)))
21/1:
import pandas as pd
import numpy as np
from datetime import datetime

import xgboost as xgb
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
21/2:
data = pd.read_csv("data/cons_training.csv", parse_dates=[0,1])
data.sort_values("start_time_utc", inplace=True)
cons_test_data = pd.read_csv("data/cons_testing_without_labels.csv", parse_dates=[0,1])
data.isnull().sum()
21/3:
# Only some temperatures and cons_fcast (which can't be used) are null. Fill them with prev vals
def pre_process(df):
    df['day_of_week'] = df.apply(lambda row: row.start_time_local.dayofweek, axis=1)
    return df

#Baseline: predict the current value from the last day's value: y[i] = y[i-24]?
data = pre_process(data)
data.describe()
21/4:
targets = ['cons_actual_plus_umm', 'cons_actual_excl_umm']
features_not_used = ['start_time_utc', 'start_time_local', 'cons_fcast_fingrid_excl_umm']
features_to_use = [c for c in data.columns if c not in targets + features_not_used]
21/5:


act_test_data = pd.read_csv("~/data/cons_testing.csv", parse_dates=[0,1])
act_test_data = pre_process(act_test_data)
act_X = act_test_data.drop(targets, axis=1)[features_to_use]
act_y = act_test_data.cons_actual_excl_umm

#predictions = my_model.predict(act_X.values)
#print("Mean Absolute Error : " + str(mean_absolute_error(predictions, act_y)))
21/6: predictions = pd.read_csv("/home/tman/Downloads/challenge/data/predictions.csv")
21/7: print("Mean Absolute Error : " + str(mean_absolute_error(predictions['predicted_cons_actual_excl_umm'], act_test_data.cons_actual_excl_umm)))
23/1:
import pandas as pd
import numpy as np
from datetime import datetime

import xgboost as xgb
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
23/2:
train_data = pd.read_csv("data/cons_training.csv", parse_dates=[0,1])
train_data.dtypes
23/3:
def pre_process(df):
    targets = ['cons_actual_plus_umm', 'cons_actual_excl_umm']
    features_not_used = ['start_time_utc', 'start_time_local', 'cons_fcast_fingrid_excl_umm']
    features_to_use = [c for c in df.columns if c not in targets + features_not_used] + ['day_of_week']
    # Only some temperatures and cons_fcast (which can't be used) are null. Fill them with prev row vals
    df = df.fillna(method='ffill')
    #Add day of week as feature
    df['day_of_week'] = df.apply(lambda row: row.start_time_local.dayofweek, axis=1)
    pre_X = df.drop(targets, axis=1)[features_to_use]
    return df, pre_X

data, X = pre_process(train_data)
cons_actual_plus_umm = data['cons_actual_plus_umm']
cons_actual_excl_umm = data['cons_actual_excl_umm']
23/4:
#Use a small test set so early stopping works
train_X, test_X, train_y, test_y = train_test_split(X, cons_actual_excl_umm, test_size=0.15)

#After some manual CV these seem to perform well. Possible add more gamma for regularization?
model_cons_excl_umm = XGBRegressor(n_estimators=1200, max_depth=9, n_jobs=4, gamma=3, colsample_bytree=0.6, 
                        min_child_weight=10, subsample=0.8)

model_cons_excl_umm.fit(train_X, train_y, early_stopping_rounds=5, 
             eval_set=[(test_X, test_y)], eval_metric='mae', verbose=False)

predictions = model_cons_excl_umm.predict(test_X)
print("Mean Absolute Error on held-out test set: " + str(mean_absolute_error(predictions, test_y)))
23/5:
train_X, test_X, train_y, test_y = train_test_split(X, cons_actual_plus_umm, test_size=0.15)

model_cons_plus_umm = XGBRegressor(n_estimators=1000, max_depth=9, n_jobs=4, gamma=3, colsample_bytree=0.6, 
                        min_child_weight=10, subsample=0.8)

model_cons_plus_umm.fit(train_X, train_y, early_stopping_rounds=5, 
             eval_set=[(test_X, test_y)], eval_metric='mae', verbose=False)

predictions = model_cons_plus_umm.predict(test_X)
print("Mean Absolute Error on held-out test set: " + str(mean_absolute_error(predictions, test_y)))
23/6:
#Use a small test set so early stopping works
train_X, test_X, train_y, test_y = train_test_split(X, cons_actual_excl_umm, test_size=0.15)

#After some manual CV these seem to perform well. Possible add more gamma for regularization?
model_cons_excl_umm = XGBRegressor(n_estimators=1700, max_depth=9, n_jobs=4, gamma=3, colsample_bytree=0.6, 
                        min_child_weight=10, subsample=0.8)

model_cons_excl_umm.fit(train_X, train_y, early_stopping_rounds=5, 
             eval_set=[(test_X, test_y)], eval_metric='mae', verbose=False)

predictions = model_cons_excl_umm.predict(test_X)
print("Mean Absolute Error on held-out test set: " + str(mean_absolute_error(predictions, test_y)))
23/7:
test_set_csv = pd.read_csv("data/cons_testing_without_labels.csv", parse_dates=[0,1])
test_set, test_set_X = pre_process(test_set_csv)
24/1:
import cv2
import datetime
import time
import numpy as np
import os
from sklearn.model_selection import train_test_split
import argparse
import warnings

from models import models_definition
from models.nn_models import NeuralNetwork
from data import data_loading
from features.preprocessing import preprocessing_dict
from metrics.model_metrics import compute_metrics
import config
from utils import str2bool
25/1:
import cv2
import datetime
import time
import numpy as np
import os
from sklearn.model_selection import train_test_split
import argparse
import warnings

from models import models_definition
from models.nn_models import NeuralNetwork
from data import data_loading
from features.preprocessing import preprocessing_dict
from metrics.model_metrics import compute_metrics
import config
from utils import str2bool
26/1:
import cv2
import datetime
import time
import numpy as np
import os
from sklearn.model_selection import train_test_split
import argparse
import warnings

from models import models_definition
from models.nn_models import NeuralNetwork
from data import data_loading
from features.preprocessing import preprocessing_dict
from metrics.model_metrics import compute_metrics
import config
from utils import str2bool
26/2:
import cv2
import datetime
import time
import numpy as np
import os
from sklearn.model_selection import train_test_split
import argparse
import warnings

from models import models_definition
from models.nn_models import NeuralNetwork
from data import data_loading
from features.preprocessing import preprocessing_dict
from metrics.model_metrics import compute_metrics
import config
from utils import str2bool
26/3:
input_dir = r"/home/tman/Work/data/harvester_data"
labels_source = "harvest"
image_source = "copernicus"

X, y, input_shape, output_dim = data_loading.import_data(input_dir)
26/4: len(X)
26/5: from PIL import Image
26/6: y[0]
26/7:
import cv2
import datetime
import time
import numpy as np
import os
from sklearn.model_selection import train_test_split
import argparse
import warnings

from models import models_definition
from models.nn_models import NeuralNetwork
from data import data_loading
from features.preprocessing import preprocessing_dict
from metrics.model_metrics import compute_metrics
import config
from utils import str2bool
from PIL import Image

import pandas as pd
import json
from tqdm import tqdm
import re
from sklearn.model_selection import train_test_split
26/8:
filename = "SE_harvest_566321,6766769,566421,6766919.geojson"
with open(os.path.join(input_dir, filename)) as f:
    data = json.load(f)
26/9: data
26/10: coord = np.asarray(data['features'][0]["geometry"]["coordinates"])[0]
26/11: coord
26/12: coord[:, 0].min(), coord[:, 0].max(), coord[:, 1].min(), coord[:, 1].max()
26/13: coord[:, 0].min(), coord[:, 1].min(), coord[:, 0].max(), coord[:, 1].max()
26/14: data['features'][0]
26/15: data['features'][0]["properties"]["fid"]
26/16: data['features'][0]["properties"]
26/17:
def cut_into_cells(input_path, labels_source, image_source, prediction_features,
                             cell_shape, __test__=False, verbose=False):
    """
    Divides a large image into cells fit for training using the data in the geojsons.
    :return X: the image data returned cut into cells as defined in the geojsons.
    :return y: the labels/targets from the geojsons
    """

    geo_jsons = [x for x in os.listdir(input_path) if '.geojson' in x and labels_source in x]

    X = []
    y = []
    big_image_id = 0

    n_faulty_cells = 0

    for file in tqdm(sorted(geo_jsons)):
        with open(os.path.join(input_path, file)) as f:
            data = json.load(f)

        file_bbox = np.asarray([int(x) for x in re.findall(r"\d+", file)])

        num_labels = len(data["features"])

        img = cv2.imread(os.path.join(input_path, file.replace(labels_source, image_source).replace(".geojson", ".png")))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_shape = img.shape

        assert input_shape[1] == (file_bbox[2] - file_bbox[0])
        assert input_shape[0] == (file_bbox[3] - file_bbox[1])

        dx = cell_shape[1]
        dy = cell_shape[0]

        if __test__:
            out_pine = np.zeros(input_shape)

        for feature in data["features"]:

            coord = np.asarray(feature["geometry"]["coordinates"])[0]

            # it is important to remember about horizontal flip.
            # i.e. y coord is reversed in qgis compared to numpy conventional order
            i = int((coord[:, 0].min() - file_bbox[0]) // dx)
            j = int(np.ceil((file_bbox[3] - coord[:, 1].max()) / dy))

            img_tmp = img[j*dy:(j+1)*dy, i*dx:(i+1)*dx, :]
            min_x = coord[:, 0].min()
            min_y = coord[:, 1].min()
            max_x = coord[:, 0].max()
            max_y = coord[:, 1].max()
            fid = feature["properties"]["fid"]
            if verbose:
                print("Expected bbox: ", coord[:, 0].min(), coord[:, 0].max(), coord[:, 1].min(), coord[:, 1].max())
                print("Retrieved bbox: ", i*dx + file_bbox[0], (i+1)*dx + file_bbox[0], file_bbox[3] - (j+1)*dy, file_bbox[3] - j*dy)

            try:
                assert img_tmp.shape == cell_shape, ("Wrong input shape.", i, j)
            except AssertionError:
                # Log when cell is irregularly shaped and can't be used, and pass this cell
                # Can happen that the cell coordinates in the geojson go over the satellite image coordinates,
                # in which case there wouldn't be enough pixels to cut an appropriate sized cell from the sat image.
                n_faulty_cells += 1
                continue
                # print("Assertion error")
                # print("coords were: ", coord)

            X.append(img_tmp)

            tmp_y = []
            tmp_y.append(fid)
            tmp_y.append(big_image_id)
            tmp_y.append(min_x)
            tmp_y.append(min_y)
            tmp_y.append(max_x)
            tmp_y.append(max_y)
            
            for item in prediction_features:
                tmp_y.append(feature["properties"][item])
            y.append(tmp_y)
            big_image_id += 1

    X = np.asarray(X)
    y = np.asarray(y)

    print("Faulty cells: ", n_faulty_cells)

    return X, y
26/18:
input_dir = r"/home/tman/Work/data/harvester_data"
labels_source = "harvest"
image_source = "copernicus"

X, y, input_shape, output_dim = data_loading.import_data(input_dir)
26/19: y[0]
29/1:
import cv2
import datetime
import time
import numpy as np
import os
from sklearn.model_selection import train_test_split
import argparse
import warnings
sys.path.insert(0, r'/home/tman/Work/linda-forestry-ml/species_prediction/regressors')

from models import models_definition
from models.nn_models import NeuralNetwork
from data import data_loading
from features.preprocessing import preprocessing_dict
from metrics.model_metrics import compute_metrics
import config
from utils import str2bool
from PIL import Image

import pandas as pd
import json
from tqdm import tqdm
import re
from sklearn.model_selection import train_test_split
29/2:
import cv2
import datetime
import time
import numpy as np
import os
from sklearn.model_selection import train_test_split
import argparse
import warnings
import sys
sys.path.insert(0, r'/home/tman/Work/linda-forestry-ml/species_prediction/regressors')

from models import models_definition
from models.nn_models import NeuralNetwork
from data import data_loading
from features.preprocessing import preprocessing_dict
from metrics.model_metrics import compute_metrics
import config
from utils import str2bool
from PIL import Image

import pandas as pd
import json
from tqdm import tqdm
import re
from sklearn.model_selection import train_test_split
29/3:
def cut_into_cells(input_path, labels_source, image_source, prediction_features,
                             cell_shape, __test__=False, verbose=False):
    """
    Divides a large image into cells fit for training using the data in the geojsons.
    :return X: the image data returned cut into cells as defined in the geojsons.
    :return y: the labels/targets from the geojsons
    """

    geo_jsons = [x for x in os.listdir(input_path) if '.geojson' in x and labels_source in x]

    X = []
    y = []
    big_image_id = 0

    n_faulty_cells = 0

    for file in tqdm(sorted(geo_jsons)):
        with open(os.path.join(input_path, file)) as f:
            data = json.load(f)

        file_bbox = np.asarray([int(x) for x in re.findall(r"\d+", file)])

        num_labels = len(data["features"])

        img = cv2.imread(os.path.join(input_path, file.replace(labels_source, image_source).replace(".geojson", ".png")))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_shape = img.shape

        assert input_shape[1] == (file_bbox[2] - file_bbox[0])
        assert input_shape[0] == (file_bbox[3] - file_bbox[1])

        dx = cell_shape[1]
        dy = cell_shape[0]

        if __test__:
            out_pine = np.zeros(input_shape)

        for feature in data["features"]:

            coord = np.asarray(feature["geometry"]["coordinates"])[0]

            # it is important to remember about horizontal flip.
            # i.e. y coord is reversed in qgis compared to numpy conventional order
            i = int((coord[:, 0].min() - file_bbox[0]) // dx)
            j = int(np.ceil((file_bbox[3] - coord[:, 1].max()) / dy))

            img_tmp = img[j*dy:(j+1)*dy, i*dx:(i+1)*dx, :]
            min_x = coord[:, 0].min()
            min_y = coord[:, 1].min()
            max_x = coord[:, 0].max()
            max_y = coord[:, 1].max()
            fid = feature["properties"]["fid"]
            if verbose:
                print("Expected bbox: ", coord[:, 0].min(), coord[:, 0].max(), coord[:, 1].min(), coord[:, 1].max())
                print("Retrieved bbox: ", i*dx + file_bbox[0], (i+1)*dx + file_bbox[0], file_bbox[3] - (j+1)*dy, file_bbox[3] - j*dy)

            try:
                assert img_tmp.shape == cell_shape, ("Wrong input shape.", i, j)
            except AssertionError:
                # Log when cell is irregularly shaped and can't be used, and pass this cell
                # Can happen that the cell coordinates in the geojson go over the satellite image coordinates,
                # in which case there wouldn't be enough pixels to cut an appropriate sized cell from the sat image.
                n_faulty_cells += 1
                continue
                # print("Assertion error")
                # print("coords were: ", coord)

            X.append(img_tmp)

            tmp_y = []
            tmp_y.append(fid)
            tmp_y.append(big_image_id)
            tmp_y.append(min_x)
            tmp_y.append(min_y)
            tmp_y.append(max_x)
            tmp_y.append(max_y)
            
            for item in prediction_features:
                tmp_y.append(feature["properties"][item])
            y.append(tmp_y)
            big_image_id += 1

    X = np.asarray(X)
    y = np.asarray(y)

    print("Faulty cells: ", n_faulty_cells)

    return X, y
29/4:
input_dir = r"/home/tman/Work/data/harvester_data"
labels_source = "harvest"
image_source = "copernicus"
prediction_features=['pine_volume', 'spruce_volume', 'birch_volume', 'other_bl_volume', 'contorta_volume']
cell_shape = (25, 25, 3)

X, y, input_shape, output_dim = cut_into_cells(input_dir, labels_source, image_source, prediction_features, cell_shape)
29/5:
input_dir = r"/home/tman/Work/data/harvester_data"
labels_source = "harvest"
image_source = "copernicus"
prediction_features=['pine_volume', 'spruce_volume', 'birch_volume', 'other_bl_volume', 'contorta_volume']
cell_shape = (25, 25, 3)

X, y = cut_into_cells(input_dir, labels_source, image_source, prediction_features, cell_shape)
29/6: y[0]
29/7: y[1]
29/8:
y_df = pd.DataFrame(y)
y_df.columns = ["fid", "big_image_id", "x_min", "y_min", "x_max", "y_max"] + prediction_features
y_df
29/9:
def cut_into_cells(input_path, output_path, labels_source, image_source, prediction_features,
                             cell_shape, __test__=False, verbose=False):
    """
    Divides a large image into cells fit for training using the data in the geojsons.
    :return X: the image data returned cut into cells as defined in the geojsons.
    :return y: the labels/targets from the geojsons
    """

    geo_jsons = [x for x in os.listdir(input_path) if '.geojson' in x and labels_source in x]

    X = []
    y = []
    virtual_cluster_id = 0 # Works as kind of a stand, since the big images are images where harvest cells are clustered.

    n_faulty_cells = 0

    for file in tqdm(sorted(geo_jsons)):
        with open(os.path.join(input_path, file)) as f:
            data = json.load(f)

        file_bbox = np.asarray([int(x) for x in re.findall(r"\d+", file)])

        num_labels = len(data["features"])

        img = cv2.imread(os.path.join(input_path, file.replace(labels_source, image_source).replace(".geojson", ".png")))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_shape = img.shape

        assert input_shape[1] == (file_bbox[2] - file_bbox[0])
        assert input_shape[0] == (file_bbox[3] - file_bbox[1])

        dx = cell_shape[1]
        dy = cell_shape[0]

        if __test__:
            out_pine = np.zeros(input_shape)

        for feature in data["features"]:

            coord = np.asarray(feature["geometry"]["coordinates"])[0]

            # it is important to remember about horizontal flip.
            # i.e. y coord is reversed in qgis compared to numpy conventional order
            i = int((coord[:, 0].min() - file_bbox[0]) // dx)
            j = int(np.ceil((file_bbox[3] - coord[:, 1].max()) / dy))

            img_tmp = img[j*dy:(j+1)*dy, i*dx:(i+1)*dx, :]
            x_min = coord[:, 0].min()
            y_min = coord[:, 1].min()
            x_max = coord[:, 0].max()
            y_max = coord[:, 1].max()
            fid = feature["properties"]["fid"] # Id in DB.
            if verbose:
                print("Expected bbox: ", coord[:, 0].min(), coord[:, 0].max(), coord[:, 1].min(), coord[:, 1].max())
                print("Retrieved bbox: ", i*dx + file_bbox[0], (i+1)*dx + file_bbox[0], file_bbox[3] - (j+1)*dy, file_bbox[3] - j*dy)

            try:
                assert img_tmp.shape == cell_shape, ("Wrong input shape.", i, j)
            except AssertionError:
                # Log when cell is irregularly shaped and can't be used, and pass this cell
                # Can happen that the cell coordinates in the geojson go over the satellite image coordinates,
                # in which case there wouldn't be enough pixels to cut an appropriate sized cell from the sat image.
                n_faulty_cells += 1
                continue
                # print("Assertion error")
                # print("coords were: ", coord)
            
            cv2.imwrite(os.path.join(output_path, str(fid), ".png"), img_tmp)
            return
            X.append(img_tmp)

            tmp_y = []
            tmp_y.append(fid)
            tmp_y.append(virtual_cluster_id)
            tmp_y.append(x_min)
            tmp_y.append(y_min)
            tmp_y.append(x_max)
            tmp_y.append(y_max)
            
            for item in prediction_features:
                tmp_y.append(feature["properties"][item])
            y.append(tmp_y)
        
        virtual_cluster_id += 1

    X = np.asarray(X)
    y = np.asarray(y)

    print("Faulty cells: ", n_faulty_cells)

    return X, y
29/10:
input_dir = r"/home/tman/Work/data/harvester_data"
output_dir = r"/home/tman/Work/data/harvester_data/harvester_data_processed"
labels_source = "harvest"
image_source = "copernicus"
prediction_features=['pine_volume', 'spruce_volume', 'birch_volume', 'other_bl_volume', 'contorta_volume']
cell_shape = (25, 25, 3)

X, y = cut_into_cells(input_dir, output_dir, labels_source, image_source, prediction_features, cell_shape)
30/1:
import cv2
import datetime
import time
import numpy as np
import os
from sklearn.model_selection import train_test_split
import argparse
import warnings
import sys
sys.path.insert(0, r'/home/tman/Work/linda-forestry-ml/species_prediction/regressors')

from models import models_definition
from models.nn_models import NeuralNetwork
from data import data_loading
from features.preprocessing import preprocessing_dict
from metrics.model_metrics import compute_metrics
import config
from utils import str2bool
from PIL import Image

import pandas as pd
import json
from tqdm import tqdm
import re
from sklearn.model_selection import train_test_split
30/2:
def cut_into_cells(input_path, output_path, labels_source, image_source, prediction_features,
                             cell_shape, __test__=False, verbose=False):
    """
    Divides a large image into cells fit for training using the data in the geojsons.
    :return X: the image data returned cut into cells as defined in the geojsons.
    :return y: the labels/targets from the geojsons
    """

    geo_jsons = [x for x in os.listdir(input_path) if '.geojson' in x and labels_source in x]

    X = []
    y = []
    virtual_cluster_id = 0 # Works as kind of a stand, since the big images are images where harvest cells are clustered.

    n_faulty_cells = 0

    for file in tqdm(sorted(geo_jsons)):
        with open(os.path.join(input_path, file)) as f:
            data = json.load(f)

        file_bbox = np.asarray([int(x) for x in re.findall(r"\d+", file)])

        num_labels = len(data["features"])

        img = cv2.imread(os.path.join(input_path, file.replace(labels_source, image_source).replace(".geojson", ".png")))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_shape = img.shape

        assert input_shape[1] == (file_bbox[2] - file_bbox[0])
        assert input_shape[0] == (file_bbox[3] - file_bbox[1])

        dx = cell_shape[1]
        dy = cell_shape[0]

        if __test__:
            out_pine = np.zeros(input_shape)

        for feature in data["features"]:

            coord = np.asarray(feature["geometry"]["coordinates"])[0]

            # it is important to remember about horizontal flip.
            # i.e. y coord is reversed in qgis compared to numpy conventional order
            i = int((coord[:, 0].min() - file_bbox[0]) // dx)
            j = int(np.ceil((file_bbox[3] - coord[:, 1].max()) / dy))

            img_tmp = img[j*dy:(j+1)*dy, i*dx:(i+1)*dx, :]
            x_min = coord[:, 0].min()
            y_min = coord[:, 1].min()
            x_max = coord[:, 0].max()
            y_max = coord[:, 1].max()
            fid = feature["properties"]["fid"] # Id in DB.
            if verbose:
                print("Expected bbox: ", coord[:, 0].min(), coord[:, 0].max(), coord[:, 1].min(), coord[:, 1].max())
                print("Retrieved bbox: ", i*dx + file_bbox[0], (i+1)*dx + file_bbox[0], file_bbox[3] - (j+1)*dy, file_bbox[3] - j*dy)

            try:
                assert img_tmp.shape == cell_shape, ("Wrong input shape.", i, j)
            except AssertionError:
                # Log when cell is irregularly shaped and can't be used, and pass this cell
                # Can happen that the cell coordinates in the geojson go over the satellite image coordinates,
                # in which case there wouldn't be enough pixels to cut an appropriate sized cell from the sat image.
                n_faulty_cells += 1
                continue
                # print("Assertion error")
                # print("coords were: ", coord)
            
            cv2.imwrite(os.path.join(output_path, str(fid), ".png"), img_tmp)
            return
            X.append(img_tmp)

            tmp_y = []
            tmp_y.append(fid)
            tmp_y.append(virtual_cluster_id)
            tmp_y.append(x_min)
            tmp_y.append(y_min)
            tmp_y.append(x_max)
            tmp_y.append(y_max)
            
            for item in prediction_features:
                tmp_y.append(feature["properties"][item])
            y.append(tmp_y)
        
        virtual_cluster_id += 1

    X = np.asarray(X)
    y = np.asarray(y)

    print("Faulty cells: ", n_faulty_cells)

    return X, y
30/3:
input_dir = r"/home/tman/Work/data/harvester_data"
output_dir = r"/home/tman/Work/data/harvester_data/harvester_data_processed"
labels_source = "harvest"
image_source = "copernicus"
prediction_features=['pine_volume', 'spruce_volume', 'birch_volume', 'other_bl_volume', 'contorta_volume']
cell_shape = (25, 25, 3)

X, y = cut_into_cells(input_dir, output_dir, labels_source, image_source, prediction_features, cell_shape)
30/4:
def cut_into_cells(input_path, output_path, labels_source, image_source, prediction_features,
                             cell_shape, __test__=False, verbose=False):
    """
    Divides a large image into cells fit for training using the data in the geojsons.
    :return X: the image data returned cut into cells as defined in the geojsons.
    :return y: the labels/targets from the geojsons
    """

    geo_jsons = [x for x in os.listdir(input_path) if '.geojson' in x and labels_source in x]
    print(len(geo_jsons))
    X = []
    y = []
    virtual_cluster_id = 0 # Works as kind of a stand, since the big images are images where harvest cells are clustered.

    n_faulty_cells = 0

    for file in tqdm(sorted(geo_jsons)):
        with open(os.path.join(input_path, file)) as f:
            data = json.load(f)

        file_bbox = np.asarray([int(x) for x in re.findall(r"\d+", file)])

        num_labels = len(data["features"])

        img = cv2.imread(os.path.join(input_path, file.replace(labels_source, image_source).replace(".geojson", ".png")))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_shape = img.shape

        assert input_shape[1] == (file_bbox[2] - file_bbox[0])
        assert input_shape[0] == (file_bbox[3] - file_bbox[1])

        dx = cell_shape[1]
        dy = cell_shape[0]

        if __test__:
            out_pine = np.zeros(input_shape)

        for feature in data["features"]:

            coord = np.asarray(feature["geometry"]["coordinates"])[0]

            # it is important to remember about horizontal flip.
            # i.e. y coord is reversed in qgis compared to numpy conventional order
            i = int((coord[:, 0].min() - file_bbox[0]) // dx)
            j = int(np.ceil((file_bbox[3] - coord[:, 1].max()) / dy))

            img_tmp = img[j*dy:(j+1)*dy, i*dx:(i+1)*dx, :]
            x_min = coord[:, 0].min()
            y_min = coord[:, 1].min()
            x_max = coord[:, 0].max()
            y_max = coord[:, 1].max()
            fid = feature["properties"]["fid"] # Id in DB.
            if verbose:
                print("Expected bbox: ", coord[:, 0].min(), coord[:, 0].max(), coord[:, 1].min(), coord[:, 1].max())
                print("Retrieved bbox: ", i*dx + file_bbox[0], (i+1)*dx + file_bbox[0], file_bbox[3] - (j+1)*dy, file_bbox[3] - j*dy)

            try:
                assert img_tmp.shape == cell_shape, ("Wrong input shape.", i, j)
            except AssertionError:
                # Log when cell is irregularly shaped and can't be used, and pass this cell
                # Can happen that the cell coordinates in the geojson go over the satellite image coordinates,
                # in which case there wouldn't be enough pixels to cut an appropriate sized cell from the sat image.
                n_faulty_cells += 1
                continue
                # print("Assertion error")
                # print("coords were: ", coord)
            
            cv2.imwrite(os.path.join(output_path, str(fid), ".png"), img_tmp)
            return
            X.append(img_tmp)

            tmp_y = []
            tmp_y.append(fid)
            tmp_y.append(virtual_cluster_id)
            tmp_y.append(x_min)
            tmp_y.append(y_min)
            tmp_y.append(x_max)
            tmp_y.append(y_max)
            
            for item in prediction_features:
                tmp_y.append(feature["properties"][item])
            y.append(tmp_y)
        
        virtual_cluster_id += 1

    X = np.asarray(X)
    y = np.asarray(y)

    print("Faulty cells: ", n_faulty_cells)

    return X, y
30/5:
input_dir = r"/home/tman/Work/data/harvester_data"
output_dir = r"/home/tman/Work/data/harvester_data/harvester_data_processed"
labels_source = "harvest"
image_source = "copernicus"
prediction_features=['pine_volume', 'spruce_volume', 'birch_volume', 'other_bl_volume', 'contorta_volume']
cell_shape = (25, 25, 3)

X, y = cut_into_cells(input_dir, output_dir, labels_source, image_source, prediction_features, cell_shape)
30/6:
def cut_into_cells(input_path, output_path, labels_source, image_source, prediction_features,
                             cell_shape, __test__=False, verbose=False):
    """
    Divides a large image into cells fit for training using the data in the geojsons.
    :return X: the image data returned cut into cells as defined in the geojsons.
    :return y: the labels/targets from the geojsons
    """

    geo_jsons = [x for x in os.listdir(input_path) if '.geojson' in x and labels_source in x]
    X = []
    y = []
    virtual_cluster_id = 0 # Works as kind of a stand, since the big images are images where harvest cells are clustered.

    n_faulty_cells = 0

    for file in tqdm(sorted(geo_jsons)):
        with open(os.path.join(input_path, file)) as f:
            data = json.load(f)

        file_bbox = np.asarray([int(x) for x in re.findall(r"\d+", file)])

        num_labels = len(data["features"])

        img = cv2.imread(os.path.join(input_path, file.replace(labels_source, image_source).replace(".geojson", ".png")))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_shape = img.shape

        assert input_shape[1] == (file_bbox[2] - file_bbox[0])
        assert input_shape[0] == (file_bbox[3] - file_bbox[1])

        dx = cell_shape[1]
        dy = cell_shape[0]

        if __test__:
            out_pine = np.zeros(input_shape)

        for feature in data["features"]:

            coord = np.asarray(feature["geometry"]["coordinates"])[0]

            # it is important to remember about horizontal flip.
            # i.e. y coord is reversed in qgis compared to numpy conventional order
            i = int((coord[:, 0].min() - file_bbox[0]) // dx)
            j = int(np.ceil((file_bbox[3] - coord[:, 1].max()) / dy))

            img_tmp = img[j*dy:(j+1)*dy, i*dx:(i+1)*dx, :]
            x_min = coord[:, 0].min()
            y_min = coord[:, 1].min()
            x_max = coord[:, 0].max()
            y_max = coord[:, 1].max()
            fid = feature["properties"]["fid"] # Id in DB.
            if verbose:
                print("Expected bbox: ", coord[:, 0].min(), coord[:, 0].max(), coord[:, 1].min(), coord[:, 1].max())
                print("Retrieved bbox: ", i*dx + file_bbox[0], (i+1)*dx + file_bbox[0], file_bbox[3] - (j+1)*dy, file_bbox[3] - j*dy)

            try:
                assert img_tmp.shape == cell_shape, ("Wrong input shape.", i, j)
            except AssertionError:
                # Log when cell is irregularly shaped and can't be used, and pass this cell
                # Can happen that the cell coordinates in the geojson go over the satellite image coordinates,
                # in which case there wouldn't be enough pixels to cut an appropriate sized cell from the sat image.
                n_faulty_cells += 1
                continue
                # print("Assertion error")
                # print("coords were: ", coord)
            
            print("gets here lol")
            cv2.imwrite(os.path.join(output_path, str(fid), ".png"), img_tmp)
            return
            X.append(img_tmp)

            tmp_y = []
            tmp_y.append(fid)
            tmp_y.append(virtual_cluster_id)
            tmp_y.append(x_min)
            tmp_y.append(y_min)
            tmp_y.append(x_max)
            tmp_y.append(y_max)
            
            for item in prediction_features:
                tmp_y.append(feature["properties"][item])
            y.append(tmp_y)
        
        virtual_cluster_id += 1

    X = np.asarray(X)
    y = np.asarray(y)

    print("Faulty cells: ", n_faulty_cells)

    return X, y
30/7:
input_dir = r"/home/tman/Work/data/harvester_data"
output_dir = r"/home/tman/Work/data/harvester_data/harvester_data_processed"
labels_source = "harvest"
image_source = "copernicus"
prediction_features=['pine_volume', 'spruce_volume', 'birch_volume', 'other_bl_volume', 'contorta_volume']
cell_shape = (25, 25, 3)

X, y = cut_into_cells(input_dir, output_dir, labels_source, image_source, prediction_features, cell_shape)
30/8:
input_dir = r"/home/tman/Work/data/harvester_data"
output_dir = r"/home/tman/Work/data/harvester_data/harvester_data_processed"
labels_source = "harvest"
image_source = "copernicus"
prediction_features=['pine_volume', 'spruce_volume', 'birch_volume', 'other_bl_volume', 'contorta_volume']
cell_shape = (25, 25, 3)

cut_into_cells(input_dir, output_dir, labels_source, image_source, prediction_features, cell_shape)
30/9:
def cut_into_cells(input_path, output_path, labels_source, image_source, prediction_features,
                             cell_shape, __test__=False, verbose=False):
    """
    Divides a large image into cells fit for training using the data in the geojsons.
    :return X: the image data returned cut into cells as defined in the geojsons.
    :return y: the labels/targets from the geojsons
    """

    geo_jsons = [x for x in os.listdir(input_path) if '.geojson' in x and labels_source in x]
    X = []
    y = []
    virtual_cluster_id = 0 # Works as kind of a stand, since the big images are images where harvest cells are clustered.

    n_faulty_cells = 0

    for file in tqdm(sorted(geo_jsons)):
        with open(os.path.join(input_path, file)) as f:
            data = json.load(f)

        file_bbox = np.asarray([int(x) for x in re.findall(r"\d+", file)])

        num_labels = len(data["features"])

        img = cv2.imread(os.path.join(input_path, file.replace(labels_source, image_source).replace(".geojson", ".png")))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_shape = img.shape

        assert input_shape[1] == (file_bbox[2] - file_bbox[0])
        assert input_shape[0] == (file_bbox[3] - file_bbox[1])

        dx = cell_shape[1]
        dy = cell_shape[0]

        if __test__:
            out_pine = np.zeros(input_shape)

        for feature in data["features"]:

            coord = np.asarray(feature["geometry"]["coordinates"])[0]

            # it is important to remember about horizontal flip.
            # i.e. y coord is reversed in qgis compared to numpy conventional order
            i = int((coord[:, 0].min() - file_bbox[0]) // dx)
            j = int(np.ceil((file_bbox[3] - coord[:, 1].max()) / dy))

            img_tmp = img[j*dy:(j+1)*dy, i*dx:(i+1)*dx, :]
            x_min = coord[:, 0].min()
            y_min = coord[:, 1].min()
            x_max = coord[:, 0].max()
            y_max = coord[:, 1].max()
            fid = feature["properties"]["fid"] # Id in DB.
            if verbose:
                print("Expected bbox: ", coord[:, 0].min(), coord[:, 0].max(), coord[:, 1].min(), coord[:, 1].max())
                print("Retrieved bbox: ", i*dx + file_bbox[0], (i+1)*dx + file_bbox[0], file_bbox[3] - (j+1)*dy, file_bbox[3] - j*dy)

            try:
                assert img_tmp.shape == cell_shape, ("Wrong input shape.", i, j)
            except AssertionError:
                # Log when cell is irregularly shaped and can't be used, and pass this cell
                # Can happen that the cell coordinates in the geojson go over the satellite image coordinates,
                # in which case there wouldn't be enough pixels to cut an appropriate sized cell from the sat image.
                n_faulty_cells += 1
                continue
                # print("Assertion error")
                # print("coords were: ", coord)
            
            print(os.path.join(output_path, str(fid), ".png"))
            cv2.imwrite(os.path.join(output_path, str(fid), ".png"), img_tmp)
            return
            X.append(img_tmp)

            tmp_y = []
            tmp_y.append(fid)
            tmp_y.append(virtual_cluster_id)
            tmp_y.append(x_min)
            tmp_y.append(y_min)
            tmp_y.append(x_max)
            tmp_y.append(y_max)
            
            for item in prediction_features:
                tmp_y.append(feature["properties"][item])
            y.append(tmp_y)
        
        virtual_cluster_id += 1

    X = np.asarray(X)
    y = np.asarray(y)

    print("Faulty cells: ", n_faulty_cells)

    return X, y
30/10:
input_dir = r"/home/tman/Work/data/harvester_data"
output_dir = r"/home/tman/Work/data/harvester_data/harvester_data_processed"
labels_source = "harvest"
image_source = "copernicus"
prediction_features=['pine_volume', 'spruce_volume', 'birch_volume', 'other_bl_volume', 'contorta_volume']
cell_shape = (25, 25, 3)

cut_into_cells(input_dir, output_dir, labels_source, image_source, prediction_features, cell_shape)
30/11:
def cut_into_cells(input_path, output_path, labels_source, image_source, prediction_features,
                             cell_shape, __test__=False, verbose=False):
    """
    Divides a large image into cells fit for training using the data in the geojsons.
    :return X: the image data returned cut into cells as defined in the geojsons.
    :return y: the labels/targets from the geojsons
    """

    geo_jsons = [x for x in os.listdir(input_path) if '.geojson' in x and labels_source in x]
    X = []
    y = []
    virtual_cluster_id = 0 # Works as kind of a stand, since the big images are images where harvest cells are clustered.

    n_faulty_cells = 0

    for file in tqdm(sorted(geo_jsons)):
        with open(os.path.join(input_path, file)) as f:
            data = json.load(f)

        file_bbox = np.asarray([int(x) for x in re.findall(r"\d+", file)])

        num_labels = len(data["features"])

        img = cv2.imread(os.path.join(input_path, file.replace(labels_source, image_source).replace(".geojson", ".png")))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_shape = img.shape

        assert input_shape[1] == (file_bbox[2] - file_bbox[0])
        assert input_shape[0] == (file_bbox[3] - file_bbox[1])

        dx = cell_shape[1]
        dy = cell_shape[0]

        if __test__:
            out_pine = np.zeros(input_shape)

        for feature in data["features"]:

            coord = np.asarray(feature["geometry"]["coordinates"])[0]

            # it is important to remember about horizontal flip.
            # i.e. y coord is reversed in qgis compared to numpy conventional order
            i = int((coord[:, 0].min() - file_bbox[0]) // dx)
            j = int(np.ceil((file_bbox[3] - coord[:, 1].max()) / dy))

            img_tmp = img[j*dy:(j+1)*dy, i*dx:(i+1)*dx, :]
            x_min = coord[:, 0].min()
            y_min = coord[:, 1].min()
            x_max = coord[:, 0].max()
            y_max = coord[:, 1].max()
            fid = feature["properties"]["fid"] # Id in DB.
            if verbose:
                print("Expected bbox: ", coord[:, 0].min(), coord[:, 0].max(), coord[:, 1].min(), coord[:, 1].max())
                print("Retrieved bbox: ", i*dx + file_bbox[0], (i+1)*dx + file_bbox[0], file_bbox[3] - (j+1)*dy, file_bbox[3] - j*dy)

            try:
                assert img_tmp.shape == cell_shape, ("Wrong input shape.", i, j)
            except AssertionError:
                # Log when cell is irregularly shaped and can't be used, and pass this cell
                # Can happen that the cell coordinates in the geojson go over the satellite image coordinates,
                # in which case there wouldn't be enough pixels to cut an appropriate sized cell from the sat image.
                n_faulty_cells += 1
                continue
                # print("Assertion error")
                # print("coords were: ", coord)
            
            print(os.path.join(output_path, str(fid) + ".png"))
            cv2.imwrite(os.path.join(output_path, str(fid), ".png"), img_tmp)
            return
            X.append(img_tmp)

            tmp_y = []
            tmp_y.append(fid)
            tmp_y.append(virtual_cluster_id)
            tmp_y.append(x_min)
            tmp_y.append(y_min)
            tmp_y.append(x_max)
            tmp_y.append(y_max)
            
            for item in prediction_features:
                tmp_y.append(feature["properties"][item])
            y.append(tmp_y)
        
        virtual_cluster_id += 1

    X = np.asarray(X)
    y = np.asarray(y)

    print("Faulty cells: ", n_faulty_cells)

    return X, y
30/12:
input_dir = r"/home/tman/Work/data/harvester_data"
output_dir = r"/home/tman/Work/data/harvester_data/harvester_data_processed"
labels_source = "harvest"
image_source = "copernicus"
prediction_features=['pine_volume', 'spruce_volume', 'birch_volume', 'other_bl_volume', 'contorta_volume']
cell_shape = (25, 25, 3)

cut_into_cells(input_dir, output_dir, labels_source, image_source, prediction_features, cell_shape)
30/13:
def cut_into_cells(input_path, output_path, labels_source, image_source, prediction_features,
                             cell_shape, __test__=False, verbose=False):
    """
    Divides a large image into cells fit for training using the data in the geojsons.
    :return X: the image data returned cut into cells as defined in the geojsons.
    :return y: the labels/targets from the geojsons
    """

    geo_jsons = [x for x in os.listdir(input_path) if '.geojson' in x and labels_source in x]
    X = []
    y = []
    virtual_cluster_id = 0 # Works as kind of a stand, since the big images are images where harvest cells are clustered.

    n_faulty_cells = 0

    for file in tqdm(sorted(geo_jsons)):
        with open(os.path.join(input_path, file)) as f:
            data = json.load(f)

        file_bbox = np.asarray([int(x) for x in re.findall(r"\d+", file)])

        num_labels = len(data["features"])

        img = cv2.imread(os.path.join(input_path, file.replace(labels_source, image_source).replace(".geojson", ".png")))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_shape = img.shape

        assert input_shape[1] == (file_bbox[2] - file_bbox[0])
        assert input_shape[0] == (file_bbox[3] - file_bbox[1])

        dx = cell_shape[1]
        dy = cell_shape[0]

        if __test__:
            out_pine = np.zeros(input_shape)

        for feature in data["features"]:

            coord = np.asarray(feature["geometry"]["coordinates"])[0]

            # it is important to remember about horizontal flip.
            # i.e. y coord is reversed in qgis compared to numpy conventional order
            i = int((coord[:, 0].min() - file_bbox[0]) // dx)
            j = int(np.ceil((file_bbox[3] - coord[:, 1].max()) / dy))

            img_tmp = img[j*dy:(j+1)*dy, i*dx:(i+1)*dx, :]
            x_min = coord[:, 0].min()
            y_min = coord[:, 1].min()
            x_max = coord[:, 0].max()
            y_max = coord[:, 1].max()
            fid = feature["properties"]["fid"] # Id in DB.
            if verbose:
                print("Expected bbox: ", coord[:, 0].min(), coord[:, 0].max(), coord[:, 1].min(), coord[:, 1].max())
                print("Retrieved bbox: ", i*dx + file_bbox[0], (i+1)*dx + file_bbox[0], file_bbox[3] - (j+1)*dy, file_bbox[3] - j*dy)

            try:
                assert img_tmp.shape == cell_shape, ("Wrong input shape.", i, j)
            except AssertionError:
                # Log when cell is irregularly shaped and can't be used, and pass this cell
                # Can happen that the cell coordinates in the geojson go over the satellite image coordinates,
                # in which case there wouldn't be enough pixels to cut an appropriate sized cell from the sat image.
                n_faulty_cells += 1
                continue
                # print("Assertion error")
                # print("coords were: ", coord)
            
            print(os.path.join(output_path, str(fid) + ".png"))
            cv2.imwrite(os.path.join(output_path, str(fid) + ".png"), img_tmp)
            return
            X.append(img_tmp)

            tmp_y = []
            tmp_y.append(fid)
            tmp_y.append(virtual_cluster_id)
            tmp_y.append(x_min)
            tmp_y.append(y_min)
            tmp_y.append(x_max)
            tmp_y.append(y_max)
            
            for item in prediction_features:
                tmp_y.append(feature["properties"][item])
            y.append(tmp_y)
        
        virtual_cluster_id += 1

    X = np.asarray(X)
    y = np.asarray(y)

    print("Faulty cells: ", n_faulty_cells)

    return X, y
30/14:
input_dir = r"/home/tman/Work/data/harvester_data"
output_dir = r"/home/tman/Work/data/harvester_data/harvester_data_processed"
labels_source = "harvest"
image_source = "copernicus"
prediction_features=['pine_volume', 'spruce_volume', 'birch_volume', 'other_bl_volume', 'contorta_volume']
cell_shape = (25, 25, 3)

cut_into_cells(input_dir, output_dir, labels_source, image_source, prediction_features, cell_shape)
30/15:
input_dir = r"/home/tman/Work/data/harvester_data"
output_dir = r"/home/tman/Work/data/harvester_data_processed"
labels_source = "harvest"
image_source = "copernicus"
prediction_features=['pine_volume', 'spruce_volume', 'birch_volume', 'other_bl_volume', 'contorta_volume']
cell_shape = (25, 25, 3)

cut_into_cells(input_dir, output_dir, labels_source, image_source, prediction_features, cell_shape)
30/16:
def cut_into_cells(input_path, output_path, labels_source, image_source, prediction_features,
                             cell_shape, __test__=False, verbose=False):
    """
    Divides a large image into cells fit for training using the data in the geojsons.
    
    Saves the images cell by cell and saves the csv that has labels with the fid as the key.
    
    """

    geo_jsons = [x for x in os.listdir(input_path) if '.geojson' in x and labels_source in x]
    X = []
    y = []
    virtual_cluster_id = 0 # Works as kind of a stand, since the big images are images where harvest cells are clustered.

    n_faulty_cells = 0

    for file in tqdm(sorted(geo_jsons)):
        with open(os.path.join(input_path, file)) as f:
            data = json.load(f)

        file_bbox = np.asarray([int(x) for x in re.findall(r"\d+", file)])

        num_labels = len(data["features"])

        img = cv2.imread(os.path.join(input_path, file.replace(labels_source, image_source).replace(".geojson", ".png")))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_shape = img.shape

        assert input_shape[1] == (file_bbox[2] - file_bbox[0])
        assert input_shape[0] == (file_bbox[3] - file_bbox[1])

        dx = cell_shape[1]
        dy = cell_shape[0]

        if __test__:
            out_pine = np.zeros(input_shape)

        for feature in data["features"]:

            coord = np.asarray(feature["geometry"]["coordinates"])[0]

            # it is important to remember about horizontal flip.
            # i.e. y coord is reversed in qgis compared to numpy conventional order
            i = int((coord[:, 0].min() - file_bbox[0]) // dx)
            j = int(np.ceil((file_bbox[3] - coord[:, 1].max()) / dy))

            img_tmp = img[j*dy:(j+1)*dy, i*dx:(i+1)*dx, :]
            x_min = coord[:, 0].min()
            y_min = coord[:, 1].min()
            x_max = coord[:, 0].max()
            y_max = coord[:, 1].max()
            fid = feature["properties"]["fid"] # Id in DB.
            if verbose:
                print("Expected bbox: ", coord[:, 0].min(), coord[:, 0].max(), coord[:, 1].min(), coord[:, 1].max())
                print("Retrieved bbox: ", i*dx + file_bbox[0], (i+1)*dx + file_bbox[0], file_bbox[3] - (j+1)*dy, file_bbox[3] - j*dy)

            try:
                assert img_tmp.shape == cell_shape, ("Wrong input shape.", i, j)
            except AssertionError:
                # Log when cell is irregularly shaped and can't be used, and pass this cell
                # Can happen that the cell coordinates in the geojson go over the satellite image coordinates,
                # in which case there wouldn't be enough pixels to cut an appropriate sized cell from the sat image.
                n_faulty_cells += 1
                continue
                # print("Assertion error")
                # print("coords were: ", coord)
            
            print(os.path.join(output_path, str(fid) + ".png"))
            cv2.imwrite(os.path.join(output_path, str(fid) + ".png"), img_tmp)
            X.append(img_tmp)

            tmp_y = []
            tmp_y.append(fid)
            tmp_y.append(virtual_cluster_id)
            tmp_y.append(x_min)
            tmp_y.append(y_min)
            tmp_y.append(x_max)
            tmp_y.append(y_max)
            
            for item in prediction_features:
                tmp_y.append(feature["properties"][item])
            y.append(tmp_y)
        
        virtual_cluster_id += 1

    X = np.asarray(X)
    y = np.asarray(y)

    print("Faulty cells: ", n_faulty_cells)

    return X, y
30/17:
input_dir = r"/home/tman/Work/data/harvester_data"
output_dir = r"/home/tman/Work/data/harvester_data_processed"
labels_source = "harvest"
image_source = "copernicus"
prediction_features = ['pine_volume', 'spruce_volume', 'birch_volume', 'other_bl_volume', 'contorta_volume']
cell_shape = (25, 25, 3)

cut_into_cells(input_dir, output_dir, labels_source, image_source, prediction_features, cell_shape)
30/18:
y_df = pd.DataFrame(y)
y_df.columns = ["fid", "virtual_cluster_id", "x_min", "y_min", "x_max", "y_max"] + prediction_features
y_df.to_csv(os.path.join(output_dir, "groundtruth.csv"))
30/19:
def cut_into_cells(input_path, output_path, labels_source, image_source, prediction_features,
                             cell_shape, __test__=False, verbose=False):
    """
    Divides a large image into cells fit for training using the data in the geojsons.
    
    Saves the images cell by cell and saves the csv that has labels with the fid as the key.
    
    """

    geo_jsons = [x for x in os.listdir(input_path) if '.geojson' in x and labels_source in x]
    X = []
    y = []
    virtual_cluster_id = 0 # Works as kind of a stand, since the big images are images where harvest cells are clustered.

    n_faulty_cells = 0

    for file in tqdm(sorted(geo_jsons)):
        with open(os.path.join(input_path, file)) as f:
            data = json.load(f)

        file_bbox = np.asarray([int(x) for x in re.findall(r"\d+", file)])

        num_labels = len(data["features"])

        img = cv2.imread(os.path.join(input_path, file.replace(labels_source, image_source).replace(".geojson", ".png")))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_shape = img.shape

        assert input_shape[1] == (file_bbox[2] - file_bbox[0])
        assert input_shape[0] == (file_bbox[3] - file_bbox[1])

        dx = cell_shape[1]
        dy = cell_shape[0]

        if __test__:
            out_pine = np.zeros(input_shape)

        for feature in data["features"]:

            coord = np.asarray(feature["geometry"]["coordinates"])[0]

            # it is important to remember about horizontal flip.
            # i.e. y coord is reversed in qgis compared to numpy conventional order
            i = int((coord[:, 0].min() - file_bbox[0]) // dx)
            j = int(np.ceil((file_bbox[3] - coord[:, 1].max()) / dy))

            img_tmp = img[j*dy:(j+1)*dy, i*dx:(i+1)*dx, :]
            x_min = coord[:, 0].min()
            y_min = coord[:, 1].min()
            x_max = coord[:, 0].max()
            y_max = coord[:, 1].max()
            fid = feature["properties"]["fid"] # Id in DB.
            if verbose:
                print("Expected bbox: ", coord[:, 0].min(), coord[:, 0].max(), coord[:, 1].min(), coord[:, 1].max())
                print("Retrieved bbox: ", i*dx + file_bbox[0], (i+1)*dx + file_bbox[0], file_bbox[3] - (j+1)*dy, file_bbox[3] - j*dy)

            try:
                assert img_tmp.shape == cell_shape, ("Wrong input shape.", i, j)
            except AssertionError:
                # Log when cell is irregularly shaped and can't be used, and pass this cell
                # Can happen that the cell coordinates in the geojson go over the satellite image coordinates,
                # in which case there wouldn't be enough pixels to cut an appropriate sized cell from the sat image.
                n_faulty_cells += 1
                continue
                # print("Assertion error")
                # print("coords were: ", coord)
            
            cv2.imwrite(os.path.join(output_path, str(fid) + ".png"), img_tmp)
            X.append(img_tmp)

            tmp_y = []
            tmp_y.append(fid)
            tmp_y.append(virtual_cluster_id)
            tmp_y.append(x_min)
            tmp_y.append(y_min)
            tmp_y.append(x_max)
            tmp_y.append(y_max)
            
            for item in prediction_features:
                tmp_y.append(feature["properties"][item])
            y.append(tmp_y)
        
        virtual_cluster_id += 1

    X = np.asarray(X)
    y = np.asarray(y)

    print("Faulty cells: ", n_faulty_cells)

    return X, y
30/20:
input_dir = r"/home/tman/Work/data/harvester_data"
output_dir = r"/home/tman/Work/data/harvester_data_processed"
labels_source = "harvest"
image_source = "copernicus"
prediction_features = ['pine_volume', 'spruce_volume', 'birch_volume', 'other_bl_volume', 'contorta_volume']
cell_shape = (25, 25, 3)

X, y = cut_into_cells(input_dir, output_dir, labels_source, image_source, prediction_features, cell_shape)
30/21:
y_df = pd.DataFrame(y)
y_df.columns = ["fid", "virtual_cluster_id", "x_min", "y_min", "x_max", "y_max"] + prediction_features
y_df.to_csv(os.path.join(output_dir, "groundtruth.csv"))
31/1: import sys, os
31/2: os.path.dirname(sys.executable)
32/1: import sys; print('Python %s on %s' % (sys.version, sys.platform))
32/2: X[0]
33/1: import sys; print('Python %s on %s' % (sys.version, sys.platform))
33/2: X_scalar[0]
33/3: X_scalar.shape
33/4: X_features = preprocessing_dict['image_features'](X)
33/5: X_features[0]
33/6: X_features[0].shape
33/7: X_features = preprocessing_dict['image_to_features'](X)
33/8: X_features[0]
33/9: X_features.shape
33/10: np.concatenate(X_scalar, X_features)
33/11: np.hstack(X_scalar, X_features)
33/12: np.hstack([X_scalar, X_features]).shape
34/1: import sys; print('Python %s on %s' % (sys.version, sys.platform))
34/2: X[0]
34/3: X[0].shape
34/4: np.sum(X > 10000)
34/5: np.isnan(X).sum()
34/6: gg = X[np.isnan(X)]
34/7: gg[0]
34/8: gg
34/9: np.where(np.isnan(X))
35/1: import sys; print('Python %s on %s' % (sys.version, sys.platform))
35/2: y.shape
36/1: import sys; print('Python %s on %s' % (sys.version, sys.platform))
36/2: y.shape
37/1: import sys; print('Python %s on %s' % (sys.version, sys.platform))
37/2: y[0]
37/3: np.sum(np.isnan(y[:,0:3]))
37/4: np.sum(np.isnan(y[:,0:4]))
37/5: np.sum(np.isnan(y[:,0:2]))
38/1: import sys; print('Python %s on %s' % (sys.version, sys.platform))
39/1: import sys; print('Python %s on %s' % (sys.version, sys.platform))
39/2: non_nan_indexes = np.any(np.array[non_nan_X, non_nan_y], axis=1)
39/3: non_nan_indexes = np.any(np.array([non_nan_X, non_nan_y]), axis=1)
39/4: non_nan_indexes.shape
39/5: non_nan_indexes = np.any(np.array([non_nan_X, non_nan_y]), axis=0)
39/6: non_nan_indexes.shape
39/7: np.sum(non_nan_indexes)
39/8: non_nan_indexes = np.all(np.array([non_nan_X, non_nan_y]), axis=0)
39/9: non_nan_indexes.shape
39/10: np.sum(non_nan_indexes)
39/11: X[non_nan_indexes].shape
40/1: tf.test.is_gpu_available()
40/2: import tensorflow as tf
40/3: tf.test.is_gpu_available()
44/1: import data_loading
44/2: import data
44/3: import pickle
44/4: datapath = r"/home/tman/Work/data/FIsampletiles"
44/5: datapath = r"/home/tman/Work/data/FIsampletiles/cache"
44/6: X, y, input_shape, output_dim = pickle.load(open(datapath, "rb"))
44/7: datapath = r"/home/tman/Work/data/FIsampletiles/cache/pickled_data.p"
44/8: X, y, input_shape, output_dim = pickle.load(open(datapath, "rb"))
44/9: X
44/10: X.shape
44/11: from sklearn.preprocessing import OneHotEncoder
44/12:
X[-5:
]
44/13: X[:5:-5]
44/14: X[:5,-5]
44/15: X[:5,-3]
44/16: test = X
44/17: test = X[:5,-3]
44/18: OneHotEncoder(test)
44/19: OneHotEncoder().fit(test)
44/20: test.shape
44/21: test.reshape(-1, 1)
44/22: test.reshape(-1, 1).shape
44/23: test = X[:5,-3].reshape(-1, 1)
44/24: OneHotEncoder().fit(test)
44/25: test = X[:10,-3].reshape(-1, 1)
44/26: test
44/27: test = X[:30,-3].reshape(-1, 1)
44/28: test
44/29: encoder = OneHotEncoder()
44/30: encoder.fit_transform(test)
44/31: encoder = OneHotEncoder(sparse=False)
44/32: encoder.fit_transform(test)
44/33: encoder.fit_transform(test).shape
44/34: test = X[:5,[-3, -1]]
44/35: test
44/36: test = X[:30,[-3, -1]]
44/37: test
44/38: encoder.fit_transform(test)
44/39: X_copy = X.copy()
44/40: X_cope
44/41: X_copy
44/42: X_copy.shape
44/43: columns = [-3, -1]
44/44: tt = X_copy[:,columns]
44/45: np.delete(X_copy, columns, axis=1)
44/46: import numpy as np
44/47: np.delete(X_copy, columns, axis=1)
44/48: np.delete(X_copy, columns, axis=1).shape
44/49: X_cope.shape
44/50: X_copy.shape
44/51: np.delete(X_copy, columns, axis=1).shape
44/52: np.delete(X_copy, [57,60], axis=1).shape
44/53: np.append(X, X, axis=1).shape
45/1:
from tqdm import tqdm
import os
from PIL import Image
import numpy as np
import json
import psycopg2
import pandas.io.sql as sqlio
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import sys
import cv2
import pickle
pd.options.display.float_format = '{:,.2f}'.format
# Add path to where utils.py is sometrics can be imported
sys.path.insert(0, r'C:/Users/Teemu/Work/linda-forestry-ml/species_prediction/regressors')

from models import models_definition
from data import data_loading
45/2:
from tqdm import tqdm
import os
from PIL import Image
import numpy as np
import json
import psycopg2
import pandas.io.sql as sqlio
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import sys
import cv2
import pickle
pd.options.display.float_format = '{:,.2f}'.format
# Add path to where utils.py is sometrics can be imported
sys.path.insert(0, r'/home/tman/Work/linda-forestry-ml/species_prediction/regressors')

from models import models_definition
from data import data_loading
45/3:
output_dim = 4

model_function = models_definition.create_xgboost ## output dim etc?
model = model_function(2, output_dim, random_state=50)
load = "../regressors/models/xgboost_scalars_generic.2018-11-13.15-21-24"
model.load(load)
45/4:
output_dim = 4

model_function = models_definition.create_xgboost ## output dim etc?
model = model_function(2, output_dim, random_state=50)
load = "../regressors/models/xgboost_scalars_generic.2018-11-13.15-21-24"
model.load(load)
46/1: import sys; print('Python %s on %s' % (sys.version, sys.platform))
46/2: data[:5]
46/3: data['gridcellid']
46/4: data['gridcellid'].shape
46/5: data['gridcellid'].values
46/6: data['gridcellid'].values.shape
46/7: gridcellids = np.expand_dims(data['gridcellid'].values, axis=1)
46/8: gridcellids.shape
51/1: from data import data_loading
51/2: data_loading.create_test_set_from_ids("/home/tman/Work/data/FIsampletiles/groundtruth.csv", "/home/tman/Work/data/FIsampletiles/")
51/3: data_loading.create_test_set_from_ids("/home/tman/Work/data/FIsampletiles/groundtruth.csv", "/home/tman/Work/data/FIsampletiles/")
52/1: from data import data_loading
52/2: data_loading.create_test_set_from_ids("/home/tman/Work/data/FIsampletiles/groundtruth.csv", "/home/tman/Work/data/FIsampletiles/")
53/1: import data_loading
53/2: data_loading.create_test_set_from_ids("/home/tman/Work/data/FIsampletiles/groundtruth.csv", "/home/tman/Work/data/FIsampletiles/")
55/1: cd data
55/2: import data_loading
55/3: input_path = r"/home/tman/Work/data/FIsampletiles", image_dir="azure_tiles_cleaned", image_type="jpg"
55/4: input_path = r"/home/tman/Work/data/FIsampletiles"
55/5: images, data = data_loading.import_data(input_path, "groundtruth.csv", image_dir="azure_tiles_cleaned", image_type="jpg")
55/6: len(data)
55/7: data['plot_type']
55/8: data.groupby('plot_type').count()
55/9: data.groupby('plot_type').value_counts()
55/10: data['plot_type'].value_counts()
55/11: images, data = data_loading.import_data(input_path, "test.csv", image_dir="azure_tiles_cleaned", image_type="jpg")
55/12: data['plot_type'].value_counts()
55/13: images, data = data_loading.import_data(input_path, "train.csv", image_dir="azure_tiles_cleaned", image_type="jpg")
55/14: data['plot_type'].value_counts()
55/15: gg = data[data['plot_type'] in [1,4]]
55/16: gg = data[data['plot_type'].isin([1,4])]
55/17: len(gg)
55/18: gg.columns
57/1:
%load_ext autoreload
%autoreload 2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import json
import os
from tqdm import tqdm
import re
import sys
from sklearn.model_selection import train_test_split
import requests
pd.options.display.float_format = '{:,.2f}'.format
# Add path to where utils.py is so metrics can be imported
sys.path.insert(0, r'C:/Users/Teemu/Work/linda-forestry-ml/species_prediction/regressors')

from data import data_loading
57/2:
%load_ext autoreload
%autoreload 2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import json
import os
from tqdm import tqdm
import re
import sys
from sklearn.model_selection import train_test_split
import requests
pd.options.display.float_format = '{:,.2f}'.format
# Add path to where utils.py is so metrics can be imported
sys.path.insert(0, r'/home/tman/linda-forestry-ml/species_prediction/regressors/')

from data import data_loading
57/3:
%load_ext autoreload
%autoreload 2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import json
import os
from tqdm import tqdm
import re
import sys
from sklearn.model_selection import train_test_split
import requests
pd.options.display.float_format = '{:,.2f}'.format
# Add path to where utils.py is so metrics can be imported
sys.path.insert(0, r'/home/tman/Work/linda-forestry-ml/species_prediction/regressors/')

from data import data_loading
57/4:
input_path = r"C:\Users\Teemu\Work\data\FIsampletiles"
labels_name = "test.csv"
image_dir = "azure_tiles_cleaned"
scalar_feature_names = ['easting', 'northing', 'measure_year', 'measure_date', 
                 'elevation', 'slope', 'aspect',  'soil_type',
                 'tree_cover', 'leaf_type', 'plot_id']
prediction_features=['vol_total', 'vol_pine', 'vol_spruce', 'vol_deciduous']

X, df = data_loading.import_data(input_path, labels_name=labels_name, image_dir="tiles")
57/5:
input_path = r"/home/tman/Work/data/FIsampletiles"
labels_name = "test.csv"
image_dir = "azure_tiles_cleaned"
scalar_feature_names = ['easting', 'northing', 'measure_year', 'measure_date', 
                 'elevation', 'slope', 'aspect',  'soil_type',
                 'tree_cover', 'leaf_type', 'plot_id']
prediction_features=['vol_total', 'vol_pine', 'vol_spruce', 'vol_deciduous']

X, df = data_loading.import_data(input_path, labels_name=labels_name, image_dir="tiles")
57/6:
### Get the metsäkeskus hila predictions on the test set and join them on the sample plot ids
### Adapted from lidar-height-and-density-analysis.ipynb notebook

api_url = 'http://51.144.230.13:10323/api/point_list'
locations = df[['easting', 'northing']].values.tolist()
plot_ids = df.plot_id.values.tolist()

# The API currently gives an error for large number of locations, so we must get the data in batches.
batch_size = 1000
data_batches = []
for batch_start in tqdm(range(0, len(locations), batch_size)):
    
    locations_batch = locations[batch_start:batch_start+batch_size]
    plot_id_batch = plot_ids[batch_start:batch_start+batch_size]
    post_json = json.dumps({
            'srid': 3067,
            'coordinates': locations_batch,
            'fids': plot_id_batch
        })
    params = {
            'schema': 'metsakeskus_hila',
            'table': 'gridcell',
            'columns': ['volumepine,volumespruce,volumedeciduous,volume']
        }

    post_headers = {'Content-Type': 'application/json'}

    res = requests.post(api_url, data=post_json, headers=post_headers, params=params)
    data_batch = res.json()
    data_batch = [(feature['properties']['fid'],
                         feature['properties']['volumepine'],feature['properties']['volumespruce'],
                         feature['properties']['volumedeciduous'],feature['properties']['volume']
                        ) 
                        for feature in data_batch['features']]
    
    data_batch = pd.DataFrame(data_batch, columns=['plot_id','volumepine', 'volumespruce', 'volumedeciduous', 'volumetotal'])
    
    data_batches.append(data_batch)
    
data = pd.concat(data_batches, axis=0, ignore_index=True)
data_unique = data.loc[data.plot_id.drop_duplicates().index]
scalar_df = data_unique.merge(df, on='plot_id').drop_duplicates(subset='plot_id')
57/7:
### Cell for running own models

from keras.models import load_model
from features import preprocessing

model_path = r"/home/tman/Work/linda-forestry-ml/species_prediction/regressors/weights/"
model = load_model(model_path)

### Use same preprocessing that the model used
57/8:
### Cell for running own models

from keras.models import load_model
from features import preprocessing

model_path = r"/home/tman/Work/linda-forestry-ml/species_prediction/regressors/weights/mobilenet_mature19-01-22_12-05.hdf5"
model = load_model(model_path)

### Use same preprocessing that the model used
57/9:
### Cell for running own models

from keras.models import load_model
from features import preprocessing

model_path = r"/home/tman/Work/linda-forestry-ml/species_prediction/regressors/weights/mobilenet_mature19-01-22_12-05.hdf5"
model = load_model(model_path)

### Use same preprocessing that the model used
57/10:
### Cell for running own models

from keras.models import load_model
from features import preprocessing
from keras.utils.generic_utils import CustomObjectScope

model_path = r"/home/tman/Work/linda-forestry-ml/species_prediction/regressors/weights/mobilenet_mature19-01-22_12-05.hdf5"
with CustomObjectScope({'relu6': keras.applications.mobilenet.relu6,'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D}):
    model = load_model(model_path)

### Use same preprocessing that the model used
57/11:
### Cell for running own models

import keras
from keras.models import load_model
from features import preprocessing
from keras.utils.generic_utils import CustomObjectScope

model_path = r"/home/tman/Work/linda-forestry-ml/species_prediction/regressors/weights/mobilenet_mature19-01-22_12-05.hdf5"
with CustomObjectScope({'relu6': keras.applications.mobilenet.relu6,'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D}):
    model = load_model(model_path)

### Use same preprocessing that the model used
57/12:
### Cell for running own models

import keras
from keras.models import load_model
from features import preprocessing
from keras.utils.generic_utils import CustomObjectScope
from keras.layers import DepthwiseConv2D
from keras_applications.mobilenet import relu6

model_path = r"/home/tman/Work/linda-forestry-ml/species_prediction/regressors/weights/mobilenet_mature19-01-22_12-05.hdf5"
with CustomObjectScope({'relu6': relu6,'DepthwiseConv2D': DepthwiseConv2D}):
    model = load_model(model_path)

### Use same preprocessing that the model used
58/1:
X_preprocessed = preprocessing.preprocessing_dict['crop_center'](X)
X_preprocessed = preprocessing.preprocessing_dict['resize'](X_preprocessed, input_dims=[128, 128])
59/1:
%load_ext autoreload
%autoreload 2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import json
import os
from tqdm import tqdm
import re
import sys
from sklearn.model_selection import train_test_split
import requests
pd.options.display.float_format = '{:,.2f}'.format
# Add path to where utils.py is so metrics can be imported
sys.path.insert(0, r'/home/tman/Work/linda-forestry-ml/species_prediction/regressors/')

from data import data_loading
59/2:
input_path = r"/home/tman/Work/data/FIsampletiles"
labels_name = "test.csv"
image_dir = "azure_tiles_cleaned"
scalar_feature_names = ['easting', 'northing', 'measure_year', 'measure_date', 
                 'elevation', 'slope', 'aspect',  'soil_type',
                 'tree_cover', 'leaf_type', 'plot_id']
prediction_features=['vol_total', 'vol_pine', 'vol_spruce', 'vol_deciduous']

X, df = data_loading.import_data(input_path, labels_name=labels_name, image_dir="tiles")
59/3:
### Get the metsäkeskus hila predictions on the test set and join them on the sample plot ids
### Adapted from lidar-height-and-density-analysis.ipynb notebook

api_url = 'http://51.144.230.13:10323/api/point_list'
locations = df[['easting', 'northing']].values.tolist()
plot_ids = df.plot_id.values.tolist()

# The API currently gives an error for large number of locations, so we must get the data in batches.
batch_size = 1000
data_batches = []
for batch_start in tqdm(range(0, len(locations), batch_size)):
    
    locations_batch = locations[batch_start:batch_start+batch_size]
    plot_id_batch = plot_ids[batch_start:batch_start+batch_size]
    post_json = json.dumps({
            'srid': 3067,
            'coordinates': locations_batch,
            'fids': plot_id_batch
        })
    params = {
            'schema': 'metsakeskus_hila',
            'table': 'gridcell',
            'columns': ['volumepine,volumespruce,volumedeciduous,volume']
        }

    post_headers = {'Content-Type': 'application/json'}

    res = requests.post(api_url, data=post_json, headers=post_headers, params=params)
    data_batch = res.json()
    data_batch = [(feature['properties']['fid'],
                         feature['properties']['volumepine'],feature['properties']['volumespruce'],
                         feature['properties']['volumedeciduous'],feature['properties']['volume']
                        ) 
                        for feature in data_batch['features']]
    
    data_batch = pd.DataFrame(data_batch, columns=['plot_id','volumepine', 'volumespruce', 'volumedeciduous', 'volumetotal'])
    
    data_batches.append(data_batch)
    
data = pd.concat(data_batches, axis=0, ignore_index=True)
data_unique = data.loc[data.plot_id.drop_duplicates().index]
scalar_df = data_unique.merge(df, on='plot_id').drop_duplicates(subset='plot_id')
59/4:
### Cell for running own models

import keras
from keras.models import load_model
from features import preprocessing
from keras.utils.generic_utils import CustomObjectScope
from keras.layers import DepthwiseConv2D
from keras_applications.mobilenet import relu6

model_path = r"/home/tman/Work/linda-forestry-ml/species_prediction/regressors/weights/mobilenet_mature19-01-22_12-05.hdf5"
with CustomObjectScope({'relu6': relu6,'DepthwiseConv2D': DepthwiseConv2D}):
    model = load_model(model_path)

### Use same preprocessing that the model used
61/1:
%load_ext autoreload
%autoreload 2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import json
import os
from tqdm import tqdm
import re
import sys
from sklearn.model_selection import train_test_split
import requests
pd.options.display.float_format = '{:,.2f}'.format
# Add path to where utils.py is so metrics can be imported
sys.path.insert(0, r'/home/tman/Work/linda-forestry-ml/species_prediction/regressors/')

from data import data_loading
61/2:
input_path = r"/home/tman/Work/data/FIsampletiles"
labels_name = "test.csv"
image_dir = "azure_tiles_cleaned"
scalar_feature_names = ['easting', 'northing', 'measure_year', 'measure_date', 
                 'elevation', 'slope', 'aspect',  'soil_type',
                 'tree_cover', 'leaf_type', 'plot_id']
prediction_features=['vol_total', 'vol_pine', 'vol_spruce', 'vol_deciduous']

X, df = data_loading.import_data(input_path, labels_name=labels_name, image_dir="tiles")
61/3:
### Get the metsäkeskus hila predictions on the test set and join them on the sample plot ids
### Adapted from lidar-height-and-density-analysis.ipynb notebook

api_url = 'http://51.144.230.13:10323/api/point_list'
locations = df[['easting', 'northing']].values.tolist()
plot_ids = df.plot_id.values.tolist()

# The API currently gives an error for large number of locations, so we must get the data in batches.
batch_size = 1000
data_batches = []
for batch_start in tqdm(range(0, len(locations), batch_size)):
    
    locations_batch = locations[batch_start:batch_start+batch_size]
    plot_id_batch = plot_ids[batch_start:batch_start+batch_size]
    post_json = json.dumps({
            'srid': 3067,
            'coordinates': locations_batch,
            'fids': plot_id_batch
        })
    params = {
            'schema': 'metsakeskus_hila',
            'table': 'gridcell',
            'columns': ['volumepine,volumespruce,volumedeciduous,volume']
        }

    post_headers = {'Content-Type': 'application/json'}

    res = requests.post(api_url, data=post_json, headers=post_headers, params=params)
    data_batch = res.json()
    data_batch = [(feature['properties']['fid'],
                         feature['properties']['volumepine'],feature['properties']['volumespruce'],
                         feature['properties']['volumedeciduous'],feature['properties']['volume']
                        ) 
                        for feature in data_batch['features']]
    
    data_batch = pd.DataFrame(data_batch, columns=['plot_id','volumepine', 'volumespruce', 'volumedeciduous', 'volumetotal'])
    
    data_batches.append(data_batch)
    
data = pd.concat(data_batches, axis=0, ignore_index=True)
data_unique = data.loc[data.plot_id.drop_duplicates().index]
scalar_df = data_unique.merge(df, on='plot_id').drop_duplicates(subset='plot_id')
61/4:
### Cell for running own models

import keras
from keras.models import load_model
from features import preprocessing
from keras.utils.generic_utils import CustomObjectScope
from keras.layers import DepthwiseConv2D
from keras_applications.mobilenet import relu6

model_path = r"/home/tman/Work/linda-forestry-ml/species_prediction/regressors/weights/mobilenet_mature19-01-22_12-05.hdf5"
with CustomObjectScope({'relu6': relu6,'DepthwiseConv2D': DepthwiseConv2D}):
    model = load_model(model_path)

### Use same preprocessing that the model used
63/1:
%load_ext autoreload
%autoreload 2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import json
import os
from tqdm import tqdm
import re
import sys
from sklearn.model_selection import train_test_split
import requests
pd.options.display.float_format = '{:,.2f}'.format
# Add path to where utils.py is so metrics can be imported
sys.path.insert(0, r'/home/tman/Work/linda-forestry-ml/species_prediction/regressors/')

from data import data_loading
63/2:
input_path = r"/home/tman/Work/data/FIsampletiles"
labels_name = "test.csv"
image_dir = "azure_tiles_cleaned"
scalar_feature_names = ['easting', 'northing', 'measure_year', 'measure_date', 
                 'elevation', 'slope', 'aspect',  'soil_type',
                 'tree_cover', 'leaf_type', 'plot_id']
prediction_features=['vol_total', 'vol_pine', 'vol_spruce', 'vol_deciduous']

X, df = data_loading.import_data(input_path, labels_name=labels_name, image_dir="tiles")
63/3:
### Get the metsäkeskus hila predictions on the test set and join them on the sample plot ids
### Adapted from lidar-height-and-density-analysis.ipynb notebook

api_url = 'http://51.144.230.13:10323/api/point_list'
locations = df[['easting', 'northing']].values.tolist()
plot_ids = df.plot_id.values.tolist()

# The API currently gives an error for large number of locations, so we must get the data in batches.
batch_size = 1000
data_batches = []
for batch_start in tqdm(range(0, len(locations), batch_size)):
    
    locations_batch = locations[batch_start:batch_start+batch_size]
    plot_id_batch = plot_ids[batch_start:batch_start+batch_size]
    post_json = json.dumps({
            'srid': 3067,
            'coordinates': locations_batch,
            'fids': plot_id_batch
        })
    params = {
            'schema': 'metsakeskus_hila',
            'table': 'gridcell',
            'columns': ['volumepine,volumespruce,volumedeciduous,volume']
        }

    post_headers = {'Content-Type': 'application/json'}

    res = requests.post(api_url, data=post_json, headers=post_headers, params=params)
    data_batch = res.json()
    data_batch = [(feature['properties']['fid'],
                         feature['properties']['volumepine'],feature['properties']['volumespruce'],
                         feature['properties']['volumedeciduous'],feature['properties']['volume']
                        ) 
                        for feature in data_batch['features']]
    
    data_batch = pd.DataFrame(data_batch, columns=['plot_id','volumepine', 'volumespruce', 'volumedeciduous', 'volumetotal'])
    
    data_batches.append(data_batch)
    
data = pd.concat(data_batches, axis=0, ignore_index=True)
data_unique = data.loc[data.plot_id.drop_duplicates().index]
scalar_df = data_unique.merge(df, on='plot_id').drop_duplicates(subset='plot_id')
63/4:
### Cell for running own models

import keras
from keras.models import load_model
from features import preprocessing
from keras.utils.generic_utils import CustomObjectScope
from keras.layers import DepthwiseConv2D
from keras_applications.mobilenet import relu6

model_path = r"/home/tman/Work/linda-forestry-ml/species_prediction/regressors/weights/mobilenet_mature19-01-22_12-05.hdf5"
with CustomObjectScope({'relu6': relu6,'DepthwiseConv2D': DepthwiseConv2D}):
    model = load_model(model_path)

### Use same preprocessing that the model used
63/5:
metsakeskus_predictions = scalar_df[['volumepine', 'volumespruce', 'volumedeciduous', 'volumetotal']]
groundtruth = scalar_df[['vol_pine', 'vol_spruce', 'vol_deciduous', 'vol_total']]
63/6:
X_preprocessed = preprocessing.crop_center(X)
# X_preprocessed = preprocessing.resize_images(X_preprocessed, input_dims=[128, 128])
65/1:
%load_ext autoreload
%autoreload 2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import json
import os
from tqdm import tqdm
import re
import sys
from sklearn.model_selection import train_test_split
import requests
pd.options.display.float_format = '{:,.2f}'.format
# Add path to where utils.py is so metrics can be imported
sys.path.insert(0, r'/home/tman/Work/linda-forestry-ml/species_prediction/regressors/')

from data import data_loading
65/2:
input_path = r"/home/tman/Work/data/FIsampletiles"
labels_name = "test.csv"
image_dir = "azure_tiles_cleaned"
scalar_feature_names = ['easting', 'northing', 'measure_year', 'measure_date', 
                 'elevation', 'slope', 'aspect',  'soil_type',
                 'tree_cover', 'leaf_type', 'plot_id']
prediction_features=['vol_total', 'vol_pine', 'vol_spruce', 'vol_deciduous']

X, df = data_loading.import_data(input_path, labels_name=labels_name, image_dir="tiles")
65/3:
### Get the metsäkeskus hila predictions on the test set and join them on the sample plot ids
### Adapted from lidar-height-and-density-analysis.ipynb notebook

api_url = 'http://51.144.230.13:10323/api/point_list'
locations = df[['easting', 'northing']].values.tolist()
plot_ids = df.plot_id.values.tolist()

# The API currently gives an error for large number of locations, so we must get the data in batches.
batch_size = 1000
data_batches = []
for batch_start in tqdm(range(0, len(locations), batch_size)):
    
    locations_batch = locations[batch_start:batch_start+batch_size]
    plot_id_batch = plot_ids[batch_start:batch_start+batch_size]
    post_json = json.dumps({
            'srid': 3067,
            'coordinates': locations_batch,
            'fids': plot_id_batch
        })
    params = {
            'schema': 'metsakeskus_hila',
            'table': 'gridcell',
            'columns': ['volumepine,volumespruce,volumedeciduous,volume']
        }

    post_headers = {'Content-Type': 'application/json'}

    res = requests.post(api_url, data=post_json, headers=post_headers, params=params)
    data_batch = res.json()
    data_batch = [(feature['properties']['fid'],
                         feature['properties']['volumepine'],feature['properties']['volumespruce'],
                         feature['properties']['volumedeciduous'],feature['properties']['volume']
                        ) 
                        for feature in data_batch['features']]
    
    data_batch = pd.DataFrame(data_batch, columns=['plot_id','volumepine', 'volumespruce', 'volumedeciduous', 'volumetotal'])
    
    data_batches.append(data_batch)
    
data = pd.concat(data_batches, axis=0, ignore_index=True)
data_unique = data.loc[data.plot_id.drop_duplicates().index]
scalar_df = data_unique.merge(df, on='plot_id').drop_duplicates(subset='plot_id')
65/4:
### Cell for running own models

import keras
from keras.models import load_model
from features import preprocessing
from keras.utils.generic_utils import CustomObjectScope
from keras.layers import DepthwiseConv2D
from keras_applications.mobilenet import relu6

model_path = r"/home/tman/Work/linda-forestry-ml/species_prediction/regressors/weights/mobilenet_mature19-01-22_12-05.hdf5"
with CustomObjectScope({'relu6': relu6,'DepthwiseConv2D': DepthwiseConv2D}):
    model = load_model(model_path)

### Use same preprocessing that the model used
66/1:
%load_ext autoreload
%autoreload 2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import json
import os
from tqdm import tqdm
import re
import sys
from sklearn.model_selection import train_test_split
import requests
pd.options.display.float_format = '{:,.2f}'.format
# Add path to where utils.py is so metrics can be imported
sys.path.insert(0, r'/home/tman/Work/linda-forestry-ml/species_prediction/regressors/')

from data import data_loading
66/2:
input_path = r"/home/tman/Work/data/FIsampletiles"
labels_name = "test.csv"
image_dir = "azure_tiles_cleaned"
scalar_feature_names = ['easting', 'northing', 'measure_year', 'measure_date', 
                 'elevation', 'slope', 'aspect',  'soil_type',
                 'tree_cover', 'leaf_type', 'plot_id']
prediction_features=['vol_total', 'vol_pine', 'vol_spruce', 'vol_deciduous']

X, df = data_loading.import_data(input_path, labels_name=labels_name, image_dir="tiles")
66/3:
### Get the metsäkeskus hila predictions on the test set and join them on the sample plot ids
### Adapted from lidar-height-and-density-analysis.ipynb notebook

api_url = 'http://51.144.230.13:10323/api/point_list'
locations = df[['easting', 'northing']].values.tolist()
plot_ids = df.plot_id.values.tolist()

# The API currently gives an error for large number of locations, so we must get the data in batches.
batch_size = 1000
data_batches = []
for batch_start in tqdm(range(0, len(locations), batch_size)):
    
    locations_batch = locations[batch_start:batch_start+batch_size]
    plot_id_batch = plot_ids[batch_start:batch_start+batch_size]
    post_json = json.dumps({
            'srid': 3067,
            'coordinates': locations_batch,
            'fids': plot_id_batch
        })
    params = {
            'schema': 'metsakeskus_hila',
            'table': 'gridcell',
            'columns': ['volumepine,volumespruce,volumedeciduous,volume']
        }

    post_headers = {'Content-Type': 'application/json'}

    res = requests.post(api_url, data=post_json, headers=post_headers, params=params)
    data_batch = res.json()
    data_batch = [(feature['properties']['fid'],
                         feature['properties']['volumepine'],feature['properties']['volumespruce'],
                         feature['properties']['volumedeciduous'],feature['properties']['volume']
                        ) 
                        for feature in data_batch['features']]
    
    data_batch = pd.DataFrame(data_batch, columns=['plot_id','volumepine', 'volumespruce', 'volumedeciduous', 'volumetotal'])
    
    data_batches.append(data_batch)
    
data = pd.concat(data_batches, axis=0, ignore_index=True)
data_unique = data.loc[data.plot_id.drop_duplicates().index]
scalar_df = data_unique.merge(df, on='plot_id').drop_duplicates(subset='plot_id')
66/4:
from features import preprocessing
X_preprocessed = preprocessing.crop_center(X)
X_preprocessed = preprocessing.resize_images(X_preprocessed, input_dims=[128, 128])
67/1:
%load_ext autoreload
%autoreload 2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import json
import os
from tqdm import tqdm
import re
import sys
from sklearn.model_selection import train_test_split
import requests
pd.options.display.float_format = '{:,.2f}'.format
# Add path to where utils.py is so metrics can be imported
sys.path.insert(0, r'/home/tman/Work/linda-forestry-ml/species_prediction/regressors/')

from data import data_loading
67/2:
input_path = r"/home/tman/Work/data/FIsampletiles"
labels_name = "test.csv"
image_dir = "azure_tiles_cleaned"
scalar_feature_names = ['easting', 'northing', 'measure_year', 'measure_date', 
                 'elevation', 'slope', 'aspect',  'soil_type',
                 'tree_cover', 'leaf_type', 'plot_id']
prediction_features=['vol_total', 'vol_pine', 'vol_spruce', 'vol_deciduous']

X, df = data_loading.import_data(input_path, labels_name=labels_name, image_dir="tiles")
67/3:
### Get the metsäkeskus hila predictions on the test set and join them on the sample plot ids
### Adapted from lidar-height-and-density-analysis.ipynb notebook

api_url = 'http://51.144.230.13:10323/api/point_list'
locations = df[['easting', 'northing']].values.tolist()
plot_ids = df.plot_id.values.tolist()

# The API currently gives an error for large number of locations, so we must get the data in batches.
batch_size = 1000
data_batches = []
for batch_start in tqdm(range(0, len(locations), batch_size)):
    
    locations_batch = locations[batch_start:batch_start+batch_size]
    plot_id_batch = plot_ids[batch_start:batch_start+batch_size]
    post_json = json.dumps({
            'srid': 3067,
            'coordinates': locations_batch,
            'fids': plot_id_batch
        })
    params = {
            'schema': 'metsakeskus_hila',
            'table': 'gridcell',
            'columns': ['volumepine,volumespruce,volumedeciduous,volume']
        }

    post_headers = {'Content-Type': 'application/json'}

    res = requests.post(api_url, data=post_json, headers=post_headers, params=params)
    data_batch = res.json()
    data_batch = [(feature['properties']['fid'],
                         feature['properties']['volumepine'],feature['properties']['volumespruce'],
                         feature['properties']['volumedeciduous'],feature['properties']['volume']
                        ) 
                        for feature in data_batch['features']]
    
    data_batch = pd.DataFrame(data_batch, columns=['plot_id','volumepine', 'volumespruce', 'volumedeciduous', 'volumetotal'])
    
    data_batches.append(data_batch)
    
data = pd.concat(data_batches, axis=0, ignore_index=True)
data_unique = data.loc[data.plot_id.drop_duplicates().index]
scalar_df = data_unique.merge(df, on='plot_id').drop_duplicates(subset='plot_id')
67/4:
from features import preprocessing
X_preprocessed = preprocessing.crop_center(X)
X_preprocessed = preprocessing.resize_images(X_preprocessed, input_dims=[128, 128])
69/1:
%load_ext autoreload
%autoreload 2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import json
import os
from tqdm import tqdm
import re
import sys
from sklearn.model_selection import train_test_split
import requests
pd.options.display.float_format = '{:,.2f}'.format
# Add path to where utils.py is so metrics can be imported
sys.path.insert(0, r'/home/tman/Work/linda-forestry-ml/species_prediction/regressors/')

from data import data_loading
69/2:
input_path = r"/home/tman/Work/data/FIsampletiles"
labels_name = "test.csv"
image_dir = "azure_tiles_cleaned"
scalar_feature_names = ['easting', 'northing', 'measure_year', 'measure_date', 
                 'elevation', 'slope', 'aspect',  'soil_type',
                 'tree_cover', 'leaf_type', 'plot_id']
prediction_features=['vol_total', 'vol_pine', 'vol_spruce', 'vol_deciduous']

X, df = data_loading.import_data(input_path, labels_name=labels_name, image_dir="tiles")
69/3:
### Get the metsäkeskus hila predictions on the test set and join them on the sample plot ids
### Adapted from lidar-height-and-density-analysis.ipynb notebook

api_url = 'http://51.144.230.13:10323/api/point_list'
locations = df[['easting', 'northing']].values.tolist()
plot_ids = df.plot_id.values.tolist()

# The API currently gives an error for large number of locations, so we must get the data in batches.
batch_size = 1000
data_batches = []
for batch_start in tqdm(range(0, len(locations), batch_size)):
    
    locations_batch = locations[batch_start:batch_start+batch_size]
    plot_id_batch = plot_ids[batch_start:batch_start+batch_size]
    post_json = json.dumps({
            'srid': 3067,
            'coordinates': locations_batch,
            'fids': plot_id_batch
        })
    params = {
            'schema': 'metsakeskus_hila',
            'table': 'gridcell',
            'columns': ['volumepine,volumespruce,volumedeciduous,volume']
        }

    post_headers = {'Content-Type': 'application/json'}

    res = requests.post(api_url, data=post_json, headers=post_headers, params=params)
    data_batch = res.json()
    data_batch = [(feature['properties']['fid'],
                         feature['properties']['volumepine'],feature['properties']['volumespruce'],
                         feature['properties']['volumedeciduous'],feature['properties']['volume']
                        ) 
                        for feature in data_batch['features']]
    
    data_batch = pd.DataFrame(data_batch, columns=['plot_id','volumepine', 'volumespruce', 'volumedeciduous', 'volumetotal'])
    
    data_batches.append(data_batch)
    
data = pd.concat(data_batches, axis=0, ignore_index=True)
data_unique = data.loc[data.plot_id.drop_duplicates().index]
scalar_df = data_unique.merge(df, on='plot_id').drop_duplicates(subset='plot_id')
69/4:
from features import preprocessing
X_preprocessed = preprocessing.crop_center(X)
X_preprocessed = preprocessing.resize_images(X_preprocessed, input_dims=[128, 128, 3])
71/1:
%load_ext autoreload
%autoreload 2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import json
import os
from tqdm import tqdm
import re
import sys
from sklearn.model_selection import train_test_split
import requests
pd.options.display.float_format = '{:,.2f}'.format
# Add path to where utils.py is so metrics can be imported
sys.path.insert(0, r'/home/tman/Work/linda-forestry-ml/species_prediction/regressors/')

from data import data_loading
71/2:
input_path = r"/home/tman/Work/data/FIsampletiles"
labels_name = "test.csv"
image_dir = "azure_tiles_cleaned"
scalar_feature_names = ['easting', 'northing', 'measure_year', 'measure_date', 
                 'elevation', 'slope', 'aspect',  'soil_type',
                 'tree_cover', 'leaf_type', 'plot_id']
prediction_features=['vol_total', 'vol_pine', 'vol_spruce', 'vol_deciduous']

X, df = data_loading.import_data(input_path, labels_name=labels_name, image_dir="tiles")
71/3:
### Get the metsäkeskus hila predictions on the test set and join them on the sample plot ids
### Adapted from lidar-height-and-density-analysis.ipynb notebook

api_url = 'http://51.144.230.13:10323/api/point_list'
locations = df[['easting', 'northing']].values.tolist()
plot_ids = df.plot_id.values.tolist()

# The API currently gives an error for large number of locations, so we must get the data in batches.
batch_size = 1000
data_batches = []
for batch_start in tqdm(range(0, len(locations), batch_size)):
    
    locations_batch = locations[batch_start:batch_start+batch_size]
    plot_id_batch = plot_ids[batch_start:batch_start+batch_size]
    post_json = json.dumps({
            'srid': 3067,
            'coordinates': locations_batch,
            'fids': plot_id_batch
        })
    params = {
            'schema': 'metsakeskus_hila',
            'table': 'gridcell',
            'columns': ['volumepine,volumespruce,volumedeciduous,volume']
        }

    post_headers = {'Content-Type': 'application/json'}

    res = requests.post(api_url, data=post_json, headers=post_headers, params=params)
    data_batch = res.json()
    data_batch = [(feature['properties']['fid'],
                         feature['properties']['volumepine'],feature['properties']['volumespruce'],
                         feature['properties']['volumedeciduous'],feature['properties']['volume']
                        ) 
                        for feature in data_batch['features']]
    
    data_batch = pd.DataFrame(data_batch, columns=['plot_id','volumepine', 'volumespruce', 'volumedeciduous', 'volumetotal'])
    
    data_batches.append(data_batch)
    
data = pd.concat(data_batches, axis=0, ignore_index=True)
data_unique = data.loc[data.plot_id.drop_duplicates().index]
scalar_df = data_unique.merge(df, on='plot_id').drop_duplicates(subset='plot_id')
71/4:
from features import preprocessing
X_preprocessed = preprocessing.crop_center(X)
X_preprocessed = np.array([cv2.resize(image, [128, 128]) for image in X])
71/5:
from features import preprocessing
X_preprocessed = preprocessing.crop_center(X)
X_preprocessed = np.array([cv2.resize(image, (128, 128)) for image in X])
72/1:
%load_ext autoreload
%autoreload 2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import json
import os
from tqdm import tqdm
import re
import sys
from sklearn.model_selection import train_test_split
import requests
pd.options.display.float_format = '{:,.2f}'.format
# Add path to where utils.py is so metrics can be imported
sys.path.insert(0, r'/home/tman/Work/linda-forestry-ml/species_prediction/regressors/')

from data import data_loading
72/2:
input_path = r"/home/tman/Work/data/FIsampletiles"
labels_name = "test.csv"
image_dir = "azure_tiles_cleaned"
scalar_feature_names = ['easting', 'northing', 'measure_year', 'measure_date', 
                 'elevation', 'slope', 'aspect',  'soil_type',
                 'tree_cover', 'leaf_type', 'plot_id']
prediction_features=['vol_total', 'vol_pine', 'vol_spruce', 'vol_deciduous']

X, df = data_loading.import_data(input_path, labels_name=labels_name, image_dir="tiles")
72/3:
### Get the metsäkeskus hila predictions on the test set and join them on the sample plot ids
### Adapted from lidar-height-and-density-analysis.ipynb notebook

api_url = 'http://51.144.230.13:10323/api/point_list'
locations = df[['easting', 'northing']].values.tolist()
plot_ids = df.plot_id.values.tolist()

# The API currently gives an error for large number of locations, so we must get the data in batches.
batch_size = 1000
data_batches = []
for batch_start in tqdm(range(0, len(locations), batch_size)):
    
    locations_batch = locations[batch_start:batch_start+batch_size]
    plot_id_batch = plot_ids[batch_start:batch_start+batch_size]
    post_json = json.dumps({
            'srid': 3067,
            'coordinates': locations_batch,
            'fids': plot_id_batch
        })
    params = {
            'schema': 'metsakeskus_hila',
            'table': 'gridcell',
            'columns': ['volumepine,volumespruce,volumedeciduous,volume']
        }

    post_headers = {'Content-Type': 'application/json'}

    res = requests.post(api_url, data=post_json, headers=post_headers, params=params)
    data_batch = res.json()
    data_batch = [(feature['properties']['fid'],
                         feature['properties']['volumepine'],feature['properties']['volumespruce'],
                         feature['properties']['volumedeciduous'],feature['properties']['volume']
                        ) 
                        for feature in data_batch['features']]
    
    data_batch = pd.DataFrame(data_batch, columns=['plot_id','volumepine', 'volumespruce', 'volumedeciduous', 'volumetotal'])
    
    data_batches.append(data_batch)
    
data = pd.concat(data_batches, axis=0, ignore_index=True)
data_unique = data.loc[data.plot_id.drop_duplicates().index]
scalar_df = data_unique.merge(df, on='plot_id').drop_duplicates(subset='plot_id')
72/4:
from features import preprocessing
X_preprocessed = preprocessing.crop_center(X)
X_preprocessed = preprocessing.resize_images(X_preprocessed, input_dims=(128, 128))
72/5: X_preprocessed.shapoe
72/6: X_preprocessed.shape
73/1:
%load_ext autoreload
%autoreload 2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import json
import os
from tqdm import tqdm
import re
import sys
from sklearn.model_selection import train_test_split
import requests
pd.options.display.float_format = '{:,.2f}'.format
# Add path to where utils.py is so metrics can be imported
sys.path.insert(0, r'/home/tman/Work/linda-forestry-ml/species_prediction/regressors/')

from data import data_loading
73/2:
input_path = r"/home/tman/Work/data/FIsampletiles"
labels_name = "test.csv"
image_dir = "azure_tiles_cleaned"
scalar_feature_names = ['easting', 'northing', 'measure_year', 'measure_date', 
                 'elevation', 'slope', 'aspect',  'soil_type',
                 'tree_cover', 'leaf_type', 'plot_id']
prediction_features=['vol_total', 'vol_pine', 'vol_spruce', 'vol_deciduous']

X, df = data_loading.import_data(input_path, labels_name=labels_name, image_dir="tiles")
73/3:
### Get the metsäkeskus hila predictions on the test set and join them on the sample plot ids
### Adapted from lidar-height-and-density-analysis.ipynb notebook

api_url = 'http://51.144.230.13:10323/api/point_list'
locations = df[['easting', 'northing']].values.tolist()
plot_ids = df.plot_id.values.tolist()

# The API currently gives an error for large number of locations, so we must get the data in batches.
batch_size = 1000
data_batches = []
for batch_start in tqdm(range(0, len(locations), batch_size)):
    
    locations_batch = locations[batch_start:batch_start+batch_size]
    plot_id_batch = plot_ids[batch_start:batch_start+batch_size]
    post_json = json.dumps({
            'srid': 3067,
            'coordinates': locations_batch,
            'fids': plot_id_batch
        })
    params = {
            'schema': 'metsakeskus_hila',
            'table': 'gridcell',
            'columns': ['volumepine,volumespruce,volumedeciduous,volume']
        }

    post_headers = {'Content-Type': 'application/json'}

    res = requests.post(api_url, data=post_json, headers=post_headers, params=params)
    data_batch = res.json()
    data_batch = [(feature['properties']['fid'],
                         feature['properties']['volumepine'],feature['properties']['volumespruce'],
                         feature['properties']['volumedeciduous'],feature['properties']['volume']
                        ) 
                        for feature in data_batch['features']]
    
    data_batch = pd.DataFrame(data_batch, columns=['plot_id','volumepine', 'volumespruce', 'volumedeciduous', 'volumetotal'])
    
    data_batches.append(data_batch)
    
data = pd.concat(data_batches, axis=0, ignore_index=True)
data_unique = data.loc[data.plot_id.drop_duplicates().index]
scalar_df = data_unique.merge(df, on='plot_id').drop_duplicates(subset='plot_id')
73/4:
from features import preprocessing
X_preprocessed, _ = preprocessing.crop_center(X)
X_preprocessed, _ = preprocessing.resize_images(X_preprocessed, input_dims=(128, 128))
74/1:
%load_ext autoreload
%autoreload 2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import json
import os
from tqdm import tqdm
import re
import sys
from sklearn.model_selection import train_test_split
import requests
pd.options.display.float_format = '{:,.2f}'.format
# Add path to where utils.py is so metrics can be imported
sys.path.insert(0, r'/home/tman/Work/linda-forestry-ml/species_prediction/regressors/')

from data import data_loading
74/2:
input_path = r"/home/tman/Work/data/FIsampletiles"
labels_name = "test.csv"
image_dir = "azure_tiles_cleaned"
scalar_feature_names = ['easting', 'northing', 'measure_year', 'measure_date', 
                 'elevation', 'slope', 'aspect',  'soil_type',
                 'tree_cover', 'leaf_type', 'plot_id']
prediction_features=['vol_total', 'vol_pine', 'vol_spruce', 'vol_deciduous']

X, df = data_loading.import_data(input_path, labels_name=labels_name, image_dir="tiles")
74/3:
### Get the metsäkeskus hila predictions on the test set and join them on the sample plot ids
### Adapted from lidar-height-and-density-analysis.ipynb notebook

api_url = 'http://51.144.230.13:10323/api/point_list'
locations = df[['easting', 'northing']].values.tolist()
plot_ids = df.plot_id.values.tolist()

# The API currently gives an error for large number of locations, so we must get the data in batches.
batch_size = 1000
data_batches = []
for batch_start in tqdm(range(0, len(locations), batch_size)):
    
    locations_batch = locations[batch_start:batch_start+batch_size]
    plot_id_batch = plot_ids[batch_start:batch_start+batch_size]
    post_json = json.dumps({
            'srid': 3067,
            'coordinates': locations_batch,
            'fids': plot_id_batch
        })
    params = {
            'schema': 'metsakeskus_hila',
            'table': 'gridcell',
            'columns': ['volumepine,volumespruce,volumedeciduous,volume']
        }

    post_headers = {'Content-Type': 'application/json'}

    res = requests.post(api_url, data=post_json, headers=post_headers, params=params)
    data_batch = res.json()
    data_batch = [(feature['properties']['fid'],
                         feature['properties']['volumepine'],feature['properties']['volumespruce'],
                         feature['properties']['volumedeciduous'],feature['properties']['volume']
                        ) 
                        for feature in data_batch['features']]
    
    data_batch = pd.DataFrame(data_batch, columns=['plot_id','volumepine', 'volumespruce', 'volumedeciduous', 'volumetotal'])
    
    data_batches.append(data_batch)
    
data = pd.concat(data_batches, axis=0, ignore_index=True)
data_unique = data.loc[data.plot_id.drop_duplicates().index]
scalar_df = data_unique.merge(df, on='plot_id').drop_duplicates(subset='plot_id')
74/4:
from features import preprocessing
X_preprocessed, X_scalar, y, ids, y_clf = preprocessing.crop_center(X)
X_preprocessed, X_scalar, y, ids, y_clf = preprocessing.resize_images(X_preprocessed, input_dims=(128, 128))
75/1:
%load_ext autoreload
%autoreload 2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import json
import os
from tqdm import tqdm
import re
import sys
from sklearn.model_selection import train_test_split
import requests
pd.options.display.float_format = '{:,.2f}'.format
# Add path to where utils.py is so metrics can be imported
sys.path.insert(0, r'/home/tman/Work/linda-forestry-ml/species_prediction/regressors/')

from data import data_loading
75/2:
input_path = r"/home/tman/Work/data/FIsampletiles"
labels_name = "test.csv"
image_dir = "azure_tiles_cleaned"
scalar_feature_names = ['easting', 'northing', 'measure_year', 'measure_date', 
                 'elevation', 'slope', 'aspect',  'soil_type',
                 'tree_cover', 'leaf_type', 'plot_id']
prediction_features=['vol_total', 'vol_pine', 'vol_spruce', 'vol_deciduous']

X, df = data_loading.import_data(input_path, labels_name=labels_name, image_dir="tiles")
75/3:
### Get the metsäkeskus hila predictions on the test set and join them on the sample plot ids
### Adapted from lidar-height-and-density-analysis.ipynb notebook

api_url = 'http://51.144.230.13:10323/api/point_list'
locations = df[['easting', 'northing']].values.tolist()
plot_ids = df.plot_id.values.tolist()

# The API currently gives an error for large number of locations, so we must get the data in batches.
batch_size = 1000
data_batches = []
for batch_start in tqdm(range(0, len(locations), batch_size)):
    
    locations_batch = locations[batch_start:batch_start+batch_size]
    plot_id_batch = plot_ids[batch_start:batch_start+batch_size]
    post_json = json.dumps({
            'srid': 3067,
            'coordinates': locations_batch,
            'fids': plot_id_batch
        })
    params = {
            'schema': 'metsakeskus_hila',
            'table': 'gridcell',
            'columns': ['volumepine,volumespruce,volumedeciduous,volume']
        }

    post_headers = {'Content-Type': 'application/json'}

    res = requests.post(api_url, data=post_json, headers=post_headers, params=params)
    data_batch = res.json()
    data_batch = [(feature['properties']['fid'],
                         feature['properties']['volumepine'],feature['properties']['volumespruce'],
                         feature['properties']['volumedeciduous'],feature['properties']['volume']
                        ) 
                        for feature in data_batch['features']]
    
    data_batch = pd.DataFrame(data_batch, columns=['plot_id','volumepine', 'volumespruce', 'volumedeciduous', 'volumetotal'])
    
    data_batches.append(data_batch)
    
data = pd.concat(data_batches, axis=0, ignore_index=True)
data_unique = data.loc[data.plot_id.drop_duplicates().index]
scalar_df = data_unique.merge(df, on='plot_id').drop_duplicates(subset='plot_id')
75/4:
from features import preprocessing
X_preprocessed = preprocessing.crop_center(X)[0]
X_preprocessed = preprocessing.resize_images(X_preprocessed, input_dims=(128, 128))[0]
75/5: X_preprocessed.shape
76/1:
%load_ext autoreload
%autoreload 2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import json
import os
from tqdm import tqdm
import re
import sys
from sklearn.model_selection import train_test_split
import requests
pd.options.display.float_format = '{:,.2f}'.format
# Add path to where utils.py is so metrics can be imported
sys.path.insert(0, r'/home/tman/Work/linda-forestry-ml/species_prediction/regressors/')

from data import data_loading
76/2:
input_path = r"/home/tman/Work/data/FIsampletiles"
labels_name = "test.csv"
image_dir = "azure_tiles_cleaned"
scalar_feature_names = ['easting', 'northing', 'measure_year', 'measure_date', 
                 'elevation', 'slope', 'aspect',  'soil_type',
                 'tree_cover', 'leaf_type', 'plot_id']
prediction_features=['vol_total', 'vol_pine', 'vol_spruce', 'vol_deciduous']

X, df = data_loading.import_data(input_path, labels_name=labels_name, image_dir=image_dir)
76/3:
### Get the metsäkeskus hila predictions on the test set and join them on the sample plot ids
### Adapted from lidar-height-and-density-analysis.ipynb notebook

api_url = 'http://51.144.230.13:10323/api/point_list'
locations = df[['easting', 'northing']].values.tolist()
plot_ids = df.plot_id.values.tolist()

# The API currently gives an error for large number of locations, so we must get the data in batches.
batch_size = 1000
data_batches = []
for batch_start in tqdm(range(0, len(locations), batch_size)):
    
    locations_batch = locations[batch_start:batch_start+batch_size]
    plot_id_batch = plot_ids[batch_start:batch_start+batch_size]
    post_json = json.dumps({
            'srid': 3067,
            'coordinates': locations_batch,
            'fids': plot_id_batch
        })
    params = {
            'schema': 'metsakeskus_hila',
            'table': 'gridcell',
            'columns': ['volumepine,volumespruce,volumedeciduous,volume']
        }

    post_headers = {'Content-Type': 'application/json'}

    res = requests.post(api_url, data=post_json, headers=post_headers, params=params)
    data_batch = res.json()
    data_batch = [(feature['properties']['fid'],
                         feature['properties']['volumepine'],feature['properties']['volumespruce'],
                         feature['properties']['volumedeciduous'],feature['properties']['volume']
                        ) 
                        for feature in data_batch['features']]
    
    data_batch = pd.DataFrame(data_batch, columns=['plot_id','volumepine', 'volumespruce', 'volumedeciduous', 'volumetotal'])
    
    data_batches.append(data_batch)
    
data = pd.concat(data_batches, axis=0, ignore_index=True)
data_unique = data.loc[data.plot_id.drop_duplicates().index]
scalar_df = data_unique.merge(df, on='plot_id').drop_duplicates(subset='plot_id')
76/4:
input_path = r"/home/tman/Work/data/FIsampletiles"
labels_name = "test.csv"
image_dir = "azure_tiles_cleaned"
scalar_feature_names = ['easting', 'northing', 'measure_year', 'measure_date', 
                 'elevation', 'slope', 'aspect',  'soil_type',
                 'tree_cover', 'leaf_type', 'plot_id']
prediction_features=['vol_total', 'vol_pine', 'vol_spruce', 'vol_deciduous']

X, df = data_loading.import_data(input_path, labels_name=labels_name, image_dir=image_dir)
76/5:
input_path = r"/home/tman/Work/data/FIsampletiles"
labels_name = "test.csv"
image_dir = "azure_tiles_cleaned"
scalar_feature_names = ['easting', 'northing', 'measure_year', 'measure_date', 
                 'elevation', 'slope', 'aspect',  'soil_type',
                 'tree_cover', 'leaf_type', 'plot_id']
prediction_features=['vol_total', 'vol_pine', 'vol_spruce', 'vol_deciduous']

X, df = data_loading.import_data(input_path, labels_name=labels_name, image_dir=image_dir, image_type="jpg")
76/6:
### Get the metsäkeskus hila predictions on the test set and join them on the sample plot ids
### Adapted from lidar-height-and-density-analysis.ipynb notebook

api_url = 'http://51.144.230.13:10323/api/point_list'
locations = df[['easting', 'northing']].values.tolist()
plot_ids = df.plot_id.values.tolist()

# The API currently gives an error for large number of locations, so we must get the data in batches.
batch_size = 1000
data_batches = []
for batch_start in tqdm(range(0, len(locations), batch_size)):
    
    locations_batch = locations[batch_start:batch_start+batch_size]
    plot_id_batch = plot_ids[batch_start:batch_start+batch_size]
    post_json = json.dumps({
            'srid': 3067,
            'coordinates': locations_batch,
            'fids': plot_id_batch
        })
    params = {
            'schema': 'metsakeskus_hila',
            'table': 'gridcell',
            'columns': ['volumepine,volumespruce,volumedeciduous,volume']
        }

    post_headers = {'Content-Type': 'application/json'}

    res = requests.post(api_url, data=post_json, headers=post_headers, params=params)
    data_batch = res.json()
    data_batch = [(feature['properties']['fid'],
                         feature['properties']['volumepine'],feature['properties']['volumespruce'],
                         feature['properties']['volumedeciduous'],feature['properties']['volume']
                        ) 
                        for feature in data_batch['features']]
    
    data_batch = pd.DataFrame(data_batch, columns=['plot_id','volumepine', 'volumespruce', 'volumedeciduous', 'volumetotal'])
    
    data_batches.append(data_batch)
    
data = pd.concat(data_batches, axis=0, ignore_index=True)
data_unique = data.loc[data.plot_id.drop_duplicates().index]
scalar_df = data_unique.merge(df, on='plot_id').drop_duplicates(subset='plot_id')
76/7:
### Cell for running own models

import keras
from keras.models import load_model
from features import preprocessing
from keras.utils.generic_utils import CustomObjectScope
from keras.layers import DepthwiseConv2D
from keras_applications.mobilenet import relu6

model_path = r"/home/tman/Work/linda-forestry-ml/species_prediction/regressors/weights/mobilenet_mature19-01-22_12-05.hdf5"
with CustomObjectScope({'relu6': relu6,'DepthwiseConv2D': DepthwiseConv2D}):
    model = load_model(model_path)

### Use same preprocessing that the model used
76/8:
from features import preprocessing
X_preprocessed = preprocessing.crop_center(X)[0]
X_preprocessed = preprocessing.resize_images(X_preprocessed, input_dims=(128, 128))[0]
76/9:
X_preprocessed = preprocessing.crop_center(X)[0]
X_preprocessed = preprocessing.resize_images(X_preprocessed, input_dims=(128, 128))[0]

preds = mode.predict(X_preprocessed)
76/10:
X_preprocessed = preprocessing.crop_center(X)[0]
X_preprocessed = preprocessing.resize_images(X_preprocessed, input_dims=(128, 128))[0]

preds = model.predict(X_preprocessed)
76/11: preds[:5]
76/12: scalar_df[['vol_pine', 'vol_spruce', 'vol_deciduous', 'vol_total']][:5]
76/13: scalar_df[['vol_pine', 'vol_spruce', 'vol_deciduous', 'vol_total']][5:10]
76/14: preds[5:10]
76/15: preds[:50]
76/16:
### Cell for running own models

import keras
from keras.models import load_model
from features import preprocessing
from keras.utils.generic_utils import CustomObjectScope
from keras.layers import DepthwiseConv2D
from keras_applications.mobilenet import relu6

model_path = r"/home/tman/Work/linda-forestry-ml/species_prediction/regressors/weights/mobilenet_mature19-01-22_11-54.hdf5"
with CustomObjectScope({'relu6': relu6,'DepthwiseConv2D': DepthwiseConv2D}):
    model = load_model(model_path)

### Use same preprocessing that the model used
76/17:
X_preprocessed = preprocessing.crop_center(X)[0]
X_preprocessed = preprocessing.resize_images(X_preprocessed, input_dims=(128, 128))[0]

preds = model.predict(X_preprocessed)
76/18: preds[:50]
77/1:
%load_ext autoreload
%autoreload 2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import json
import os
from tqdm import tqdm
import re
import sys
from sklearn.model_selection import train_test_split
import requests
pd.options.display.float_format = '{:,.2f}'.format
# Add path to where utils.py is so metrics can be imported
sys.path.insert(0, r'/home/tman/Work/linda-forestry-ml/species_prediction/regressors/')

from data import data_loading
77/2:
input_path = r"/home/tman/Work/data/FIsampletiles"
labels_name = "test.csv"
image_dir = "azure_tiles_cleaned"
scalar_feature_names = ['easting', 'northing', 'measure_year', 'measure_date', 
                 'elevation', 'slope', 'aspect',  'soil_type',
                 'tree_cover', 'leaf_type', 'plot_id']
prediction_features=['vol_total', 'vol_pine', 'vol_spruce', 'vol_deciduous']

X, df = data_loading.import_data(input_path, labels_name=labels_name, image_dir=image_dir, image_type="jpg")
77/3:
### Get the metsäkeskus hila predictions on the test set and join them on the sample plot ids
### Adapted from lidar-height-and-density-analysis.ipynb notebook

api_url = 'http://51.144.230.13:10323/api/point_list'
locations = df[['easting', 'northing']].values.tolist()
plot_ids = df.plot_id.values.tolist()

# The API currently gives an error for large number of locations, so we must get the data in batches.
batch_size = 1000
data_batches = []
for batch_start in tqdm(range(0, len(locations), batch_size)):
    
    locations_batch = locations[batch_start:batch_start+batch_size]
    plot_id_batch = plot_ids[batch_start:batch_start+batch_size]
    post_json = json.dumps({
            'srid': 3067,
            'coordinates': locations_batch,
            'fids': plot_id_batch
        })
    params = {
            'schema': 'metsakeskus_hila',
            'table': 'gridcell',
            'columns': ['volumepine,volumespruce,volumedeciduous,volume']
        }

    post_headers = {'Content-Type': 'application/json'}

    res = requests.post(api_url, data=post_json, headers=post_headers, params=params)
    data_batch = res.json()
    data_batch = [(feature['properties']['fid'],
                         feature['properties']['volumepine'],feature['properties']['volumespruce'],
                         feature['properties']['volumedeciduous'],feature['properties']['volume']
                        ) 
                        for feature in data_batch['features']]
    
    data_batch = pd.DataFrame(data_batch, columns=['plot_id','volumepine', 'volumespruce', 'volumedeciduous', 'volumetotal'])
    
    data_batches.append(data_batch)
    
data = pd.concat(data_batches, axis=0, ignore_index=True)
data_unique = data.loc[data.plot_id.drop_duplicates().index]
scalar_df = data_unique.merge(df, on='plot_id').drop_duplicates(subset='plot_id')
77/4:
from features import preprocessing
X_preprocessed = preprocessing.crop_center(X)[0]
X_preprocessed = preprocessing.resize_images(X_preprocessed, input_dims=(128, 128))[0]
77/5:
plt.imshow(X[0])
plt.imshow(X_preprocessed[0])
77/6: plt.imshow(X[0])
77/7: plt.imshow(X_preprocessed[0])
78/1:
%load_ext autoreload
%autoreload 2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import json
import os
from tqdm import tqdm
import re
import sys
from sklearn.model_selection import train_test_split
import requests
pd.options.display.float_format = '{:,.2f}'.format
# Add path to where utils.py is so metrics can be imported
sys.path.insert(0, r'/home/tman/Work/linda-forestry-ml/species_prediction/regressors/')

from data import data_loading
78/2:
input_path = r"/home/tman/Work/data/FIsampletiles"
labels_name = "test.csv"
image_dir = "azure_tiles_cleaned"
scalar_feature_names = ['easting', 'northing', 'measure_year', 'measure_date', 
                 'elevation', 'slope', 'aspect',  'soil_type',
                 'tree_cover', 'leaf_type', 'plot_id']
prediction_features=['vol_total', 'vol_pine', 'vol_spruce', 'vol_deciduous']

X, df = data_loading.import_data(input_path, labels_name=labels_name, image_dir=image_dir, image_type="jpg")
78/3:
### Get the metsäkeskus hila predictions on the test set and join them on the sample plot ids
### Adapted from lidar-height-and-density-analysis.ipynb notebook

api_url = 'http://51.144.230.13:10323/api/point_list'
locations = df[['easting', 'northing']].values.tolist()
plot_ids = df.plot_id.values.tolist()

# The API currently gives an error for large number of locations, so we must get the data in batches.
batch_size = 1000
data_batches = []
for batch_start in tqdm(range(0, len(locations), batch_size)):
    
    locations_batch = locations[batch_start:batch_start+batch_size]
    plot_id_batch = plot_ids[batch_start:batch_start+batch_size]
    post_json = json.dumps({
            'srid': 3067,
            'coordinates': locations_batch,
            'fids': plot_id_batch
        })
    params = {
            'schema': 'metsakeskus_hila',
            'table': 'gridcell',
            'columns': ['volumepine,volumespruce,volumedeciduous,volume']
        }

    post_headers = {'Content-Type': 'application/json'}

    res = requests.post(api_url, data=post_json, headers=post_headers, params=params)
    data_batch = res.json()
    data_batch = [(feature['properties']['fid'],
                         feature['properties']['volumepine'],feature['properties']['volumespruce'],
                         feature['properties']['volumedeciduous'],feature['properties']['volume']
                        ) 
                        for feature in data_batch['features']]
    
    data_batch = pd.DataFrame(data_batch, columns=['plot_id','volumepine', 'volumespruce', 'volumedeciduous', 'volumetotal'])
    
    data_batches.append(data_batch)
    
data = pd.concat(data_batches, axis=0, ignore_index=True)
data_unique = data.loc[data.plot_id.drop_duplicates().index]
scalar_df = data_unique.merge(df, on='plot_id').drop_duplicates(subset='plot_id')
78/4:
### Cell for running own models

import keras
from keras.models import load_model
from keras.utils.generic_utils import CustomObjectScope
from keras.layers import DepthwiseConv2D
from keras_applications.mobilenet import relu6

model_path = r"/home/tman/Work/linda-forestry-ml/species_prediction/regressors/weights/lenet_mature19-01-22_13-23.hdf5"
model = load_model(model_path)

### Use same preprocessing that the model used
78/5:
from features import preprocessing
X_preprocessed = preprocessing.crop_center(X)[0]
78/6:
preds = model.predict(X_preprocessed)
preds[:10]
78/7: scalar_df[['vol_pine', 'vol_spruce', 'vol_deciduous', 'vol_total']][:10]
79/1:
%load_ext autoreload
%autoreload 2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import json
import os
from tqdm import tqdm
import re
import sys
from sklearn.model_selection import train_test_split
import requests
pd.options.display.float_format = '{:,.2f}'.format
# Add path to where utils.py is so metrics can be imported
sys.path.insert(0, r'/home/tman/Work/linda-forestry-ml/species_prediction/regressors/')

from data import data_loading
79/2:
input_path = r"/home/tman/Work/data/FIsampletiles"
labels_name = "test.csv"
image_dir = "azure_tiles_cleaned"
scalar_feature_names = ['easting', 'northing', 'measure_year', 'measure_date', 
                 'elevation', 'slope', 'aspect',  'soil_type',
                 'tree_cover', 'leaf_type', 'plot_id']
prediction_features=['vol_total', 'vol_pine', 'vol_spruce', 'vol_deciduous']

X, df = data_loading.import_data(input_path, labels_name=labels_name, image_dir=image_dir, image_type="jpg")
79/3:
### Get the metsäkeskus hila predictions on the test set and join them on the sample plot ids
### Adapted from lidar-height-and-density-analysis.ipynb notebook

api_url = 'http://51.144.230.13:10323/api/point_list'
locations = df[['easting', 'northing']].values.tolist()
plot_ids = df.plot_id.values.tolist()

# The API currently gives an error for large number of locations, so we must get the data in batches.
batch_size = 1000
data_batches = []
for batch_start in tqdm(range(0, len(locations), batch_size)):
    
    locations_batch = locations[batch_start:batch_start+batch_size]
    plot_id_batch = plot_ids[batch_start:batch_start+batch_size]
    post_json = json.dumps({
            'srid': 3067,
            'coordinates': locations_batch,
            'fids': plot_id_batch
        })
    params = {
            'schema': 'metsakeskus_hila',
            'table': 'gridcell',
            'columns': ['volumepine,volumespruce,volumedeciduous,volume']
        }

    post_headers = {'Content-Type': 'application/json'}

    res = requests.post(api_url, data=post_json, headers=post_headers, params=params)
    data_batch = res.json()
    data_batch = [(feature['properties']['fid'],
                         feature['properties']['volumepine'],feature['properties']['volumespruce'],
                         feature['properties']['volumedeciduous'],feature['properties']['volume']
                        ) 
                        for feature in data_batch['features']]
    
    data_batch = pd.DataFrame(data_batch, columns=['plot_id','volumepine', 'volumespruce', 'volumedeciduous', 'volumetotal'])
    
    data_batches.append(data_batch)
    
data = pd.concat(data_batches, axis=0, ignore_index=True)
data_unique = data.loc[data.plot_id.drop_duplicates().index]
scalar_df = data_unique.merge(df, on='plot_id').drop_duplicates(subset='plot_id')
79/4:
### Cell for running own models

import keras
from keras.models import load_model
from keras.utils.generic_utils import CustomObjectScope
from keras.layers import DepthwiseConv2D
from keras_applications.mobilenet import relu6

model_path = r"/home/tman/Work/linda-forestry-ml/species_prediction/regressors/weights/lenet_mature19-01-22_13-23.hdf5"
model = load_model(model_path)

### Use same preprocessing that the model used
79/5:
from features import preprocessing
X_preprocessed = preprocessing.crop_center(X)[0]
79/6:
preds = model.predict(X_preprocessed)
preds[:10]
79/7: scalar_df[['vol_pine', 'vol_spruce', 'vol_deciduous', 'vol_total']][:10]
80/1:
%load_ext autoreload
%autoreload 2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import json
import os
from tqdm import tqdm
import re
import sys
from sklearn.model_selection import train_test_split
import requests
pd.options.display.float_format = '{:,.2f}'.format
# Add path to where utils.py is so metrics can be imported
sys.path.insert(0, r'/home/tman/Work/linda-forestry-ml/species_prediction/regressors/')

from data import data_loading
80/2:
%load_ext autoreload
%autoreload 2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import json
import os
from tqdm import tqdm
import re
import sys
from sklearn.model_selection import train_test_split
import requests
pd.options.display.float_format = '{:,.2f}'.format
# Add path to where utils.py is so metrics can be imported
sys.path.insert(0, r'/home/tman/Work/linda-forestry-ml/species_prediction/regressors/')

from data import data_loading
80/3:
input_path = r"/home/tman/Work/data/FIsampletiles"
labels_name = "test.csv"
image_dir = "azure_tiles_cleaned"
scalar_feature_names = ['easting', 'northing', 'measure_year', 'measure_date', 
                 'elevation', 'slope', 'aspect',  'soil_type',
                 'tree_cover', 'leaf_type', 'plot_id']
prediction_features=['vol_total', 'vol_pine', 'vol_spruce', 'vol_deciduous']

X, df = data_loading.import_data(input_path, labels_name=labels_name, image_dir=image_dir, image_type="jpg")
80/4:
### Get the metsäkeskus hila predictions on the test set and join them on the sample plot ids
### Adapted from lidar-height-and-density-analysis.ipynb notebook

api_url = 'http://51.144.230.13:10323/api/point_list'
locations = df[['easting', 'northing']].values.tolist()
plot_ids = df.plot_id.values.tolist()

# The API currently gives an error for large number of locations, so we must get the data in batches.
batch_size = 1000
data_batches = []
for batch_start in tqdm(range(0, len(locations), batch_size)):
    
    locations_batch = locations[batch_start:batch_start+batch_size]
    plot_id_batch = plot_ids[batch_start:batch_start+batch_size]
    post_json = json.dumps({
            'srid': 3067,
            'coordinates': locations_batch,
            'fids': plot_id_batch
        })
    params = {
            'schema': 'metsakeskus_hila',
            'table': 'gridcell',
            'columns': ['volumepine,volumespruce,volumedeciduous,volume']
        }

    post_headers = {'Content-Type': 'application/json'}

    res = requests.post(api_url, data=post_json, headers=post_headers, params=params)
    data_batch = res.json()
    data_batch = [(feature['properties']['fid'],
                         feature['properties']['volumepine'],feature['properties']['volumespruce'],
                         feature['properties']['volumedeciduous'],feature['properties']['volume']
                        ) 
                        for feature in data_batch['features']]
    
    data_batch = pd.DataFrame(data_batch, columns=['plot_id','volumepine', 'volumespruce', 'volumedeciduous', 'volumetotal'])
    
    data_batches.append(data_batch)
    
data = pd.concat(data_batches, axis=0, ignore_index=True)
data_unique = data.loc[data.plot_id.drop_duplicates().index]
scalar_df = data_unique.merge(df, on='plot_id').drop_duplicates(subset='plot_id')
80/5:
### Cell for running own models

import keras
from keras.models import load_model
from keras.utils.generic_utils import CustomObjectScope
from keras.layers import DepthwiseConv2D
from keras_applications.mobilenet import relu6

model_path = r"/home/tman/Work/linda-forestry-ml/species_prediction/regressors/weights/lenet_mature19-01-22_13-43.hdf5"
model = load_model(model_path)

### Use same preprocessing that the model used
80/6:
from features import preprocessing
X_preprocessed = preprocessing.crop_center(X)[0]
80/7:
preds = model.predict(X_preprocessed)
preds[:10]
80/8:
preds = model.predict(X_preprocessed)
from sklearn.metrics import mean_squared_error
80/9: scalar_df[['vol_pine', 'vol_spruce', 'vol_deciduous', 'vol_total']][:10]
80/10:
preds = model.predict(X_preprocessed)
from sklearn.metrics import mean_squared_error
groundtruth = scalar_df[['vol_pine', 'vol_spruce', 'vol_deciduous']]
mean_squared_error(preds, groundtruth, multioutput='raw_values')) / np.mean(groundtruth, axis=0)
80/11:
preds = model.predict(X_preprocessed)
from sklearn.metrics import mean_squared_error
groundtruth = scalar_df[['vol_pine', 'vol_spruce', 'vol_deciduous']]
mean_squared_error(preds, groundtruth, multioutput='raw_values')
81/1:
%load_ext autoreload
%autoreload 2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import json
import os
from tqdm import tqdm
import re
import sys
from sklearn.model_selection import train_test_split
import requests
pd.options.display.float_format = '{:,.2f}'.format
# Add path to where utils.py is so metrics can be imported
sys.path.insert(0, r'/home/tman/Work/linda-forestry-ml/species_prediction/regressors/')

from data import data_loading
81/2:
input_path = r"/home/tman/Work/data/FIsampletiles"
labels_name = "train.csv"
image_dir = "azure_tiles_cleaned"
scalar_feature_names = ['easting', 'northing', 'measure_year', 'measure_date', 
                 'elevation', 'slope', 'aspect',  'soil_type',
                 'tree_cover', 'leaf_type', 'plot_id']
prediction_features=['vol_total', 'vol_pine', 'vol_spruce', 'vol_deciduous']

X, df = data_loading.import_data(input_path, labels_name=labels_name, image_dir=image_dir, image_type="jpg")
81/3:
### Get the metsäkeskus hila predictions on the test set and join them on the sample plot ids
### Adapted from lidar-height-and-density-analysis.ipynb notebook

api_url = 'http://51.144.230.13:10323/api/point_list'
locations = df[['easting', 'northing']].values.tolist()
plot_ids = df.plot_id.values.tolist()

# The API currently gives an error for large number of locations, so we must get the data in batches.
batch_size = 1000
data_batches = []
for batch_start in tqdm(range(0, len(locations), batch_size)):
    
    locations_batch = locations[batch_start:batch_start+batch_size]
    plot_id_batch = plot_ids[batch_start:batch_start+batch_size]
    post_json = json.dumps({
            'srid': 3067,
            'coordinates': locations_batch,
            'fids': plot_id_batch
        })
    params = {
            'schema': 'metsakeskus_hila',
            'table': 'gridcell',
            'columns': ['volumepine,volumespruce,volumedeciduous,volume']
        }

    post_headers = {'Content-Type': 'application/json'}

    res = requests.post(api_url, data=post_json, headers=post_headers, params=params)
    data_batch = res.json()
    data_batch = [(feature['properties']['fid'],
                         feature['properties']['volumepine'],feature['properties']['volumespruce'],
                         feature['properties']['volumedeciduous'],feature['properties']['volume']
                        ) 
                        for feature in data_batch['features']]
    
    data_batch = pd.DataFrame(data_batch, columns=['plot_id','volumepine', 'volumespruce', 'volumedeciduous', 'volumetotal'])
    
    data_batches.append(data_batch)
    
data = pd.concat(data_batches, axis=0, ignore_index=True)
data_unique = data.loc[data.plot_id.drop_duplicates().index]
scalar_df = data_unique.merge(df, on='plot_id').drop_duplicates(subset='plot_id')
81/4:
### Cell for running own models

import keras
from keras.models import load_model
from keras.utils.generic_utils import CustomObjectScope
from keras.layers import DepthwiseConv2D
from keras_applications.mobilenet import relu6

model_path = r"/home/tman/Work/linda-forestry-ml/species_prediction/regressors/weights/lenet_mature19-01-22_13-43.hdf5"
model = load_model(model_path)

### Use same preprocessing that the model used
81/5:
from features import preprocessing
X_preprocessed = preprocessing.crop_center(X)[0]
81/6:
preds = model.predict(X_preprocessed)
from sklearn.metrics import mean_squared_error
groundtruth = scalar_df[['vol_pine', 'vol_spruce', 'vol_deciduous']]
mean_squared_error(preds, groundtruth, multioutput='raw_values')
81/7: df[['vol_pine', 'vol_spruce', 'vol_deciduous', 'vol_total']][:10]
81/8: scalar_df[['vol_pine', 'vol_spruce', 'vol_deciduous', 'vol_total']][:10]
82/1:
%load_ext autoreload
%autoreload 2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import json
import os
from tqdm import tqdm
import re
import sys
from sklearn.model_selection import train_test_split
import requests
pd.options.display.float_format = '{:,.2f}'.format
# Add path to where utils.py is so metrics can be imported
sys.path.insert(0, r'/home/tman/Work/linda-forestry-ml/species_prediction/regressors/')

from data import data_loading
82/2:
input_path = r"/home/tman/Work/data/FIsampletiles"
labels_name = "test.csv"
image_dir = "azure_tiles_cleaned"
scalar_feature_names = ['easting', 'northing', 'measure_year', 'measure_date', 
                 'elevation', 'slope', 'aspect',  'soil_type',
                 'tree_cover', 'leaf_type', 'plot_id']
prediction_features=['vol_total', 'vol_pine', 'vol_spruce', 'vol_deciduous']

X, df = data_loading.import_data(input_path, labels_name=labels_name, image_dir=image_dir, image_type="jpg")
82/3:
### Get the metsäkeskus hila predictions on the test set and join them on the sample plot ids
### Adapted from lidar-height-and-density-analysis.ipynb notebook

api_url = 'http://51.144.230.13:10323/api/point_list'
locations = df[['easting', 'northing']].values.tolist()
plot_ids = df.plot_id.values.tolist()

# The API currently gives an error for large number of locations, so we must get the data in batches.
batch_size = 1000
data_batches = []
for batch_start in tqdm(range(0, len(locations), batch_size)):
    
    locations_batch = locations[batch_start:batch_start+batch_size]
    plot_id_batch = plot_ids[batch_start:batch_start+batch_size]
    post_json = json.dumps({
            'srid': 3067,
            'coordinates': locations_batch,
            'fids': plot_id_batch
        })
    params = {
            'schema': 'metsakeskus_hila',
            'table': 'gridcell',
            'columns': ['volumepine,volumespruce,volumedeciduous,volume']
        }

    post_headers = {'Content-Type': 'application/json'}

    res = requests.post(api_url, data=post_json, headers=post_headers, params=params)
    data_batch = res.json()
    data_batch = [(feature['properties']['fid'],
                         feature['properties']['volumepine'],feature['properties']['volumespruce'],
                         feature['properties']['volumedeciduous'],feature['properties']['volume']
                        ) 
                        for feature in data_batch['features']]
    
    data_batch = pd.DataFrame(data_batch, columns=['plot_id','volumepine', 'volumespruce', 'volumedeciduous', 'volumetotal'])
    
    data_batches.append(data_batch)
    
data = pd.concat(data_batches, axis=0, ignore_index=True)
data_unique = data.loc[data.plot_id.drop_duplicates().index]

### TODO: are the images and this scalar df matched? ALSO NORMALIZATION SHEESH
scalar_df = data_unique.merge(df, on='plot_id')
82/4:
### Cell for running own models

import keras
from keras.models import load_model
from keras.utils.generic_utils import CustomObjectScope
from keras.layers import DepthwiseConv2D
from keras_applications.mobilenet import relu6

model_path = r"/home/tman/Work/linda-forestry-ml/species_prediction/regressors/weights/mobilenet_mature19-01-22_11-54.hdf5"
with CustomObjectScope({'relu6': relu6,'DepthwiseConv2D': DepthwiseConv2D}):
    model = load_model(model_path)

### Use same preprocessing that the model used
82/5:
from features import preprocessing
from keras.applications.mobilenet import preprocess_input as preprocess_input_mobilenet
X_preprocessed = preprocessing.crop_center(X)[0]
X_preprocessed = preprocessing.resize_images(X_preprocessed, input_dims=(128, 128))[0]
X_preprocessed = preprocess_input_mobilenet(X_preprocessed)
82/6:
preds = model.predict(X_preprocessed)
preds[:10]
82/7: df[['vol_pine', 'vol_spruce', 'vol_deciduous', 'vol_total']][:10]
82/8:
metsakeskus_predictions = scalar_df[['volumepine', 'volumespruce', 'volumedeciduous']]
groundtruth = scalar_df[['vol_pine', 'vol_spruce', 'vol_deciduous']]
82/9:
from sklearn.metrics import mean_squared_error

print("NRMSE% of metsakeskus predictions on the test set:")

(np.sqrt(mean_squared_error(metsakeskus_predictions, groundtruth, multioutput='raw_values')) / np.mean(groundtruth, axis=0))*100

print("NRMSE% of metsakeskus predictions on the test set:")

(np.sqrt(mean_squared_error(preds, groundtruth, multioutput='raw_values')) / np.mean(groundtruth, axis=0))*100
82/10:
from sklearn.metrics import mean_squared_error

print("NRMSE% of metsakeskus predictions on the test set:")

print((np.sqrt(mean_squared_error(metsakeskus_predictions, groundtruth, multioutput='raw_values')) / np.mean(groundtruth, axis=0))*100)

print("NRMSE% of metsakeskus predictions on the test set:")

print((np.sqrt(mean_squared_error(preds, groundtruth, multioutput='raw_values')) / np.mean(groundtruth, axis=0))*100)
82/11:
from sklearn.metrics import mean_squared_error

print("NRMSE% of metsakeskus predictions on the test set:")

print((np.sqrt(mean_squared_error(metsakeskus_predictions, groundtruth, multioutput='raw_values')) / np.mean(groundtruth, axis=0))*100)

print("NRMSE% of our predictions on the test set:")

print((np.sqrt(mean_squared_error(preds, groundtruth, multioutput='raw_values')) / np.mean(groundtruth, axis=0))*100)
87/1:
%load_ext autoreload
%autoreload 2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import json
import os
from tqdm import tqdm
import re
import sys
from sklearn.model_selection import train_test_split
import requests
pd.options.display.float_format = '{:,.2f}'.format
# Add path to where utils.py is so metrics can be imported
sys.path.insert(0, r'C:/Users/Teemu/Work/linda-forestry-ml/species_prediction/regressors')

from models import models_definition
from data import data_loading
87/2:
%load_ext autoreload
%autoreload 2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import json
import os
from tqdm import tqdm
import re
import sys
from sklearn.model_selection import train_test_split
import requests
pd.options.display.float_format = '{:,.2f}'.format
# Add path to where utils.py is so metrics can be imported
# sys.path.insert(0, r'C:/Users/Teemu/Work/linda-forestry-ml/species_prediction/regressors')
87/3:
input_path = r"/home/tman/Work/data/FIsampletiles"
labels_name = "groundtruth.csv"
image_dir = "azure_tiles_cleaned"
#scalar_feature_names = ['easting', 'northing', 'measure_year', 'measure_date', 
#                 'elevation', 'slope', 'aspect',  'soil_type',
#                 'tree_cover', 'leaf_type', 'plot_id']
prediction_features=['vol_total', 'vol_pine', 'vol_spruce', 'vol_deciduous']

#X, df = data_loading.import_data(input_path, labels_name=labels_name, image_dir="tiles")
df = pd.read_csv(os.path.join(input_path, 'groundtruth.csv'))
87/4: df
87/5:
def get_metsakeskus_predictions(df)
    schema_list = ['metsakeskus_hila']
    tables_list = ['gridcell']
    columns_list = [['volumepine', 'volumespruce', 'volumedeciduous', 'volume']]

    api = GeoAPI(default_locations=df[['easting', 'northing']].values.tolist(),
                default_srid=3067, default_plot_ids=df.plot_id.values.tolist())

    hiladata = api.request_data(schema_list, tables_list, columns_list, batch_size=1000)

    # Return plot_ids from index to a column.
    hiladata.reset_index()
    return hiladata
    
hd = get_metsakeskus_predictions(df)
87/6:
def get_metsakeskus_predictions(df):
    schema_list = ['metsakeskus_hila']
    tables_list = ['gridcell']
    columns_list = [['volumepine', 'volumespruce', 'volumedeciduous', 'volume']]

    api = GeoAPI(default_locations=df[['easting', 'northing']].values.tolist(),
                default_srid=3067, default_plot_ids=df.plot_id.values.tolist())

    hiladata = api.request_data(schema_list, tables_list, columns_list, batch_size=1000)

    # Return plot_ids from index to a column.
    hiladata.reset_index()
    return hiladata
    
hd = get_metsakeskus_predictions(df)
87/7:
%load_ext autoreload
%autoreload 2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import json
import os
from tqdm import tqdm
import re
import sys
from sklearn.model_selection import train_test_split
import requests
sys.path.append('../regressors/')
from data.data_loading import import_data, GeoAPI, split_from_ids

pd.options.display.float_format = '{:,.2f}'.format

# Add path to where utils.py is so metrics can be imported
# sys.path.insert(0, r'C:/Users/Teemu/Work/linda-forestry-ml/species_prediction/regressors')
87/8:
def get_metsakeskus_predictions(df):
    schema_list = ['metsakeskus_hila']
    tables_list = ['gridcell']
    columns_list = [['volumepine', 'volumespruce', 'volumedeciduous', 'volume']]

    api = GeoAPI(default_locations=df[['easting', 'northing']].values.tolist(),
                default_srid=3067, default_plot_ids=df.plot_id.values.tolist())

    hiladata = api.request_data(schema_list, tables_list, columns_list, batch_size=1000)

    # Return plot_ids from index to a column.
    hiladata.reset_index()
    return hiladata
    
hd = get_metsakeskus_predictions(df)
87/9: hd
88/1:
%load_ext autoreload
%autoreload 2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import json
import os
from tqdm import tqdm
import re
import sys
from sklearn.model_selection import train_test_split
import requests
sys.path.append('../regressors/')
from data.data_loading import import_data, GeoAPI, split_from_ids

pd.options.display.float_format = '{:,.2f}'.format

# Add path to where utils.py is so metrics can be imported
# sys.path.insert(0, r'C:/Users/Teemu/Work/linda-forestry-ml/species_prediction/regressors')
88/2:
input_path = r"/home/tman/Work/data/FIsampletiles"
labels_name = "groundtruth.csv"
image_dir = "azure_tiles_cleaned"
#scalar_feature_names = ['easting', 'northing', 'measure_year', 'measure_date', 
#                 'elevation', 'slope', 'aspect',  'soil_type',
#                 'tree_cover', 'leaf_type', 'plot_id']
prediction_features=['vol_pine', 'vol_spruce', 'vol_deciduous']

#X, df = data_loading.import_data(input_path, labels_name=labels_name, image_dir="tiles")
df = pd.read_csv(os.path.join(input_path, 'test.csv'))
88/3:
def get_metsakeskus_predictions(df):
    schema_list = ['metsakeskus_hila']
    tables_list = ['gridcell']
    columns_list = [['volumepine', 'volumespruce', 'volumedeciduous']]

    api = GeoAPI(default_locations=df[['easting', 'northing']].values.tolist(),
                default_srid=3067, default_plot_ids=df.plot_id.values.tolist())

    hiladata = api.request_data(schema_list, tables_list, columns_list, batch_size=1000)

    # Return plot_ids from index to a column.
    hiladata.reset_index()
    return hiladata

def metsakeskus_errors(df):
    from sklearn.metrics import mean_squared_error
    
    metsakeskus_predictions = get_metsakeskus_predictions(df)
    prediction_features=['vol_pine', 'vol_spruce', 'vol_deciduous']
    groundtruth = df[prediction_features]
    mse = mean_squared_error(metsakeskus_predictions, groundtruth, multioutput='raw_values')
    rmse = np.sqrt(mse)
    gt_means = np.mean(groundtruth, axis=0)
    nrmse = (rmse / gt_means)*100
    return gt_means, rmse, nrmse
    
gt_means, rmse, nrmse = metsakeskus_errors(df)
print(gt_means)
print(rmse)
print(nrmse)
88/4: len(df)
88/5: metsakeskus_predictions = get_metsakeskus_predictions(df)
88/6: len(metsakeskus_predictions)
88/7: metsakeskus_predictions
88/8: metsakeskus_predictions.isna().sum()
88/9: metsakeskus_predictions.duplicated().sum()
88/10: len(metsakeskus_predictions)
88/11: df.merge(metsakeskus_predictions)
88/12: metsakeskues_predictions
88/13: metsakeskus_predictions
88/14:
def get_metsakeskus_predictions(df):
    schema_list = ['metsakeskus_hila']
    tables_list = ['gridcell']
    columns_list = [['volumepine', 'volumespruce', 'volumedeciduous']]

    api = GeoAPI(default_locations=df[['easting', 'northing']].values.tolist(),
                default_srid=3067, default_plot_ids=df.plot_id.values.tolist())

    hiladata = api.request_data(schema_list, tables_list, columns_list, batch_size=1000)

    # Return plot_ids from index to a column.
    hiladata.reset_index(inplace=True)
    return hiladata

def metsakeskus_errors(df):
    from sklearn.metrics import mean_squared_error
    
    metsakeskus_predictions = get_metsakeskus_predictions(df)
    prediction_features=['vol_pine', 'vol_spruce', 'vol_deciduous']
    groundtruth = df[prediction_features]
    mse = mean_squared_error(groundtruth, metsakeskus_predictions, multioutput='raw_values')
    rmse = np.sqrt(mse)
    gt_means = np.mean(groundtruth, axis=0)
    nrmse = (rmse / gt_means)*100
    return gt_means, rmse, nrmse
    
gt_means, rmse, nrmse = metsakeskus_errors(df)
print(gt_means)
print(rmse)
print(nrmse)
88/15: metsakeskus_predictions = get_metsakeskus_predictions(df)
88/16: metsakeskus_predictions
88/17: df.merge(metsakeskus_predictions, on='plot_id')
88/18: df.merge(metsakeskus_predictions, on='plot_id').len()
88/19: df.merge(metsakeskus_predictions, on='plot_id').shape
88/20: df.merge(metsakeskus_predictions, on='plot_id').drop_duplicates()
88/21: df.merge(metsakeskus_predictions, on='plot_id').drop_duplicates().shape
88/22: df.merge(metsakeskus_predictions, on='plot_id').drop_duplicates(subset='plot_id').shape
88/23:
def get_metsakeskus_predictions(df, columns_list):
    schema_list = ['metsakeskus_hila']
    tables_list = ['gridcell']
    columns_list = [['volumepine', 'volumespruce', 'volumedeciduous']]

    api = GeoAPI(default_locations=df[['easting', 'northing']].values.tolist(),
                default_srid=3067, default_plot_ids=df.plot_id.values.tolist())

    hiladata = api.request_data(schema_list, tables_list, columns_list, batch_size=1000)

    # Return plot_ids from index to a column.
    hiladata.reset_index(inplace=True)
    return hiladata

def metsakeskus_errors(df):
    from sklearn.metrics import mean_squared_error
    
    prediction_features=['vol_pine', 'vol_spruce', 'vol_deciduous']
    metsakeskus_pred_columns = ['volumepine', 'volumespruce', 'volumedeciduous']
    
    metsakeskus_data = get_metsakeskus_predictions(df, metsakeskus_pred_columns)
    # API returns duplicated somewhat often with gridcell data, remove duplicates
    merged = df.merge(metsakeskus_predictions, on='plot_id').drop_duplicates(subset='plot_id')
    groundtruth = merged[prediction_features]
    metsakeskus_predictions = merged[metsakeskus_pred_columns]
    mse = mean_squared_error(groundtruth, metsakeskus_predictions, multioutput='raw_values')
    rmse = np.sqrt(mse)
    gt_means = np.mean(groundtruth, axis=0)
    nrmse = (rmse / gt_means)*100
    return gt_means, rmse, nrmse
    
gt_means, rmse, nrmse = metsakeskus_errors(df)
print(gt_means)
print(rmse)
print(nrmse)
88/24:
def get_metsakeskus_predictions(df, columns_list):
    schema_list = ['metsakeskus_hila']
    tables_list = ['gridcell']
    columns_list = [['volumepine', 'volumespruce', 'volumedeciduous']]

    api = GeoAPI(default_locations=df[['easting', 'northing']].values.tolist(),
                default_srid=3067, default_plot_ids=df.plot_id.values.tolist())

    hiladata = api.request_data(schema_list, tables_list, columns_list, batch_size=1000)

    # Return plot_ids from index to a column.
    hiladata.reset_index(inplace=True)
    return hiladata

def metsakeskus_errors(df):
    from sklearn.metrics import mean_squared_error
    
    prediction_features=['vol_pine', 'vol_spruce', 'vol_deciduous']
    metsakeskus_pred_columns = ['volumepine', 'volumespruce', 'volumedeciduous']
    
    metsakeskus_data = get_metsakeskus_predictions(df, metsakeskus_pred_columns)
    # API returns duplicated somewhat often with gridcell data, remove duplicates
    merged = df.merge(metsakeskus_data, on='plot_id').drop_duplicates(subset='plot_id')
    groundtruth = merged[prediction_features]
    metsakeskus_predictions = merged[metsakeskus_pred_columns]
    mse = mean_squared_error(groundtruth, metsakeskus_predictions, multioutput='raw_values')
    rmse = np.sqrt(mse)
    gt_means = np.mean(groundtruth, axis=0)
    nrmse = (rmse / gt_means)*100
    return gt_means, rmse, nrmse
    
gt_means, rmse, nrmse = metsakeskus_errors(df)
print(gt_means)
print(rmse)
print(nrmse)
88/25:
def get_metsakeskus_predictions(df, columns_list):
    schema_list = ['metsakeskus_hila']
    tables_list = ['gridcell']
    columns_list = [['volumepine', 'volumespruce', 'volumedeciduous']]

    api = GeoAPI(default_locations=df[['easting', 'northing']].values.tolist(),
                default_srid=3067, default_plot_ids=df.plot_id.values.tolist())

    hiladata = api.request_data(schema_list, tables_list, columns_list, batch_size=1000)

    # Return plot_ids from index to a column.
    hiladata.reset_index(inplace=True)
    return hiladata

def metsakeskus_errors(df):
    from sklearn.metrics import mean_squared_error
    
    prediction_features=['vol_pine', 'vol_spruce', 'vol_deciduous']
    metsakeskus_pred_columns = ['volumepine', 'volumespruce', 'volumedeciduous']
    
    metsakeskus_data = get_metsakeskus_predictions(df, metsakeskus_pred_columns)
    # API returns duplicated somewhat often with gridcell data, remove duplicates
    merged = df.merge(metsakeskus_data, on='plot_id').drop_duplicates(subset='plot_id')
    groundtruth = merged[prediction_features]
    metsakeskus_predictions = merged[metsakeskus_pred_columns]
    print(metsakeskus_predictions)
    mse = mean_squared_error(groundtruth, metsakeskus_predictions, multioutput='raw_values')
    rmse = np.sqrt(mse)
    gt_means = np.mean(groundtruth, axis=0)
    nrmse = (rmse / gt_means)*100
    return gt_means, rmse, nrmse
    
gt_means, rmse, nrmse = metsakeskus_errors(df)
print(gt_means)
print(rmse)
print(nrmse)
88/26:
def get_metsakeskus_predictions(df, columns_list):
    schema_list = ['metsakeskus_hila']
    tables_list = ['gridcell']
    columns_list = [['volumepine', 'volumespruce', 'volumedeciduous']]

    api = GeoAPI(default_locations=df[['easting', 'northing']].values.tolist(),
                default_srid=3067, default_plot_ids=df.plot_id.values.tolist())

    hiladata = api.request_data(schema_list, tables_list, columns_list, batch_size=1000)

    # Return plot_ids from index to a column.
    hiladata.reset_index(inplace=True)
    return hiladata

def metsakeskus_errors(df):
    from sklearn.metrics import mean_squared_error
    
    prediction_features=['vol_pine', 'vol_spruce', 'vol_deciduous']
    metsakeskus_pred_columns = ['volumepine', 'volumespruce', 'volumedeciduous']
    
    metsakeskus_data = get_metsakeskus_predictions(df, metsakeskus_pred_columns)
    # API returns duplicated somewhat often with gridcell data, remove duplicates
    merged = df.merge(metsakeskus_data, on='plot_id').drop_duplicates(subset='plot_id')
    groundtruth = merged[prediction_features]
    metsakeskus_predictions = merged[metsakeskus_pred_columns]
    print(np.mean(metsakeskus_predictions, axis=0))
    mse = mean_squared_error(groundtruth, metsakeskus_predictions, multioutput='raw_values')
    rmse = np.sqrt(mse)
    gt_means = np.mean(groundtruth, axis=0)
    nrmse = (rmse / gt_means)*100
    return gt_means, rmse, nrmse
    
gt_means, rmse, nrmse = metsakeskus_errors(df)
print(gt_means)
print(rmse)
print(nrmse)
89/1:
%load_ext autoreload
%autoreload 2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import json
import os
from tqdm import tqdm
import re
import sys
from sklearn.model_selection import train_test_split
import requests
sys.path.append('../regressors/')
from data.data_loading import import_data, GeoAPI, split_from_ids

pd.options.display.float_format = '{:,.2f}'.format

# Add path to where utils.py is so metrics can be imported
# sys.path.insert(0, r'C:/Users/Teemu/Work/linda-forestry-ml/species_prediction/regressors')
89/2:
input_path = r"/home/tman/Work/data/FIsampletiles"
labels_name = "groundtruth.csv"
image_dir = "azure_tiles_cleaned"
#scalar_feature_names = ['easting', 'northing', 'measure_year', 'measure_date', 
#                 'elevation', 'slope', 'aspect',  'soil_type',
#                 'tree_cover', 'leaf_type', 'plot_id']
prediction_features=['vol_pine', 'vol_spruce', 'vol_deciduous']

X, df = import_data(input_path, labels_name=labels_name, image_dir="tiles")
#df = pd.read_csv(os.path.join(input_path, 'test.csv'))
89/3:
input_path = r"/home/tman/Work/data/FIsampletiles"
labels_name = "test.csv"
image_dir = "azure_tiles_cleaned"
#scalar_feature_names = ['easting', 'northing', 'measure_year', 'measure_date', 
#                 'elevation', 'slope', 'aspect',  'soil_type',
#                 'tree_cover', 'leaf_type', 'plot_id']
prediction_features=['vol_pine', 'vol_spruce', 'vol_deciduous']

X, df = import_data(input_path, labels_name=labels_name, image_dir="tiles")
#df = pd.read_csv(os.path.join(input_path, 'test.csv'))
89/4:
input_path = r"/home/tman/Work/data/FIsampletiles"
labels_name = "test.csv"
image_dir = "azure_tiles_cleaned"
#scalar_feature_names = ['easting', 'northing', 'measure_year', 'measure_date', 
#                 'elevation', 'slope', 'aspect',  'soil_type',
#                 'tree_cover', 'leaf_type', 'plot_id']
prediction_features=['vol_pine', 'vol_spruce', 'vol_deciduous']

X, df = import_data(input_path, labels_name=labels_name, image_dir="tiles", image_type="jpg")
#df = pd.read_csv(os.path.join(input_path, 'test.csv'))
89/5: X
89/6: df
89/7:
input_path = r"/home/tman/Work/data/FIsampletiles"
labels_name = "test.csv"
image_dir = "azure_tiles_cleaned"
#scalar_feature_names = ['easting', 'northing', 'measure_year', 'measure_date', 
#                 'elevation', 'slope', 'aspect',  'soil_type',
#                 'tree_cover', 'leaf_type', 'plot_id']
prediction_features=['vol_pine', 'vol_spruce', 'vol_deciduous']

X, df = import_data(input_path, labels_name=labels_name, image_dir="azure_tiles_cleaned", image_type="jpg")
#df = pd.read_csv(os.path.join(input_path, 'test.csv'))
89/8:
def get_metsakeskus_predictions(df, columns_list):
    schema_list = ['metsakeskus_hila']
    tables_list = ['gridcell']
    columns_list = [['volumepine', 'volumespruce', 'volumedeciduous']]

    api = GeoAPI(default_locations=df[['easting', 'northing']].values.tolist(),
                default_srid=3067, default_plot_ids=df.plot_id.values.tolist())

    hiladata = api.request_data(schema_list, tables_list, columns_list, batch_size=1000)

    # Return plot_ids from index to a column.
    hiladata.reset_index(inplace=True)
    return hiladata

def metsakeskus_errors(df):
    from sklearn.metrics import mean_squared_error
    
    prediction_features=['vol_pine', 'vol_spruce', 'vol_deciduous']
    metsakeskus_pred_columns = ['volumepine', 'volumespruce', 'volumedeciduous']
    
    metsakeskus_data = get_metsakeskus_predictions(df, metsakeskus_pred_columns)
    # API returns duplicated somewhat often with gridcell data, remove duplicates
    merged = df.merge(metsakeskus_data, on='plot_id').drop_duplicates(subset='plot_id')
    groundtruth = merged[prediction_features]
    metsakeskus_predictions = merged[metsakeskus_pred_columns]
    mse = mean_squared_error(groundtruth, metsakeskus_predictions, multioutput='raw_values')
    rmse = np.sqrt(mse)
    gt_means = np.mean(groundtruth, axis=0)
    nrmse = (rmse / gt_means)*100
    return gt_means, rmse, nrmse
    
gt_means, rmse, nrmse = metsakeskus_errors(df)
print(gt_means)
print(rmse)
print(nrmse)
90/1:
%load_ext autoreload
%autoreload 2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import json
import os
from tqdm import tqdm
import re
import sys
from sklearn.model_selection import train_test_split
import requests
pd.options.display.float_format = '{:,.2f}'.format
# Add path to where utils.py is so metrics can be imported
sys.path.insert(0, r'/home/tman/Work/linda-forestry-ml/species_predictions/regressors')
from data.data_loading import import_data, GeoAPI, split_from_ids
90/2:
%load_ext autoreload
%autoreload 2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import json
import os
from tqdm import tqdm
import re
import sys
from sklearn.model_selection import train_test_split
import requests
pd.options.display.float_format = '{:,.2f}'.format
# Add path to where utils.py is so metrics can be imported
sys.path.insert(0, r'/home/tman/Work/linda-forestry-ml/species_prediction/regressors')
from data.data_loading import import_data, GeoAPI, split_from_ids
90/3:
input_path = r"/home/tman/Work/data/FIsampletiles"
image_dir = "azure_tiles_cleaned"
scalar_feature_names = ['easting', 'northing', 'measure_year', 'measure_date', 
                 'elevation', 'slope', 'aspect',  'soil_type',
                 'tree_cover', 'leaf_type', 'plot_id']
prediction_features=['vol_total', 'vol_pine', 'vol_spruce', 'vol_deciduous']

df = pd.read_csv(os.path.join(input_path, 'groundtruth.csv'))
#X, df = data_loading.import_data(input_path, labels_name=labels_name, image_dir=image_dir, image_type="jpg")
90/4:
### Get the metsäkeskus hila data

columns_string = """volumepine,volumespruce,volumedeciduous,volume,soiltype,fertilityclass,maingroup,subgroup,\
laserheight,laserdensity"""

schema_list = ['metsakeskus_hila']
tables_list = ['gridcell']
columns_list = [[columns_string.split(",")]

api = GeoAPI(default_locations=df[['easting', 'northing']].values.tolist(),
            default_srid=3067, default_plot_ids=df.plot_id.values.tolist())

hiladata = api.request_data(schema_list, tables_list, columns_list, batch_size=1000)

# Return plot_ids from index to a column.
hiladata.reset_index(inplace=True)
90/5:
### Get the metsäkeskus hila data

columns_string = """volumepine,volumespruce,volumedeciduous,volume,soiltype,fertilityclass,maingroup,subgroup,\
laserheight,laserdensity"""

schema_list = ['metsakeskus_hila']
tables_list = ['gridcell']
columns_list = [[columns_string.split(",")]]

api = GeoAPI(default_locations=df[['easting', 'northing']].values.tolist(),
            default_srid=3067, default_plot_ids=df.plot_id.values.tolist())

hiladata = api.request_data(schema_list, tables_list, columns_list, batch_size=1000)

# Return plot_ids from index to a column.
hiladata.reset_index(inplace=True)
90/6:
### Get the metsäkeskus hila data

columns_string = """volumepine,volumespruce,volumedeciduous,volume,soiltype,fertilityclass,maingroup,subgroup,\
laserheight,laserdensity"""

schema_list = ['metsakeskus_hila']
tables_list = ['gridcell']
columns_list = [columns_string.split(",")]

api = GeoAPI(default_locations=df[['easting', 'northing']].values.tolist(),
            default_srid=3067, default_plot_ids=df.plot_id.values.tolist())

hiladata = api.request_data(schema_list, tables_list, columns_list, batch_size=1000)

# Return plot_ids from index to a column.
hiladata.reset_index(inplace=True)
90/7: hiladata
90/8: hiladata.isna().sum()
95/1:
%load_ext autoreload
%autoreload 2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import json
import os
from tqdm import tqdm
import re
import sys
from sklearn.model_selection import train_test_split
import requests
pd.options.display.float_format = '{:,.2f}'.format
# Add path to where utils.py is so metrics can be imported
sys.path.insert(0, r'/home/tman/Work/linda-forestry-ml/species_prediction/regressors')
from data.data_loading import import_data, GeoAPI, split_from_ids
from data import data_loading
95/2:
input_path = r"/home/tman/Work/data/FIsampletiles"

df = pd.read_csv(os.path.join(input_path, 'groundtruth.csv'))
95/3:
### Get the metsäkeskus hila data
def get_metsakeskus_data(df)
    columns_string = """volumepine,volumespruce,volumedeciduous,volume,soiltype,fertilityclass,maingroup,subgroup,\
    laserheight,laserdensity"""

    schema_list = ['metsakeskus_hila']
    tables_list = ['gridcell']
    columns_list = [columns_string.split(",")]

    api = GeoAPI(default_locations=df[['easting', 'northing']].values.tolist(),
                default_srid=3067, default_plot_ids=df.plot_id.values.tolist())

    hiladata = api.request_data(schema_list, tables_list, columns_list, batch_size=1000)

    # Return plot_ids from index to a column.
    hiladata.reset_index(inplace=True)
    hiladata = hiladata.drop_duplicates()
    
    return hiladata

hiladata = get_metsakeskus_data(df)
95/4:
### Get the metsäkeskus hila data
def get_metsakeskus_data(df):
    columns_string = """volumepine,volumespruce,volumedeciduous,volume,soiltype,fertilityclass,maingroup,subgroup,\
    laserheight,laserdensity"""

    schema_list = ['metsakeskus_hila']
    tables_list = ['gridcell']
    columns_list = [columns_string.split(",")]

    api = GeoAPI(default_locations=df[['easting', 'northing']].values.tolist(),
                default_srid=3067, default_plot_ids=df.plot_id.values.tolist())

    hiladata = api.request_data(schema_list, tables_list, columns_list, batch_size=1000)

    # Return plot_ids from index to a column.
    hiladata.reset_index(inplace=True)
    hiladata = hiladata.drop_duplicates()
    
    return hiladata

hiladata = get_metsakeskus_data(df)
95/5:
### Get the metsäkeskus hila data
def get_metsakeskus_data(df):
    columns_string = """volumepine,volumespruce,volumedeciduous,volume,soiltype,fertilityclass,maingroup,subgroup,
    laserheight,laserdensity"""

    schema_list = ['metsakeskus_hila']
    tables_list = ['gridcell']
    columns_list = [columns_string.split(",")]

    api = GeoAPI(default_locations=df[['easting', 'northing']].values.tolist(),
                default_srid=3067, default_plot_ids=df.plot_id.values.tolist())

    hiladata = api.request_data(schema_list, tables_list, columns_list, batch_size=1000)

    # Return plot_ids from index to a column.
    hiladata.reset_index(inplace=True)
    hiladata = hiladata.drop_duplicates()
    
    return hiladata

hiladata = get_metsakeskus_data(df)
95/6:
### Get the metsäkeskus hila data
def get_metsakeskus_data(df):
    columns_string = """volumepine,volumespruce,volumedeciduous,volume,soiltype,fertilityclass,maingroup,subgroup,laserheight,laserdensity"""

    schema_list = ['metsakeskus_hila']
    tables_list = ['gridcell']
    columns_list = [columns_string.split(",")]

    api = GeoAPI(default_locations=df[['easting', 'northing']].values.tolist(),
                default_srid=3067, default_plot_ids=df.plot_id.values.tolist())

    hiladata = api.request_data(schema_list, tables_list, columns_list, batch_size=1000)

    # Return plot_ids from index to a column.
    hiladata.reset_index(inplace=True)
    hiladata = hiladata.drop_duplicates()
    
    return hiladata

hiladata = get_metsakeskus_data(df)
95/7:
# Drop duplicate plot ids - any chance of some kind of difference in data here? Merge scalar and hila data
hiladata = hiladata.drop_duplicates(subset="plot_id")
full_data = pd.merge(df, hiladata, on='plot_id', how='inner')
# use gridcell soiltype, not sample plots soil_type - 
# former is metsäkeskus and apparently more accurate? the latter is afaik from LUKE.
full_data = full_data.drop('soil_type', axis=1) # Drop LUKE soil type
# Save for use with train.py?
full_data.to_csv(input_path + "scalar_and_gridcell.csv", index=False)
95/8:
# Remove rows with NAs - gridcell data is missing from thousands of rows, aspect from about hundred.
full_data_nona = full_data.dropna()
95/9:
# Split to train and test. Uses metsäkeskus test set.
# full_data_high_pine = full_data_nona[full_data_nona.vol_pine > 100]
full_data_train, full_data_test = split_from_ids(full_data_nona)
95/10:
# Get only the features to be used
feature_columns = ['easting', 'northing', 'elevation', 'slope', 'aspect', 'tree_cover', 'leaf_type', 
                   'soiltype', 'fertilityclass', 'laserheight', 'laserdensity']

# Test just pine for now
target_columns = ['vol_pine']
# Get training data - 
features_train = full_data_train[feature_columns]
targets_train = full_data_train[target_columns]
# Get testing data
features_test = full_data_test[feature_columns]
targets_test = full_data_test[target_columns]
95/11:
# XGBoost. Try with just pine at first?
from xgboost import XGBRegressor

xgb = XGBRegressor(objective='reg:linear', nthread=-1)
xgb.fit(features_train, targets_train)
95/12:
# Ridge regression
from models import models_definition

ridge = models_definition.create_ridge(len(feature_columns), len(target_columns))
ridge.fit(features_train, targets_train)
95/13:
# Get predictions
xgb_preds = xgb.predict(features_test)
ridge_preds = ridge.predict(features_test)

# Metsäkeskus errors
target_metsakeskus_columns = ['volume']
95/14:
from metrics import model_metrics
print("Metsakeskus errors on the set:")
# compute_metrics requires a list, which is why it's wrapped this way. Warnings are related to ci_95 calcs
model_metrics.compute_metrics([targets_test.values], [full_data_test[target_metsakeskus_columns].values])

print("XGBoost prediction errors on the set:")
model_metrics.compute_metrics([targets_test.values], [np.expand_dims(xgb_preds,axis=1)])

print("Ridge prediction errors on the set:")
model_metrics.compute_metrics([targets_test.values], [ridge_preds])
95/15:
# Get predictions
xgb_preds = xgb.predict(features_test)
ridge_preds = ridge.predict(features_test)

# Metsäkeskus errors
target_metsakeskus_columns = ['volumepine']
95/16:
from metrics import model_metrics
print("Metsakeskus errors on the set:")
# compute_metrics requires a list, which is why it's wrapped this way. Warnings are related to ci_95 calcs
model_metrics.compute_metrics([targets_test.values], [full_data_test[target_metsakeskus_columns].values])

print("XGBoost prediction errors on the set:")
model_metrics.compute_metrics([targets_test.values], [np.expand_dims(xgb_preds,axis=1)])

print("Ridge prediction errors on the set:")
model_metrics.compute_metrics([targets_test.values], [ridge_preds])
95/17:
from metrics import model_metrics
print("Metsakeskus errors on the set:")
# compute_metrics requires a list, which is why it's wrapped this way. Warnings are related to ci_95 calcs
model_metrics.compute_metrics([targets_test.values], [full_data_test[target_metsakeskus_columns].values])

print("\n")
print("XGBoost prediction errors on the set:")
model_metrics.compute_metrics([targets_test.values], [np.expand_dims(xgb_preds,axis=1)])

print("\n")
print("Ridge prediction errors on the set:")
model_metrics.compute_metrics([targets_test.values], [ridge_preds])
96/1:
%load_ext autoreload
%autoreload 2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import json
import os
from tqdm import tqdm
import re
import sys
from sklearn.model_selection import train_test_split
import requests
pd.options.display.float_format = '{:,.2f}'.format
# Add path to where utils.py is so metrics can be imported
sys.path.insert(0, r'/home/tman/Work/linda-forestry-ml/species_prediction/regressors')
from data.data_loading import import_data, GeoAPI, split_from_ids
from data import data_loading
96/2:
input_path = r"/home/tman/Work/data/FIsampletiles"

df = pd.read_csv(os.path.join(input_path, 'groundtruth.csv'))
96/3:
### Get the metsäkeskus hila data
def get_metsakeskus_data(df):
    columns_string = """volumepine,volumespruce,volumedeciduous,volume,soiltype,fertilityclass,maingroup,subgroup,laserheight,laserdensity"""

    schema_list = ['metsakeskus_hila']
    tables_list = ['gridcell']
    columns_list = [columns_string.split(",")]

    api = GeoAPI(default_locations=df[['easting', 'northing']].values.tolist(),
                default_srid=3067, default_plot_ids=df.plot_id.values.tolist())

    hiladata = api.request_data(schema_list, tables_list, columns_list, batch_size=1000)

    # Return plot_ids from index to a column.
    hiladata.reset_index(inplace=True)
    hiladata = hiladata.drop_duplicates()
    
    return hiladata

hiladata = get_metsakeskus_data(df)
96/4:
# Drop duplicate plot ids - any chance of some kind of difference in data here? Merge scalar and hila data
hiladata = hiladata.drop_duplicates(subset="plot_id")
full_data = pd.merge(df, hiladata, on='plot_id', how='inner')
# use gridcell soiltype, not sample plots soil_type - 
# former is metsäkeskus and apparently more accurate? the latter is afaik from LUKE.
full_data = full_data.drop('soil_type', axis=1) # Drop LUKE soil type
# Save for use with train.py?
full_data.to_csv(input_path + "scalar_and_gridcell.csv", index=False)
96/5:
# Remove rows with NAs - gridcell data is missing from thousands of rows, aspect from about hundred.
full_data_nona = full_data.dropna()
# Filter so that only mature plots are used
full_data_nona = full_data_nona[full_data_nona['plot_type'].isin([1, 4])].reset_index(drop=True)
96/6:
# Split to train and test. Uses metsäkeskus test set.
# full_data_high_pine = full_data_nona[full_data_nona.vol_pine > 100]
full_data_train, full_data_test = split_from_ids(full_data_nona)
96/7:
# Get only the features to be used
feature_columns = ['easting', 'northing', 'elevation', 'slope', 'aspect', 'tree_cover', 'leaf_type', 
                   'soiltype', 'fertilityclass', 'laserheight', 'laserdensity']

# Test just pine for now
target_columns = ['vol_pine']
# Get training data - 
features_train = full_data_train[feature_columns]
targets_train = full_data_train[target_columns]
# Get testing data
features_test = full_data_test[feature_columns]
targets_test = full_data_test[target_columns]
96/8:
# XGBoost. Try with just pine at first?
from xgboost import XGBRegressor

xgb = XGBRegressor(objective='reg:linear', nthread=-1)
xgb.fit(features_train, targets_train)
96/9:
# Ridge regression
from models import models_definition

ridge = models_definition.create_ridge(len(feature_columns), len(target_columns))
ridge.fit(features_train, targets_train)
96/10:
# Get predictions
xgb_preds = xgb.predict(features_test)
ridge_preds = ridge.predict(features_test)

# Metsäkeskus errors
target_metsakeskus_columns = ['volumepine']
96/11:
from metrics import model_metrics
print("Metsakeskus errors on the set:")
# compute_metrics requires a list, which is why it's wrapped this way. Warnings are related to ci_95 calcs
model_metrics.compute_metrics([targets_test.values], [full_data_test[target_metsakeskus_columns].values])

print("\n")
print("XGBoost prediction errors on the set:")
model_metrics.compute_metrics([targets_test.values], [np.expand_dims(xgb_preds,axis=1)])

print("\n")
print("Ridge prediction errors on the set:")
model_metrics.compute_metrics([targets_test.values], [ridge_preds])
96/12:
# Remove rows with NAs - gridcell data is missing from thousands of rows, aspect from about hundred.
full_data_nona = full_data.dropna()
# Filter so that only mature plots are used
# full_data_nona = full_data_nona[full_data_nona['plot_type'].isin([1, 4])].reset_index(drop=True)
full_data_nona = full_data_nona[full_data_nona['vol_pine'] > 100]
96/13:
# Split to train and test. Uses metsäkeskus test set.
# full_data_high_pine = full_data_nona[full_data_nona.vol_pine > 100]
full_data_train, full_data_test = split_from_ids(full_data_nona)
96/14: len(full_data_test)
96/15: len(full_data_train)
96/16:
# Remove rows with NAs - gridcell data is missing from thousands of rows, aspect from about hundred.
full_data_nona = full_data.dropna()
# Filter so that only mature plots are used
# full_data_nona = full_data_nona[full_data_nona['plot_type'].isin([1, 4])].reset_index(drop=True)
full_data_nona = full_data_nona[full_data_nona['vol_total'] > 100]
96/17:
# Split to train and test. Uses metsäkeskus test set.
# full_data_high_pine = full_data_nona[full_data_nona.vol_pine > 100]
full_data_train, full_data_test = split_from_ids(full_data_nona)
96/18:
# Get only the features to be used
feature_columns = ['easting', 'northing', 'elevation', 'slope', 'aspect', 'tree_cover', 'leaf_type', 
                   'soiltype', 'fertilityclass', 'laserheight', 'laserdensity']

# Test just pine for now
target_columns = ['vol_pine']
# Get training data - 
features_train = full_data_train[feature_columns]
targets_train = full_data_train[target_columns]
# Get testing data
features_test = full_data_test[feature_columns]
targets_test = full_data_test[target_columns]
96/19: len(full_data_train)
96/20: len(full_data_test)
96/21:
# XGBoost. Try with just pine at first?
from xgboost import XGBRegressor

xgb = XGBRegressor(objective='reg:linear', nthread=-1)
xgb.fit(features_train, targets_train)
96/22:
# Ridge regression
from models import models_definition

ridge = models_definition.create_ridge(len(feature_columns), len(target_columns))
ridge.fit(features_train, targets_train)
96/23:
# Get predictions
xgb_preds = xgb.predict(features_test)
ridge_preds = ridge.predict(features_test)

# Metsäkeskus errors
target_metsakeskus_columns = ['volumepine']
96/24:
from metrics import model_metrics
print("Metsakeskus errors on the set:")
# compute_metrics requires a list, which is why it's wrapped this way. Warnings are related to ci_95 calcs
model_metrics.compute_metrics([targets_test.values], [full_data_test[target_metsakeskus_columns].values])

print("\n")
print("XGBoost prediction errors on the set:")
model_metrics.compute_metrics([targets_test.values], [np.expand_dims(xgb_preds,axis=1)])

print("\n")
print("Ridge prediction errors on the set:")
model_metrics.compute_metrics([targets_test.values], [ridge_preds])
97/1:
%load_ext autoreload
%autoreload 2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import json
import os
from tqdm import tqdm
import re
import sys
from sklearn.model_selection import train_test_split
import requests
pd.options.display.float_format = '{:,.2f}'.format
# Add path to where utils.py is so metrics can be imported
sys.path.insert(0, r'/home/tman/Work/linda-forestry-ml/species_prediction/regressors')
from data.data_loading import import_data, GeoAPI, split_from_ids
from data import data_loading
97/2:
input_path = r"/home/tman/Work/data/FIsampletiles"

df = pd.read_csv(os.path.join(input_path, 'groundtruth.csv'))
97/3:
### Get the metsäkeskus hila data
def get_metsakeskus_data(df):
    columns_string = """volumepine,volumespruce,volumedeciduous,volume,soiltype,fertilityclass,maingroup,subgroup,laserheight,laserdensity"""

    schema_list = ['metsakeskus_hila']
    tables_list = ['gridcell']
    columns_list = [columns_string.split(",")]

    api = GeoAPI(default_locations=df[['easting', 'northing']].values.tolist(),
                default_srid=3067, default_plot_ids=df.plot_id.values.tolist())

    hiladata = api.request_data(schema_list, tables_list, columns_list, batch_size=1000)

    # Return plot_ids from index to a column.
    hiladata.reset_index(inplace=True)
    hiladata = hiladata.drop_duplicates()
    
    return hiladata

hiladata = get_metsakeskus_data(df)
97/4:
# Drop duplicate plot ids - any chance of some kind of difference in data here? Merge scalar and hila data
hiladata = hiladata.drop_duplicates(subset="plot_id")
full_data = pd.merge(df, hiladata, on='plot_id', how='inner')
# use gridcell soiltype, not sample plots soil_type - 
# former is metsäkeskus and apparently more accurate? the latter is afaik from LUKE.
full_data = full_data.drop('soil_type', axis=1) # Drop LUKE soil type
# Save for use with train.py?
full_data.to_csv(input_path + "scalar_and_gridcell.csv", index=False)
97/5:
# Remove rows with NAs - gridcell data is missing from thousands of rows, aspect from about hundred.
full_data_nona = full_data.dropna()
# Filter so that only mature plots are used
# full_data_nona = full_data_nona[full_data_nona['plot_type'].isin([1, 4])].reset_index(drop=True)
# full_data_nona = full_data_nona[full_data_nona['vol_total'] > 100]
97/6:
# Split to train and test. Uses metsäkeskus test set.
# full_data_high_pine = full_data_nona[full_data_nona.vol_pine > 100]
full_data_train, full_data_test = split_from_ids(full_data_nona)
97/7:
# Get only the features to be used
feature_columns = ['easting', 'northing', 'elevation', 'slope', 'aspect', 'tree_cover', 'leaf_type', 
                   'soiltype', 'fertilityclass', 'laserheight', 'laserdensity']

# Test just pine for now
target_columns = ['vol_pine']
# Get training data - 
features_train = full_data_train[feature_columns]
targets_train = full_data_train[target_columns]
# Get testing data
features_test = full_data_test[feature_columns]
targets_test = full_data_test[target_columns]
97/8: len(full_data_test)
97/9:
# XGBoost. Try with just pine at first?
from xgboost import XGBRegressor

xgb = XGBRegressor(objective='reg:linear', nthread=-1)
xgb.fit(features_train, targets_train)
97/10:
# Ridge regression
from models import models_definition

ridge = models_definition.create_ridge(len(feature_columns), len(target_columns))
ridge.fit(features_train, targets_train)
97/11:
# Get predictions
xgb_preds = xgb.predict(features_test)
ridge_preds = ridge.predict(features_test)

# Metsäkeskus errors
target_metsakeskus_columns = ['volumepine']
97/12:
from metrics import model_metrics
print("Metsakeskus errors on the set:")
# compute_metrics requires a list, which is why it's wrapped this way. Warnings are related to ci_95 calcs
model_metrics.compute_metrics([targets_test.values], [full_data_test[target_metsakeskus_columns].values])

print("\n")
print("XGBoost prediction errors on the set:")
model_metrics.compute_metrics([targets_test.values], [np.expand_dims(xgb_preds,axis=1)])

print("\n")
print("Ridge prediction errors on the set:")
model_metrics.compute_metrics([targets_test.values], [ridge_preds])
100/1:
%load_ext autoreload
%autoreload 2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import json
import os
from tqdm import tqdm
import re
import sys
from sklearn.model_selection import train_test_split
import requests
pd.options.display.float_format = '{:,.2f}'.format
# Add path to where utils.py is so metrics can be imported
sys.path.insert(0, r'C:/Users/Teemu/Work/linda-forestry-ml/species_prediction/regressors')
from data.data_loading import import_data, GeoAPI, split_from_ids
from data import data_loading
100/2:
%load_ext autoreload
%autoreload 2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import json
import os
from tqdm import tqdm
import re
import sys
from sklearn.model_selection import train_test_split
import requests
pd.options.display.float_format = '{:,.2f}'.format
# Add path to where utils.py is so metrics can be imported
sys.path.insert(0, r'/home/tman/Work/linda-forestry-ml/species_prediction/regressors')
from data.data_loading import import_data, GeoAPI, split_from_ids
from data import data_loading
100/3:
input_path = r"/home/tmanTeemu/Work/data/FIsampletiles"
image_dir = "azure_tiles_cleaned"

df = pd.read_csv(os.path.join(input_path, 'groundtruth.csv'))
100/4:
input_path = r"/home/tman/Work/data/FIsampletiles"
image_dir = "azure_tiles_cleaned"

df = pd.read_csv(os.path.join(input_path, 'groundtruth.csv'))
100/5:
### Get the metsäkeskus hila data
def get_metsakeskus_data(df):
    # Should we use maingroup/subgroup?
    columns_string = """volumepine,volumespruce,volumedeciduous,volume,soiltype,fertilityclass,laserheight,laserdensity"""

    schema_list = ['metsakeskus_hila']
    tables_list = ['gridcell']
    columns_list = [columns_string.split(",")]

    api = GeoAPI(default_locations=df[['easting', 'northing']].values.tolist(),
                default_srid=3067, default_plot_ids=df.plot_id.values.tolist())

    hiladata = api.request_data(schema_list, tables_list, columns_list, batch_size=1000)

    # Return plot_ids from index to a column.
    hiladata.reset_index(inplace=True)
    hiladata = hiladata.drop_duplicates()
    
    return hiladata

hiladata = get_metsakeskus_data(df)
100/6:
onehot = True
if onehot:
    for column in ['soiltype', 'fertilityclass']:
        hiladata[column] = pd.Categorical(hiladata[column])
    
    hiladata = pd.get_dummies(hiladata)
100/7:
# Drop duplicate plot ids - any chance of some kind of difference in data here? Merge scalar and hila data
hiladata = hiladata.drop_duplicates(subset="plot_id")
full_data = pd.merge(df, hiladata, on='plot_id', how='inner')
# use gridcell soiltype, not sample plots soil_type - 
# former is metsäkeskus and apparently more accurate? the latter is afaik from LUKE.
full_data = full_data.drop('soil_type', axis=1) # Drop LUKE soil type
# Save for use with train.py?
full_data.to_csv(input_path + "scalar_and_gridcell.csv", index=False)
100/8:
# Remove rows with NAs - gridcell data is missing from thousands of rows, aspect from about hundred.
full_data_nona = full_data.dropna()
100/9:
# Split to train and test. Uses metsäkeskus test set.
# full_data_high_pine = full_data_nona[full_data_nona.vol_pine > 100]
full_data_train, full_data_test = split_from_ids(full_data_nona)
100/10:
# Get only the features to be used
feature_columns = ['easting', 'northing', 'elevation', 'slope', 'aspect', 'tree_cover', 'leaf_type', 
                   'soiltype', 'fertilityclass', 'laserheight', 'laserdensity']

target_columns = ['vol_pine', 'vol_spruce', 'vol_deciduous', 'vol_total']
# Get training data - 
features_train = full_data_train[feature_columns]
targets_train = full_data_train[target_columns]
# Get testing data
features_test = full_data_test[feature_columns]
targets_test = full_data_test[target_columns]
101/1:
%load_ext autoreload
%autoreload 2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import json
import os
from tqdm import tqdm
import re
import sys
from sklearn.model_selection import train_test_split
import requests
pd.options.display.float_format = '{:,.2f}'.format
# Add path to where utils.py is so metrics can be imported
sys.path.insert(0, r'/home/tman/Work/linda-forestry-ml/species_prediction/regressors')
from data.data_loading import import_data, GeoAPI, split_from_ids
from data import data_loading
101/2:
input_path = r"/home/tman/Work/data/FIsampletiles"
image_dir = "azure_tiles_cleaned"

df = pd.read_csv(os.path.join(input_path, 'groundtruth.csv'))
101/3:
### Get the metsäkeskus hila data
def get_metsakeskus_data(df):
    # Should we use maingroup/subgroup?
    columns_string = """volumepine,volumespruce,volumedeciduous,volume,soiltype,fertilityclass,laserheight,laserdensity"""

    schema_list = ['metsakeskus_hila']
    tables_list = ['gridcell']
    columns_list = [columns_string.split(",")]

    api = GeoAPI(default_locations=df[['easting', 'northing']].values.tolist(),
                default_srid=3067, default_plot_ids=df.plot_id.values.tolist())

    hiladata = api.request_data(schema_list, tables_list, columns_list, batch_size=1000)

    # Return plot_ids from index to a column.
    hiladata.reset_index(inplace=True)
    hiladata = hiladata.drop_duplicates()
    
    return hiladata

hiladata = get_metsakeskus_data(df)
101/4:
onehot = True
if onehot:
    for column in ['soiltype', 'fertilityclass']:
        hiladata[column] = pd.Categorical(hiladata[column])
    
    hiladata = pd.get_dummies(hiladata)
101/5:
# Drop duplicate plot ids - any chance of some kind of difference in data here? Merge scalar and hila data
hiladata = hiladata.drop_duplicates(subset="plot_id")
full_data = pd.merge(df, hiladata, on='plot_id', how='inner')
# use gridcell soiltype, not sample plots soil_type - 
# former is metsäkeskus and apparently more accurate? the latter is afaik from LUKE.
full_data = full_data.drop('soil_type', axis=1) # Drop LUKE soil type
# Save for use with train.py?
full_data.to_csv(input_path + "scalar_and_gridcell.csv", index=False)
101/6:
# Remove rows with NAs - gridcell data is missing from thousands of rows, aspect from about hundred.
full_data_nona = full_data.dropna()
101/7:
# Split to train and test. Uses metsäkeskus test set.
# full_data_high_pine = full_data_nona[full_data_nona.vol_pine > 100]
full_data_train, full_data_test = split_from_ids(full_data_nona)
101/8:
# Get training data - 
features_train = full_data_train.drop(target_columns)
targets_train = full_data_train[target_columns]
# Get testing data
features_test = full_data_test[feature_columns]
targets_test = full_data_test[target_columns]
101/9:
target_columns = ['vol_pine', 'vol_spruce', 'vol_deciduous', 'vol_total']
# Get training data - 
features_train = full_data_train.drop(target_columns)
targets_train = full_data_train[target_columns]
# Get testing data
features_test = full_data_test[feature_columns]
targets_test = full_data_test[target_columns]
101/10:
target_columns = ['vol_pine', 'vol_spruce', 'vol_deciduous', 'vol_total']
# Get training data - 
features_train = full_data_train.drop(target_columns)
targets_train = full_data_train[target_columns]
# Get testing data
features_test = full_data_test[feature_columns]
targets_test = full_data_test[target_columns]
101/11: full_data_train
101/12: full_data_train.columns
101/13:
target_columns = ['vol_pine', 'vol_spruce', 'vol_deciduous', 'vol_total']
# Get training data - 
features_train = full_data_train.drop(target_columns)
targets_train = full_data_train[target_columns]
# Get testing data
features_test = full_data_test[feature_columns]
targets_test = full_data_test[target_columns]
101/14:
target_columns = ['vol_pine', 'vol_spruce', 'vol_deciduous', 'vol_total']
# Get training data - 
features_train = full_data_train.drop(target_columns, axis=1)
targets_train = full_data_train[target_columns]
# Get testing data
features_test = full_data_test[feature_columns]
targets_test = full_data_test[target_columns]
101/15:
target_columns = ['vol_pine', 'vol_spruce', 'vol_deciduous', 'vol_total']
metsakeskus_pred_columns = ['volumepine','volumespruce','volumedeciduous','volume']
# Get training data - 
features_train = full_data_train.drop(target_columns + metsakeskus_pred_columns, axis=1)
targets_train = full_data_train[target_columns]
# Get testing data
features_test = full_data_train.drop(target_columns + metsakeskus_pred_columns, axis=1)
targets_test = full_data_test[target_columns]
101/16: features_train[:2]
102/1:
%load_ext autoreload
%autoreload 2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import json
import os
from tqdm import tqdm
import re
import sys
from sklearn.model_selection import train_test_split
import requests
pd.options.display.float_format = '{:,.2f}'.format
# Add path to where utils.py is so metrics can be imported
sys.path.insert(0, r'/home/tman/Work/linda-forestry-ml/species_prediction/regressors')
from data.data_loading import import_data, GeoAPI, split_from_ids
from data import data_loading
102/2:
input_path = r"/home/tman/Work/data/FIsampletiles"
image_dir = "azure_tiles_cleaned"

df = pd.read_csv(os.path.join(input_path, 'groundtruth.csv'))
102/3:
### Get the metsäkeskus hila data
def get_metsakeskus_data(df):
    # Should we use maingroup/subgroup?
    columns_string = """volumepine,volumespruce,volumedeciduous,volume,soiltype,fertilityclass,laserheight,laserdensity"""

    schema_list = ['metsakeskus_hila']
    tables_list = ['gridcell']
    columns_list = [columns_string.split(",")]

    api = GeoAPI(default_locations=df[['easting', 'northing']].values.tolist(),
                default_srid=3067, default_plot_ids=df.plot_id.values.tolist())

    hiladata = api.request_data(schema_list, tables_list, columns_list, batch_size=1000)

    # Return plot_ids from index to a column.
    hiladata.reset_index(inplace=True)
    hiladata = hiladata.drop_duplicates()
    
    return hiladata

hiladata = get_metsakeskus_data(df)
102/4: hiladata.columns
102/5:
onehot = False
if onehot:
    for column in ['soiltype', 'fertilityclass']:
        hiladata[column] = pd.Categorical(hiladata[column])
    
    hiladata = pd.get_dummies(hiladata)
102/6:
# Drop duplicate plot ids - any chance of some kind of difference in data here? Merge scalar and hila data
hiladata = hiladata.drop_duplicates(subset="plot_id")
full_data = pd.merge(df, hiladata, on='plot_id', how='inner')
# use gridcell soiltype, not sample plots soil_type - 
# former is metsäkeskus and apparently more accurate? the latter is afaik from LUKE.
full_data = full_data.drop('soil_type', axis=1) # Drop LUKE soil type
# Save for use with train.py?
full_data.to_csv(input_path + "scalar_and_gridcell.csv", index=False)
102/7: full_data_train.columns
102/8:
# Remove rows with NAs - gridcell data is missing from thousands of rows, aspect from about hundred.
full_data_nona = full_data.dropna()
102/9:
# Remove rows with NAs - gridcell data is missing from thousands of rows, aspect from about hundred.
full_data_nona = full_data.dropna()
full_data_nona.columns
103/1:
%load_ext autoreload
%autoreload 2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import json
import os
from tqdm import tqdm
import re
import sys
from sklearn.model_selection import train_test_split
import requests
pd.options.display.float_format = '{:,.2f}'.format
# Add path to where utils.py is so metrics can be imported
sys.path.insert(0, r'/home/tman/Work/linda-forestry-ml/species_prediction/regressors')
from data.data_loading import import_data, GeoAPI, split_from_ids
from data import data_loading
103/2:
input_path = r"/home/tman/Work/data/FIsampletiles"
image_dir = "azure_tiles_cleaned"

df = pd.read_csv(os.path.join(input_path, 'groundtruth.csv'))
103/3:
### Get the metsäkeskus hila data
def get_metsakeskus_data(df):
    # Should we use maingroup/subgroup?
    columns_string = """volumepine,volumespruce,volumedeciduous,volume,soiltype,fertilityclass,laserheight,laserdensity"""

    schema_list = ['metsakeskus_hila']
    tables_list = ['gridcell']
    columns_list = [columns_string.split(",")]

    api = GeoAPI(default_locations=df[['easting', 'northing']].values.tolist(),
                default_srid=3067, default_plot_ids=df.plot_id.values.tolist())

    hiladata = api.request_data(schema_list, tables_list, columns_list, batch_size=1000)

    # Return plot_ids from index to a column.
    hiladata.reset_index(inplace=True)
    hiladata = hiladata.drop_duplicates()
    
    return hiladata

hiladata = get_metsakeskus_data(df)
103/4:
# Drop duplicate plot ids - any chance of some kind of difference in data here? Merge scalar and hila data
hiladata = hiladata.drop_duplicates(subset="plot_id")
full_data = pd.merge(df, hiladata, on='plot_id', how='inner')
# use gridcell soiltype, not sample plots soil_type - 
# former is metsäkeskus and apparently more accurate? the latter is afaik from LUKE.
feature_columns = ['easting', 'northing', 'elevation', 'slope', 'aspect', 'tree_cover', 'leaf_type', 
                   'soiltype', 'fertilityclass', 'laserheight', 'laserdensity']
target_columns = ['vol_pine', 'vol_spruce', 'vol_deciduous', 'vol_total']
metsakeskus_pred_columns = ['volumepine','volumespruce','volumedeciduous','volume']
usable_columns = feature_columns + target_columns + metsakeskus_pred_columns
full_data = full_data[usable_columns]

full_data = full_data.drop('soil_type', axis=1) # Drop LUKE soil type
# Save for use with train.py?
full_data.to_csv(input_path + "scalar_and_gridcell.csv", index=False)
103/5:
# Drop duplicate plot ids - any chance of some kind of difference in data here? Merge scalar and hila data
hiladata = hiladata.drop_duplicates(subset="plot_id")
full_data = pd.merge(df, hiladata, on='plot_id', how='inner')
# use gridcell soiltype, not sample plots soil_type - 
# former is metsäkeskus and apparently more accurate? the latter is afaik from LUKE.
feature_columns = ['easting', 'northing', 'elevation', 'slope', 'aspect', 'tree_cover', 'leaf_type', 
                   'soiltype', 'fertilityclass', 'laserheight', 'laserdensity']
target_columns = ['vol_pine', 'vol_spruce', 'vol_deciduous', 'vol_total']
metsakeskus_pred_columns = ['volumepine','volumespruce','volumedeciduous','volume']
usable_columns = feature_columns + target_columns + metsakeskus_pred_columns
full_data = full_data[usable_columns]

# Save for use with train.py?
full_data.to_csv(input_path + "scalar_and_gridcell.csv", index=False)
103/6: full_data.columns
103/7:
# Remove rows with NAs - gridcell data is missing from thousands of rows, aspect from about hundred.
full_data_nona = full_data.dropna()
103/8:
onehot = True
if onehot:
    for column in ['soiltype', 'fertilityclass']:
        hiladata[column] = pd.Categorical(hiladata[column])
    
    hiladata = pd.get_dummies(hiladata)
103/9:
# Drop duplicate plot ids - any chance of some kind of difference in data here? Merge scalar and hila data
hiladata = hiladata.drop_duplicates(subset="plot_id")
full_data = pd.merge(df, hiladata, on='plot_id', how='inner')
# use gridcell soiltype, not sample plots soil_type - 
# former is metsäkeskus and apparently more accurate? the latter is afaik from LUKE.
feature_columns = ['easting', 'northing', 'elevation', 'slope', 'aspect', 'tree_cover', 'leaf_type', 
                   'soiltype', 'fertilityclass', 'laserheight', 'laserdensity']
target_columns = ['vol_pine', 'vol_spruce', 'vol_deciduous', 'vol_total']
metsakeskus_pred_columns = ['volumepine','volumespruce','volumedeciduous','volume']
usable_columns = feature_columns + target_columns + metsakeskus_pred_columns
full_data = full_data[usable_columns]

# Save for use with train.py?
full_data.to_csv(input_path + "scalar_and_gridcell.csv", index=False)

onehot = True
if onehot:
    for column in ['soiltype', 'fertilityclass']:
        full_data[column] = pd.Categorical(full_data[column])
    
    full_data = pd.get_dummies(full_data)
104/1:
%load_ext autoreload
%autoreload 2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import json
import os
from tqdm import tqdm
import re
import sys
from sklearn.model_selection import train_test_split
import requests
pd.options.display.float_format = '{:,.2f}'.format
# Add path to where utils.py is so metrics can be imported
sys.path.insert(0, r'/home/tman/Work/linda-forestry-ml/species_prediction/regressors')
from data.data_loading import import_data, GeoAPI, split_from_ids
from data import data_loading
104/2:
input_path = r"/home/tman/Work/data/FIsampletiles"
image_dir = "azure_tiles_cleaned"

df = pd.read_csv(os.path.join(input_path, 'groundtruth.csv'))
104/3:
### Get the metsäkeskus hila data
def get_metsakeskus_data(df):
    # Should we use maingroup/subgroup?
    columns_string = """volumepine,volumespruce,volumedeciduous,volume,soiltype,fertilityclass,laserheight,laserdensity"""

    schema_list = ['metsakeskus_hila']
    tables_list = ['gridcell']
    columns_list = [columns_string.split(",")]

    api = GeoAPI(default_locations=df[['easting', 'northing']].values.tolist(),
                default_srid=3067, default_plot_ids=df.plot_id.values.tolist())

    hiladata = api.request_data(schema_list, tables_list, columns_list, batch_size=1000)

    # Return plot_ids from index to a column.
    hiladata.reset_index(inplace=True)
    hiladata = hiladata.drop_duplicates()
    
    return hiladata

hiladata = get_metsakeskus_data(df)
104/4:
# Drop duplicate plot ids - any chance of some kind of difference in data here? Merge scalar and hila data
hiladata = hiladata.drop_duplicates(subset="plot_id")
full_data = pd.merge(df, hiladata, on='plot_id', how='inner')
# use gridcell soiltype, not sample plots soil_type - 
# former is metsäkeskus and apparently more accurate? the latter is afaik from LUKE.
feature_columns = ['easting', 'northing', 'elevation', 'slope', 'aspect', 'tree_cover', 'leaf_type', 
                   'soiltype', 'fertilityclass', 'laserheight', 'laserdensity']
target_columns = ['vol_pine', 'vol_spruce', 'vol_deciduous', 'vol_total']
metsakeskus_pred_columns = ['volumepine','volumespruce','volumedeciduous','volume']
usable_columns = feature_columns + target_columns + metsakeskus_pred_columns
full_data = full_data[usable_columns]

# Save for use with train.py?
full_data.to_csv(input_path + "scalar_and_gridcell.csv", index=False)

onehot = True
if onehot:
    for column in ['soiltype', 'fertilityclass']:
        full_data[column] = pd.Categorical(full_data[column])
    
    full_data = pd.get_dummies(full_data)
104/5:
# Remove rows with NAs - gridcell data is missing from thousands of rows, aspect from about hundred.
full_data_nona = full_data.dropna()
104/6:
# Split to train and test. Uses metsäkeskus test set.
# full_data_high_pine = full_data_nona[full_data_nona.vol_pine > 100]
full_data_train, full_data_test = split_from_ids(full_data_nona)
105/1:
%load_ext autoreload
%autoreload 2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import json
import os
from tqdm import tqdm
import re
import sys
from sklearn.model_selection import train_test_split
import requests
pd.options.display.float_format = '{:,.2f}'.format
# Add path to where utils.py is so metrics can be imported
sys.path.insert(0, r'/home/tman/Work/linda-forestry-ml/species_prediction/regressors')
from data.data_loading import import_data, GeoAPI, split_from_ids
from data import data_loading
105/2:
input_path = r"/home/tman/Work/data/FIsampletiles"
image_dir = "azure_tiles_cleaned"

df = pd.read_csv(os.path.join(input_path, 'groundtruth.csv'))
105/3:
### Get the metsäkeskus hila data
def get_metsakeskus_data(df):
    # Should we use maingroup/subgroup?
    columns_string = """volumepine,volumespruce,volumedeciduous,volume,soiltype,fertilityclass,laserheight,laserdensity"""

    schema_list = ['metsakeskus_hila']
    tables_list = ['gridcell']
    columns_list = [columns_string.split(",")]

    api = GeoAPI(default_locations=df[['easting', 'northing']].values.tolist(),
                default_srid=3067, default_plot_ids=df.plot_id.values.tolist())

    hiladata = api.request_data(schema_list, tables_list, columns_list, batch_size=1000)

    # Return plot_ids from index to a column.
    hiladata.reset_index(inplace=True)
    hiladata = hiladata.drop_duplicates()
    
    return hiladata

hiladata = get_metsakeskus_data(df)
105/4:
# Drop duplicate plot ids - any chance of some kind of difference in data here? Merge scalar and hila data
hiladata = hiladata.drop_duplicates(subset="plot_id")
full_data = pd.merge(df, hiladata, on='plot_id', how='inner')
# use gridcell soiltype, not sample plots soil_type - 
# former is metsäkeskus and apparently more accurate? the latter is afaik from LUKE.
feature_columns = ['plot_id', 'easting', 'northing', 'elevation', 'slope', 'aspect', 'tree_cover', 'leaf_type', 
                   'soiltype', 'fertilityclass', 'laserheight', 'laserdensity']
target_columns = ['vol_pine', 'vol_spruce', 'vol_deciduous', 'vol_total']
metsakeskus_pred_columns = ['volumepine','volumespruce','volumedeciduous','volume']
usable_columns = feature_columns + target_columns + metsakeskus_pred_columns
full_data = full_data[usable_columns]

# Save for use with train.py?
full_data.to_csv(input_path + "scalar_and_gridcell.csv", index=False)

onehot = True
if onehot:
    for column in ['soiltype', 'fertilityclass']:
        full_data[column] = pd.Categorical(full_data[column])
    
    full_data = pd.get_dummies(full_data)
105/5:
# Drop duplicate plot ids - any chance of some kind of difference in data here? Merge scalar and hila data
hiladata = hiladata.drop_duplicates(subset="plot_id")
full_data = pd.merge(df, hiladata, on='plot_id', how='inner')
# use gridcell soiltype, not sample plots soil_type - 
# former is metsäkeskus and apparently more accurate? the latter is afaik from LUKE.
feature_columns = ['easting', 'northing', 'elevation', 'slope', 'aspect', 'tree_cover', 'leaf_type', 
                   'soiltype', 'fertilityclass', 'laserheight', 'laserdensity']
target_columns = ['vol_pine', 'vol_spruce', 'vol_deciduous', 'vol_total']
metsakeskus_pred_columns = ['volumepine','volumespruce','volumedeciduous','volume']
usable_columns = feature_columns + target_columns + metsakeskus_pred_columns
full_data = full_data[['plot_id'] + usable_columns]

# Save for use with train.py?
full_data.to_csv(input_path + "scalar_and_gridcell.csv", index=False)

onehot = True
if onehot:
    for column in ['soiltype', 'fertilityclass']:
        full_data[column] = pd.Categorical(full_data[column])
    
    full_data = pd.get_dummies(full_data)
105/6:
# Remove rows with NAs - gridcell data is missing from thousands of rows, aspect from about hundred.
full_data_nona = full_data.dropna()
105/7:
# Split to train and test. Uses metsäkeskus test set.
# full_data_high_pine = full_data_nona[full_data_nona.vol_pine > 100]
full_data_train, full_data_test = split_from_ids(full_data_nona)
105/8: full_data_train.columns
105/9:
# Get only the features to be used
target_columns = ['vol_pine', 'vol_spruce', 'vol_deciduous', 'vol_total']
# Get training data - 
features_train = full_data_train.drop(target_columns + metsakeskus_pred_columns)
targets_train = full_data_train[target_columns]
# Get testing data
features_test = full_data_test.drop(target_columns + metsakeskus_pred_columns)
targets_test = full_data_test[target_columns]
105/10:
# Get only the features to be used
# Get training data - 
features_train = full_data_train.drop(target_columns + metsakeskus_pred_columns)
targets_train = full_data_train[target_columns]
# Get testing data
features_test = full_data_test.drop(target_columns + metsakeskus_pred_columns)
targets_test = full_data_test[target_columns]
105/11:
# Drop duplicate plot ids - any chance of some kind of difference in data here? Merge scalar and hila data
hiladata = hiladata.drop_duplicates(subset="plot_id")
full_data = pd.merge(df, hiladata, on='plot_id', how='inner')
# use gridcell soiltype, not sample plots soil_type - 
# former is metsäkeskus and apparently more accurate? the latter is afaik from LUKE.
feature_columns = ['easting', 'northing', 'elevation', 'slope', 'aspect', 'tree_cover', 'leaf_type', 
                   'soiltype', 'fertilityclass', 'laserheight', 'laserdensity']
target_columns = ['vol_pine', 'vol_spruce', 'vol_deciduous', 'vol_total']
metsakeskus_pred_columns = ['volumepine','volumespruce','volumedeciduous','volume']
usable_columns = feature_columns + target_columns + metsakeskus_pred_columns
full_data = full_data[['plot_id'] + usable_columns]

# Save for use with train.py?
full_data.to_csv(input_path + "scalar_and_gridcell.csv", index=False)

onehot = True
if onehot:
    for column in ['soiltype', 'fertilityclass']:
        full_data[column] = pd.Categorical(full_data[column])
    
    full_data = pd.get_dummies(full_data)
105/12:
# Remove rows with NAs - gridcell data is missing from thousands of rows, aspect from about hundred.
full_data_nona = full_data.dropna()
105/13:
# Split to train and test. Uses metsäkeskus test set.
# full_data_high_pine = full_data_nona[full_data_nona.vol_pine > 100]
full_data_train, full_data_test = split_from_ids(full_data_nona)
105/14: full_data_train.columns
105/15:
# Get only the features to be used
# Get training data - 
features_train = full_data_train.drop(target_columns + metsakeskus_pred_columns)
targets_train = full_data_train[target_columns]
# Get testing data
features_test = full_data_test.drop(target_columns + metsakeskus_pred_columns)
targets_test = full_data_test[target_columns]
105/16:
# Get only the features to be used
# Get training data - 
features_train = full_data_train.drop(target_columns)
targets_train = full_data_train[target_columns]
# Get testing data
features_test = full_data_test.drop(target_columns)
targets_test = full_data_test[target_columns]
105/17: full_data_train.columns
105/18:
# Get only the features to be used
# Get training data - 
features_train = full_data_train.drop(target_columns + metsakeskus_pred_columns, axis=1)
targets_train = full_data_train[target_columns]
# Get testing data
features_test = full_data_test.drop(target_columns + metsakeskus_pred_columns, axis=1)
targets_test = full_data_test[target_columns]
105/19: features_train.columns
105/20:
# Get only the features to be used
# Get training data - 
features_train = full_data_train.drop(target_columns + metsakeskus_pred_columns + ['plot_id'], axis=1)
targets_train = full_data_train[target_columns]
# Get testing data
features_test = full_data_test.drop(target_columns + metsakeskus_pred_columns + ['plot_id'], axis=1)
targets_test = full_data_test[target_columns]
105/21: features_train.columns
105/22:
# XGBoost. Try with just pine at first?
from xgboost import XGBRegressor
from models import models_definition
from sklearn.metrics import mean_squared_error

target_to_metsakeskus = {
    'vol_pine': 'volumepine',
    'vol_spruce': 'volumespruce',
    'vol_deciduous': 'volumedeciduous',
    'vol_total': 'volume',
}

for col in targets_train.columns:
    y_train, y_test = targets_train[col].values, targets_test[col].values
    X_train, X_test = features_train.values, features_test.values
    xgb = XGBRegressor(objective='reg:linear', nthread=-1)
    xgb.fit(X_train, y_train)
    pred = xgb.predict(X_test)
    metsakeskus_pred = full_data_test[target_to_metsakeskus[col]].values
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    y_mean = y_test.mean()
    nrmse = rmse / y_mean * 100
    nrmse_metsakeskus = np.sqrt(mean_squared_error(y_test, metsakeskus_pred)) / y_mean * 100
    
    print('Mean for {}: {:.5f}'.format(col, y_mean))
    print('NRMSE for {}: {:.5f}'.format(col, nrmse))
    print('Metsäkeskus NRMSE for {}: {:.5f}'.format(col, nrmse_metsakeskus))
106/1:
%load_ext autoreload
%autoreload 2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import json
import os
from tqdm import tqdm
import re
import sys
from sklearn.model_selection import train_test_split
import requests
pd.options.display.float_format = '{:,.2f}'.format
# Add path to where utils.py is so metrics can be imported
sys.path.insert(0, r'/home/tman/Work/linda-forestry-ml/species_prediction/regressors')
from data.data_loading import import_data, GeoAPI, split_from_ids
from data import data_loading
106/2:
input_path = r"/home/tman/Work/data/FIsampletiles"
image_dir = "azure_tiles_cleaned"

df = pd.read_csv(os.path.join(input_path, 'groundtruth.csv'))
106/3:
### Get the metsäkeskus hila data
def get_metsakeskus_data(df):
    # Should we use maingroup/subgroup?
    columns_string = """volumepine,volumespruce,volumedeciduous,volume,soiltype,fertilityclass,laserheight,laserdensity"""

    schema_list = ['metsakeskus_hila']
    tables_list = ['gridcell']
    columns_list = [columns_string.split(",")]

    api = GeoAPI(default_locations=df[['easting', 'northing']].values.tolist(),
                default_srid=3067, default_plot_ids=df.plot_id.values.tolist())

    hiladata = api.request_data(schema_list, tables_list, columns_list, batch_size=1000)

    # Return plot_ids from index to a column.
    hiladata.reset_index(inplace=True)
    hiladata = hiladata.drop_duplicates()
    
    return hiladata

hiladata = get_metsakeskus_data(df)
106/4:
# Drop duplicate plot ids - any chance of some kind of difference in data here? Merge scalar and hila data
hiladata = hiladata.drop_duplicates(subset="plot_id")
full_data = pd.merge(df, hiladata, on='plot_id', how='inner')
# use gridcell soiltype, not sample plots soil_type - 
# former is metsäkeskus and apparently more accurate? the latter is afaik from LUKE.
full_data = full_data.drop('soil_type', axis=1) # Drop LUKE soil type
# Save for use with train.py?
full_data.to_csv(input_path + "scalar_and_gridcell.csv", index=False)
106/5:
# Remove rows with NAs - gridcell data is missing from thousands of rows, aspect from about hundred.
full_data_nona = full_data.dropna()
106/6:
# Split to train and test. Uses metsäkeskus test set.
# full_data_high_pine = full_data_nona[full_data_nona.vol_pine > 100]
full_data_train, full_data_test = split_from_ids(full_data_nona)
106/7:
# Get only the features to be used
# Get training data - 
feature_columns = ['easting', 'northing', 'elevation', 'slope', 'aspect', 'tree_cover', 'leaf_type', 
                   'soiltype', 'fertilityclass', 'laserheight', 'laserdensity']
target_columns = ['vol_pine', 'vol_spruce', 'vol_deciduous', 'vol_total']
metsakeskus_pred_columns = ['volumepine','volumespruce','volumedeciduous','volume']

features_train = full_data_train[feature_columns]
targets_train = full_data_train[target_columns]
# Get testing data
features_test = full_data_test[feature_columns]
targets_test = full_data_test[target_columns]

onehot = True
if onehot:
    for column in ['soiltype', 'fertilityclass']:
        features_train[column] = pd.Categorical(features_train[column])
        features_test[column] = pd.Categorical(features_test[column])
    
    features_train = pd.get_dummies(features_train)
    features_test = pd.get_dummies(features_test)
107/1:
%load_ext autoreload
%autoreload 2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import json
import os
from tqdm import tqdm
import re
import sys
from sklearn.model_selection import train_test_split
import requests
pd.options.display.float_format = '{:,.2f}'.format
# Add path to where utils.py is so metrics can be imported
sys.path.insert(0, r'/home/tman/Work/linda-forestry-ml/species_prediction/regressors')
from data.data_loading import import_data, GeoAPI, split_from_ids
from data import data_loading
107/2:
input_path = r"/home/tman/Work/data/FIsampletiles"
image_dir = "azure_tiles_cleaned"

df = pd.read_csv(os.path.join(input_path, 'groundtruth.csv'))
107/3:
### Get the metsäkeskus hila data
def get_metsakeskus_data(df):
    # Should we use maingroup/subgroup?
    columns_string = """volumepine,volumespruce,volumedeciduous,volume,soiltype,fertilityclass,laserheight,laserdensity"""

    schema_list = ['metsakeskus_hila']
    tables_list = ['gridcell']
    columns_list = [columns_string.split(",")]

    api = GeoAPI(default_locations=df[['easting', 'northing']].values.tolist(),
                default_srid=3067, default_plot_ids=df.plot_id.values.tolist())

    hiladata = api.request_data(schema_list, tables_list, columns_list, batch_size=1000)

    # Return plot_ids from index to a column.
    hiladata.reset_index(inplace=True)
    hiladata = hiladata.drop_duplicates()
    
    return hiladata

hiladata = get_metsakeskus_data(df)
107/4:
# Drop duplicate plot ids - any chance of some kind of difference in data here? Merge scalar and hila data
hiladata = hiladata.drop_duplicates(subset="plot_id")
full_data = pd.merge(df, hiladata, on='plot_id', how='inner')
# use gridcell soiltype, not sample plots soil_type - 
# former is metsäkeskus and apparently more accurate? the latter is afaik from LUKE.
full_data = full_data.drop('soil_type', axis=1) # Drop LUKE soil type
# Set these columns as categorical in case we try onehot later
for column in ['soiltype', 'fertilityclass']:
    full_data[column] = pd.Categorical(full_data[column])
# Save for use with train.py?
full_data.to_csv(input_path + "scalar_and_gridcell.csv", index=False)
107/5:
# Remove rows with NAs - gridcell data is missing from thousands of rows, aspect from about hundred.
full_data_nona = full_data.dropna()
107/6:
# Split to train and test. Uses metsäkeskus test set.
# full_data_high_pine = full_data_nona[full_data_nona.vol_pine > 100]
full_data_train, full_data_test = split_from_ids(full_data_nona)
107/7:
# Get only the features to be used
# Get training data - 
feature_columns = ['easting', 'northing', 'elevation', 'slope', 'aspect', 'tree_cover', 'leaf_type', 
                   'soiltype', 'fertilityclass', 'laserheight', 'laserdensity']
target_columns = ['vol_pine', 'vol_spruce', 'vol_deciduous', 'vol_total']
metsakeskus_pred_columns = ['volumepine','volumespruce','volumedeciduous','volume']

features_train = full_data_train[feature_columns]
targets_train = full_data_train[target_columns]
# Get testing data
features_test = full_data_test[feature_columns]
targets_test = full_data_test[target_columns]

onehot = True
if onehot:
    features_train = pd.get_dummies(features_train)
    features_test = pd.get_dummies(features_test)
107/8: features_train
107/9:
# cv search of best model - actually gives worse results than default?

from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, make_scorer

param_distributions = {'max_depth': [6,8,10],
                       'learning_rate': [0.1,0.01,0.001,0.0001],
                       'n_estimators': [100, 200, 300, 400],
                       'min_child_weight': [2, 8, 15, 25],
                       'colsample_bytree': [1, 0.8, 0.5],
                       'subsample': [0.6, 0.8],
                       'reg_alpha': [0.01, 0.08, 0.2],
                       'colsample_bylevel': [0.6, 0.8],
                       'reg_lambda': [0.7, 0.8, 0.95]
                       }

# model = XGBRegressor(max_depth=5, learning_rate=0.1, n_estimators=500, n_jobs=3)

search = RandomizedSearchCV(XGBRegressor(), scoring=make_scorer(mean_squared_error), param_distributions=param_distributions, 
                   n_jobs=-1, cv=5, verbose=True, n_iter=50)

search.fit(features_train, targets_train)
best_params = search.best_params_
model = XGBRegressor(**best_params)
model.fit(features_train, targets_train)
107/10:
# XGBoost. Try with just pine at first?
from xgboost import XGBRegressor
from models import models_definition
from sklearn.metrics import mean_squared_error

target_to_metsakeskus = {
    'vol_pine': 'volumepine',
    'vol_spruce': 'volumespruce',
    'vol_deciduous': 'volumedeciduous',
    'vol_total': 'volume',
}

for col in targets_train.columns:
    y_train, y_test = targets_train[col].values, targets_test[col].values
    X_train, X_test = features_train.values, features_test.values
    xgb = XGBRegressor(objective='reg:linear', nthread=-1)
    xgb.fit(X_train, y_train)
    pred = xgb.predict(X_test)
    metsakeskus_pred = full_data_test[target_to_metsakeskus[col]].values
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    y_mean = y_test.mean()
    nrmse = rmse / y_mean * 100
    nrmse_metsakeskus = np.sqrt(mean_squared_error(y_test, metsakeskus_pred)) / y_mean * 100
    
    print('Mean for {}: {:.5f}'.format(col, y_mean))
    print('NRMSE for {}: {:.5f}'.format(col, nrmse))
    print('Metsäkeskus NRMSE for {}: {:.5f}'.format(col, nrmse_metsakeskus))
112/1:
%load_ext autoreload
%autoreload 2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import json
import os
from tqdm import tqdm
import re
import sys
from sklearn.model_selection import train_test_split
import requests
pd.options.display.float_format = '{:,.2f}'.format
# Add path to where utils.py is so metrics can be imported
sys.path.insert(0, r'../regressors')
from data.data_loading import import_data, GeoAPI, split_from_ids
from data import data_loading
112/2:
input_path = r"../../../data/FIsampletiles"
image_dir = "azure_tiles_cleaned"

df = pd.read_csv(os.path.join(input_path, 'groundtruth.csv'))
api = GeoAPI(default_locations=df[['easting', 'northing']].values.tolist(),
                default_srid=3067, default_plot_ids=df.plot_id.values.tolist())
112/3:
### Get the metsäkeskus hila data

def get_metsakeskus_data()
    columns_string = [["volumepine","volumespruce","volumedeciduous","volume","soiltype","fertilityclass","maingroup",
                      "subgroup","laserheight","laserdensity"]]

    schema_list = ['metsakeskus_hila']
    tables_list = ['gridcell']
    columns_list = [columns_string.split(",")]

    data = api.request_data(schema_list, tables_list, columns_list, batch_size=1000)

    # Return plot_ids from index to a column.
    data.reset_index(inplace=True)
    data = data.drop_duplicates(subset='plot_id')
    
    return hiladata

def get_copernicus_data():
    data = api.request_data(data_groups=['soilgrids'])
    data = data.reset_index()
    data = data.drop_duplicates(subset='plot_id')
        
    return data

def get_soilgrids_data():
    data = api.request_data(data_groups=['soilgrids'])
    data = data.reset_index()
    data = data.drop_duplicates(subset='plot_id')
        
    return data

def get_climate_data():
    data = api.request_data(data_groups=['climate_data '])
    data = data.reset_index()
    data = data.drop_duplicates(subset='plot_id')
        
    return data

metsakeskus_data = get_metsakeskus_data()
copernicus_data = get_copernicus_data()
soilgrids_data = get_soilgrids_data()
climate_data = get_climate_data()
112/4:
### Get the metsäkeskus hila data

def get_metsakeskus_data():
    columns_string = [["volumepine","volumespruce","volumedeciduous","volume","soiltype","fertilityclass","maingroup",
                      "subgroup","laserheight","laserdensity"]]

    schema_list = ['metsakeskus_hila']
    tables_list = ['gridcell']
    columns_list = [columns_string.split(",")]

    data = api.request_data(schema_list, tables_list, columns_list, batch_size=1000)

    # Return plot_ids from index to a column.
    data.reset_index(inplace=True)
    data = data.drop_duplicates(subset='plot_id')
    
    return hiladata

def get_copernicus_data():
    data = api.request_data(data_groups=['soilgrids'])
    data = data.reset_index()
    data = data.drop_duplicates(subset='plot_id')
        
    return data

def get_soilgrids_data():
    data = api.request_data(data_groups=['soilgrids'])
    data = data.reset_index()
    data = data.drop_duplicates(subset='plot_id')
        
    return data

def get_climate_data():
    data = api.request_data(data_groups=['climate_data '])
    data = data.reset_index()
    data = data.drop_duplicates(subset='plot_id')
        
    return data

metsakeskus_data = get_metsakeskus_data()
copernicus_data = get_copernicus_data()
soilgrids_data = get_soilgrids_data()
climate_data = get_climate_data()
112/5:
### Get the metsäkeskus hila data

def get_metsakeskus_data():
    columns_string = [["volumepine","volumespruce","volumedeciduous","volume","soiltype","fertilityclass","maingroup",
                      "subgroup","laserheight","laserdensity"]]

    schema_list = ['metsakeskus_hila']
    tables_list = ['gridcell']

    data = api.request_data(schema_list, tables_list, columns_list, batch_size=1000)

    # Return plot_ids from index to a column.
    data.reset_index(inplace=True)
    data = data.drop_duplicates(subset='plot_id')
    
    return hiladata

def get_copernicus_data():
    data = api.request_data(data_groups=['soilgrids'])
    data = data.reset_index()
    data = data.drop_duplicates(subset='plot_id')
        
    return data

def get_soilgrids_data():
    data = api.request_data(data_groups=['soilgrids'])
    data = data.reset_index()
    data = data.drop_duplicates(subset='plot_id')
        
    return data

def get_climate_data():
    data = api.request_data(data_groups=['climate_data '])
    data = data.reset_index()
    data = data.drop_duplicates(subset='plot_id')
        
    return data

metsakeskus_data = get_metsakeskus_data()
copernicus_data = get_copernicus_data()
soilgrids_data = get_soilgrids_data()
climate_data = get_climate_data()
112/6:
### Get the metsäkeskus hila data

def get_metsakeskus_data():
    columns_list = [["volumepine","volumespruce","volumedeciduous","volume","soiltype","fertilityclass","maingroup",
                      "subgroup","laserheight","laserdensity"]]

    schema_list = ['metsakeskus_hila']
    tables_list = ['gridcell']

    data = api.request_data(schema_list, tables_list, columns_list, batch_size=1000)

    # Return plot_ids from index to a column.
    data.reset_index(inplace=True)
    data = data.drop_duplicates(subset='plot_id')
    
    return hiladata

def get_copernicus_data():
    data = api.request_data(data_groups=['soilgrids'])
    data = data.reset_index()
    data = data.drop_duplicates(subset='plot_id')
        
    return data

def get_soilgrids_data():
    data = api.request_data(data_groups=['soilgrids'])
    data = data.reset_index()
    data = data.drop_duplicates(subset='plot_id')
        
    return data

def get_climate_data():
    data = api.request_data(data_groups=['climate_data '])
    data = data.reset_index()
    data = data.drop_duplicates(subset='plot_id')
        
    return data

metsakeskus_data = get_metsakeskus_data()
copernicus_data = get_copernicus_data()
soilgrids_data = get_soilgrids_data()
climate_data = get_climate_data()
112/7:
### Get the metsäkeskus hila data

def get_metsakeskus_data():
    columns_list = [["volumepine","volumespruce","volumedeciduous","volume","soiltype","fertilityclass","maingroup",
                      "subgroup","laserheight","laserdensity"]]

    schema_list = ['metsakeskus_hila']
    tables_list = ['gridcell']

    data = api.request_data(schema_list, tables_list, columns_list, batch_size=1000)

    # Return plot_ids from index to a column.
    data.reset_index(inplace=True)
    data = data.drop_duplicates(subset='plot_id')
    
    return data

def get_copernicus_data():
    data = api.request_data(data_groups=['soilgrids'])
    data = data.reset_index()
    data = data.drop_duplicates(subset='plot_id')
        
    return data

def get_soilgrids_data():
    data = api.request_data(data_groups=['soilgrids'])
    data = data.reset_index()
    data = data.drop_duplicates(subset='plot_id')
        
    return data

def get_climate_data():
    data = api.request_data(data_groups=['climate_data '])
    data = data.reset_index()
    data = data.drop_duplicates(subset='plot_id')
        
    return data

metsakeskus_data = get_metsakeskus_data()
copernicus_data = get_copernicus_data()
soilgrids_data = get_soilgrids_data()
climate_data = get_climate_data()
112/8:
### Get the metsäkeskus hila data

def get_metsakeskus_data():
    columns_list = [["volumepine","volumespruce","volumedeciduous","volume","soiltype","fertilityclass","maingroup",
                      "subgroup","laserheight","laserdensity"]]

    schema_list = ['metsakeskus_hila']
    tables_list = ['gridcell']

    data = api.request_data(schema_list, tables_list, columns_list, batch_size=1000)

    # Return plot_ids from index to a column.
    data.reset_index(inplace=True)
    data = data.drop_duplicates(subset='plot_id')
    
    return data

def get_copernicus_data():
    data = api.request_data(data_groups=['soilgrids'])
    data = data.reset_index()
    data = data.drop_duplicates(subset='plot_id')
        
    return data

def get_soilgrids_data():
    data = api.request_data(data_groups=['soilgrids'])
    data = data.reset_index()
    data = data.drop_duplicates(subset='plot_id')
        
    return data

def get_climate_data():
    data = api.request_data(data_groups=['climate_data'])
    data = data.reset_index()
    data = data.drop_duplicates(subset='plot_id')
        
    return data

#metsakeskus_data = get_metsakeskus_data()
#copernicus_data = get_copernicus_data()
#soilgrids_data = get_soilgrids_data()
climate_data = get_climate_data()
112/9:
### Get the metsäkeskus hila data

def get_metsakeskus_data():
    columns_list = [["volumepine","volumespruce","volumedeciduous","volume","soiltype","fertilityclass","maingroup",
                      "subgroup","laserheight","laserdensity"]]

    schema_list = ['metsakeskus_hila']
    tables_list = ['gridcell']

    data = api.request_data(schema_list, tables_list, columns_list, batch_size=1000)

    # Return plot_ids from index to a column.
    data.reset_index(inplace=True)
    data = data.drop_duplicates(subset='plot_id')
    
    return data

def get_copernicus_data():
    tables_list = ["copernicus_dem", "copernicus_slope", "copernicus_aspect"]
    columns_list = [None]*len(tables_list)
    schema_list = ['physical']*len(tables_list)
    
    data = api.request_data(schema_list, tables_list, columns_list, batch_size=1000)
    data = data.reset_index()
    data = data.drop_duplicates(subset='plot_id')
        
    return data

def get_soilgrids_data():
    data = api.request_data(data_groups=['soilgrids'])
    data = data.reset_index()
    data = data.drop_duplicates(subset='plot_id')
        
    return data

def get_climate_data():
    data = api.request_data(data_groups=['climate_data'])
    data = data.reset_index()
    data = data.drop_duplicates(subset='plot_id')
        
    return data

#metsakeskus_data = get_metsakeskus_data()
copernicus_data = get_copernicus_data()
#soilgrids_data = get_soilgrids_data()
#climate_data = get_climate_data()
112/10: metsakeskus_data[:2]
112/11: copernicus_data[:2]
112/12: soilgrids_data[:2]
112/13: climate_data[:2]
112/14:
### Get the metsäkeskus hila data

def get_metsakeskus_data():
    columns_list = [["volumepine","volumespruce","volumedeciduous","volume","soiltype","fertilityclass","maingroup",
                      "subgroup","laserheight","laserdensity"]]

    schema_list = ['metsakeskus_hila']
    tables_list = ['gridcell']

    data = api.request_data(schema_list, tables_list, columns_list, batch_size=1000)

    # Return plot_ids from index to a column.
    data.reset_index(inplace=True)
    data = data.drop_duplicates(subset='plot_id')
    
    return data

def get_copernicus_data():
    tables_list = ["copernicus_dem", "copernicus_slope", "copernicus_aspect"]
    columns_list = [None]*len(tables_list)
    schema_list = ['physical']*len(tables_list)
    
    data = api.request_data(schema_list, tables_list, columns_list, batch_size=1000)
    data = data.reset_index()
    data = data.drop_duplicates(subset='plot_id')
        
    return data

def get_soilgrids_data():
    data = api.request_data(data_groups=['soilgrids_all'])
    data = data.reset_index()
    data = data.drop_duplicates(subset='plot_id')
        
    return data

def get_climate_data():
    data = api.request_data(data_groups=['climate_data'])
    data = data.reset_index()
    data = data.drop_duplicates(subset='plot_id')
        
    return data

#metsakeskus_data = get_metsakeskus_data()
#copernicus_data = get_copernicus_data()
soilgrids_data = get_soilgrids_data()
#climate_data = get_climate_data()
112/15: metsakeskus_data[:2]
112/16: copernicus_data[:2]
112/17: soilgrids_data[:2]
112/18: climate_data[:2]
112/19: df[:2]
112/20: df.drop(["geom"], axis=1)
112/21:
unusable_features = ["geom", "soil_type", "plot_type", "cluster_id", "vol_pine", "vol_spruce", "vol_deciduous",
                    "vol_total", "measure_date", "measure_year"]
full_data = df.drop(unusable_features, axis=1)
112/22:
# Remove rows with NAs - gridcell data is missing from thousands of rows, aspect from about hundred.
full_data_nona = full_data.dropna()
115/1:
%load_ext autoreload
%autoreload 2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import json
import os
from tqdm import tqdm
import re
import sys
from sklearn.model_selection import train_test_split
import requests
pd.options.display.float_format = '{:,.2f}'.format
# Add path to where utils.py is so metrics can be imported
sys.path.insert(0, r'../regressors')
from data.data_loading import import_data, GeoAPI, split_from_ids
from data import data_loading
115/2:
input_path = r"../../../data/FIsampletiles"
image_dir = "azure_tiles_cleaned"

df = pd.read_csv(os.path.join(input_path, 'groundtruth.csv'))
df[['aspect']] = df[['aspect']].fillna(0)
df = df.dropna(subset=["soil_type"]) # only about 200 NAs here, just drop, not much lost
api = GeoAPI(default_locations=df[['easting', 'northing']].values.tolist(),
                default_srid=3067, default_plot_ids=df.plot_id.values.tolist())
115/3:
### Get the metsäkeskus hila data

def get_metsakeskus_data():
    columns_list = [["volumepine","volumespruce","volumedeciduous","volume","soiltype","fertilityclass","laserheight","laserdensity"]]

    schema_list = ['metsakeskus_hila']
    tables_list = ['gridcell']

    data = api.request_data(schema_list, tables_list, columns_list, batch_size=1000)

    # Return plot_ids from index to a column.
    data.reset_index(inplace=True)
    data = data.drop_duplicates(subset='plot_id')
    
    return data

def get_copernicus_data():
    tables_list = ["copernicus_dem", "copernicus_slope", "copernicus_aspect"]
    columns_list = [None]*len(tables_list)
    schema_list = ['physical']*len(tables_list)
    
    data = api.request_data(schema_list, tables_list, columns_list, batch_size=1000)
    data = data.reset_index()
    data = data.drop_duplicates(subset='plot_id')
        
    return data

def get_soilgrids_data():
    data = api.request_data(data_groups=['soilgrids'])
    data = data.reset_index()
    data = data.drop_duplicates(subset='plot_id')
        
    return data

def get_climate_data():
    data = api.request_data(data_groups=['climate_data'])
    data = data.reset_index()
    data = data.drop_duplicates(subset='plot_id')
        
    return data

metsakeskus_data = get_metsakeskus_data()
copernicus_data = get_copernicus_data()
soilgrids_data = get_soilgrids_data()
soilgrids_data = soilgrids_data.dropna() # only 47 rows missing, ok to drop nas
climate_data = get_climate_data()
115/4:
print("NAs in df data:\n", df.isna().sum())
print("NAs in metsakeskus data:\n", metsakeskus_data.isna().sum())
print("NAs in copernicus data:\n", copernicus_data.isna().sum())
print("NAs in soilgrids data:\n", soilgrids_data.isna().sum())
print("NAs in climate data:\n", climate_data.isna().sum())
115/5:
metsakeskus_columns = list(metsakeskus_data.columns)
copernicus_columns = list(copernicus_data.columns)
soilgrids_columns = list(soilgrids_data.columns)
climate_columns = list(climate_data.columns)
df_columns = ["easting", "northing", "elevation", "aspect" "slope", "soil_type", "tree_cover", "leaf_type"]

full_data = df.merge(metsakeskus_data, on='plot_id').\
merge(copernicus_data, on="plot_id").\
merge(soilgrids_data, on="plot_id").\
merge(climate_data, on="plot_id")

### drop na before or after how to make sure they're the same and such?
#unusable_features = ["geom", "soil_type", "plot_type", "cluster_id", "vol_pine", "vol_spruce", "vol_deciduous",
#                    "vol_total", "measure_date", "measure_year"]
#targets = ["vol_pine"]
#df = df.dropna()
#full_data_features = df.drop(unusable_features, axis=1)
#full_data_targets = df[targets]
115/6:
def test_different_models(data, feature_columns, target_columns = ["vol_pine"]):
    # Test different models with this data (mix different types, eg. soilgrids with metsakeskus and so on)
    
    from xgboost import XGBRegressor
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import mean_squared_error, make_scorer
    from sklearn.model_selection import RandomizedSearchCV
    
    features = data[feature_columns].drop("plot_id", axis=1)
    print(features.columns)
    targets = data[target_columns]
    # Default XGB
    default_xgb = XGBRegressor(objective='reg:linear', nthread=-1)
    
    # Search best parameters by CV
    param_distributions = {'max_depth': [3,8,15],
                           'learning_rate': [0.1,0.01,0.001],
                           'n_estimators': [100, 300, 500],
                           'min_child_weight': [1, 2, 5],
                           'colsample_bytree': [0.5, 0.8, 1],
                           'reg_alpha': [0, 0.1, 0.2],
                           'reg_lambda': [0.7, 1],
                           'subsample': [0.8, 0.9],
                           'gamma': [0, 0.07]
                       }

    search = RandomizedSearchCV(XGBRegressor(), scoring=make_scorer(mean_squared_error), param_distributions=param_distributions, 
                       n_jobs=-1, cv=5, verbose=True, n_iter=35)

    search.fit(features, targets)

    best_params = search.best_params_
    cv_xgb = XGBRegressor(**best_params)
    
    
    def scorer(estimator, x, y):
        preds = estimator.predict(x)
        error = np.sqrt(mean_squared_error(preds, y))
        return error
    
    scores = cross_val_score(default_xgb, features, targets, cv=5, scoring=scorer)
    print("RMSE mean of 5-fold CV with default XGB: ", np.mean(scores))
    scores = cross_val_score(cv_xgb, features, targets, cv=5, scoring=scorer)
    print("RMSE mean of 5-fold CV with CV optimized XGB: ", np.mean(scores))

# df contains some basic features such as easting and northing, which have repeatedly proven to  be good features
# so add them in all
copernicus = df_columns + copernicus_columns
print("CV 5-fold RMSE using just copernicus data: \n")
test_different_models(full_data, copernicus)

climate = df_columns + climate_columns
print("CV 5-fold RMSE using just climate data: \n")
test_different_models(full_data, climate)

copernicus_and_climate = df_columns + copernicus_columns + climate_columns
print("CV 5-fold RMSE using copernicus and climate data: \n")
test_different_models(full_data, copernicus_and_climate)

soilgrids = df_columns + soilgrids_columns
print("CV 5-fold RMSE using copernicus and climate data: \n")
test_different_models(full_data, soilgrids)

# So many NAs in metsakeskus, not worth?
#metsakeskus = 
#print("CV 5-fold RMSE using copernicus and climate data: \n")
#test_different_models(full_data, metsakeskus)

soilgrids_and_climate = df_columns + soilgrids_columns + climate_columns
print("CV 5-fold RMSE using soilgrids and climate data: \n")
test_different_models(full_data, soilgrids_and_climate)

soilgrids_and_copernicus = df_columns + soilgrids_columns + copernicus_columns
print("CV 5-fold RMSE using soilgrids and climate data: \n")
test_different_models(full_data, soilgrids_and_copernicus)

all_data = df_columns + soilgrids_columns + climate_columns + copernicus_columns
print("CV 5-fold RMSE using soilgrids and climate data: \n")
test_different_models(full_data, all_data)
115/7:
metsakeskus_columns = list(metsakeskus_data.columns)
copernicus_columns = list(copernicus_data.columns)
soilgrids_columns = list(soilgrids_data.columns)
climate_columns = list(climate_data.columns)
df_columns = ["easting", "northing", "elevation", "aspect", "slope", "soil_type", "tree_cover", "leaf_type"]

full_data = df.merge(metsakeskus_data, on='plot_id').\
merge(copernicus_data, on="plot_id").\
merge(soilgrids_data, on="plot_id").\
merge(climate_data, on="plot_id")

### drop na before or after how to make sure they're the same and such?
#unusable_features = ["geom", "soil_type", "plot_type", "cluster_id", "vol_pine", "vol_spruce", "vol_deciduous",
#                    "vol_total", "measure_date", "measure_year"]
#targets = ["vol_pine"]
#df = df.dropna()
#full_data_features = df.drop(unusable_features, axis=1)
#full_data_targets = df[targets]
115/8:
def test_different_models(data, feature_columns, target_columns = ["vol_pine"]):
    # Test different models with this data (mix different types, eg. soilgrids with metsakeskus and so on)
    
    from xgboost import XGBRegressor
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import mean_squared_error, make_scorer
    from sklearn.model_selection import RandomizedSearchCV
    
    features = data[feature_columns].drop("plot_id", axis=1)
    print(features.columns)
    targets = data[target_columns]
    # Default XGB
    default_xgb = XGBRegressor(objective='reg:linear', nthread=-1)
    
    # Search best parameters by CV
    param_distributions = {'max_depth': [3,8,15],
                           'learning_rate': [0.1,0.01,0.001],
                           'n_estimators': [100, 300, 500],
                           'min_child_weight': [1, 2, 5],
                           'colsample_bytree': [0.5, 0.8, 1],
                           'reg_alpha': [0, 0.1, 0.2],
                           'reg_lambda': [0.7, 1],
                           'subsample': [0.8, 0.9],
                           'gamma': [0, 0.07]
                       }

    search = RandomizedSearchCV(XGBRegressor(), scoring=make_scorer(mean_squared_error), param_distributions=param_distributions, 
                       n_jobs=-1, cv=5, verbose=True, n_iter=35)

    search.fit(features, targets)

    best_params = search.best_params_
    cv_xgb = XGBRegressor(**best_params)
    
    
    def scorer(estimator, x, y):
        preds = estimator.predict(x)
        error = np.sqrt(mean_squared_error(preds, y))
        return error
    
    scores = cross_val_score(default_xgb, features, targets, cv=5, scoring=scorer)
    print("RMSE mean of 5-fold CV with default XGB: ", np.mean(scores))
    scores = cross_val_score(cv_xgb, features, targets, cv=5, scoring=scorer)
    print("RMSE mean of 5-fold CV with CV optimized XGB: ", np.mean(scores))

# df contains some basic features such as easting and northing, which have repeatedly proven to  be good features
# so add them in all
copernicus = df_columns + copernicus_columns
print("CV 5-fold RMSE using just copernicus data: \n")
test_different_models(full_data, copernicus)

climate = df_columns + climate_columns
print("CV 5-fold RMSE using just climate data: \n")
test_different_models(full_data, climate)

copernicus_and_climate = df_columns + copernicus_columns + climate_columns
print("CV 5-fold RMSE using copernicus and climate data: \n")
test_different_models(full_data, copernicus_and_climate)

soilgrids = df_columns + soilgrids_columns
print("CV 5-fold RMSE using copernicus and climate data: \n")
test_different_models(full_data, soilgrids)

# So many NAs in metsakeskus, not worth?
#metsakeskus = 
#print("CV 5-fold RMSE using copernicus and climate data: \n")
#test_different_models(full_data, metsakeskus)

soilgrids_and_climate = df_columns + soilgrids_columns + climate_columns
print("CV 5-fold RMSE using soilgrids and climate data: \n")
test_different_models(full_data, soilgrids_and_climate)

soilgrids_and_copernicus = df_columns + soilgrids_columns + copernicus_columns
print("CV 5-fold RMSE using soilgrids and climate data: \n")
test_different_models(full_data, soilgrids_and_copernicus)

all_data = df_columns + soilgrids_columns + climate_columns + copernicus_columns
print("CV 5-fold RMSE using soilgrids and climate data: \n")
test_different_models(full_data, all_data)
116/1:
from scipy.stats import uniform

uniform.rvs(10)
116/2:
from scipy.stats import uniform

uniform.rvs(100)
116/3:
from scipy.stats import uniform

uniform.rvs(size=10)
116/4:
from scipy.stats import uniform

uniform.rvs(scale=100)
116/5:
from scipy.stats import uniform

uniform.rvs(scale=100)
116/6:
from scipy.stats import uniform

uniform.rvs(scale=100)
116/7:
from scipy.stats import uniform

uniform.rvs(scale=100)
116/8:
from scipy.stats import uniform

uniform.rvs(scale=100)
116/9:
from scipy.stats import uniform

uniform.rvs(scale=100)
116/10:
from scipy.stats import uniform

uniform.rvs(scale=100)
116/11:
from scipy.stats import uniform

uniform.rvs(scale=100)
116/12:
from scipy.stats import uniform

uniform.rvs(scale=100)
116/13:
from scipy.stats import uniform

uniform.rvs(scale=100)
116/14:
from scipy.stats import uniform

uniform.rvs(scale=100)
116/15:
from scipy.stats import uniform

uniform.rvs(scale=100)
116/16:
from scipy.stats import uniform

uniform.rvs(scale=100)
116/17:
from scipy.stats import uniform

uniform.rvs(scale=100)
116/18:
from scipy.stats import uniform

uniform.rvs(scale=[100, 400])
116/19:
from scipy.stats import uniform

uniform.rvs(scale=[100, 400])
116/20:
from scipy.stats import uniform

uniform.rvs(scale=[100, 400])
116/21:
from scipy.stats import uniform

uniform.rvs(loc=100, scale=400)
116/22:
from scipy.stats import uniform

uniform.rvs(loc=100, scale=400)
116/23:
from scipy.stats import uniform

uniform.rvs(loc=100, scale=400)
116/24:
from scipy.stats import uniform

uniform.rvs(loc=100, scale=400)
116/25:
from scipy.stats import uniform

uniform.rvs(loc=100, scale=400)
116/26:
from scipy.stats import uniform

uniform.rvs(loc=100, scale=400)
116/27:
from scipy.stats import uniform

uniform.rvs(loc=100, scale=400)
116/28:
from scipy.stats import uniform

uniform(loc=100, scale=400)
116/29:
from scipy.stats import uniform

uni = uniform(loc=100, scale=400)

uni.rvs()
116/30:
from scipy.stats import uniform

uni = uniform(loc=100, scale=400)

uni.rvs()
116/31:
from scipy.stats import uniform

uni = uniform(loc=100, scale=400)

uni.rvs()
116/32:
from scipy.stats import uniform

uni = uniform(loc=100, scale=400)

uni.rvs()
116/33:
from scipy.stats import uniform

uni = uniform(loc=100, scale=400)

uni.rvs()
116/34:
from scipy.stats import uniform

uni = uniform(loc=100, scale=400)

uni.rvs()
116/35:
from scipy.stats import uniform

uni = uniform(loc=100, scale=400)

uni.rvs()
116/36:
from scipy.stats import uniform

uni = uniform(loc=100, scale=400)

uni.rvs()
116/37:
from scipy.stats import uniform

uni = uniform(loc=100, scale=400)

uni.rvs()
116/38:
from scipy.stats import uniform

uni = uniform(loc=100, scale=400)

uni.rvs()
118/1:
%load_ext autoreload
%autoreload 2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import json
import os
from tqdm import tqdm
import re
import sys
from sklearn.model_selection import train_test_split
import requests
pd.options.display.float_format = '{:,.2f}'.format
# Add path to where utils.py is so metrics can be imported
sys.path.insert(0, r'../regressors')
from data.data_loading import import_data, GeoAPI, split_from_ids
from data import data_loading
118/2:
input_path = r"../../../data/FIsampletiles"
image_dir = "azure_tiles_cleaned"

df = pd.read_csv(os.path.join(input_path, 'groundtruth.csv'))
df[['aspect']] = df[['aspect']].fillna(0)
df = df.dropna(subset=["soil_type"]) # only about 200 NAs here, just drop, not much lost
api = GeoAPI(default_locations=df[['easting', 'northing']].values.tolist(),
                default_srid=3067, default_plot_ids=df.plot_id.values.tolist())
118/3:
### Get the metsäkeskus hila data

def get_metsakeskus_data():
    columns_list = [["volumepine","volumespruce","volumedeciduous","volume","soiltype","fertilityclass","laserheight","laserdensity"]]

    schema_list = ['metsakeskus_hila']
    tables_list = ['gridcell']

    data = api.request_data(schema_list, tables_list, columns_list, batch_size=1000)

    # Return plot_ids from index to a column.
    data.reset_index(inplace=True)
    data = data.drop_duplicates(subset='plot_id')
    
    return data

def get_copernicus_data():
    tables_list = ["copernicus_dem", "copernicus_slope", "copernicus_aspect"]
    columns_list = [None]*len(tables_list)
    schema_list = ['physical']*len(tables_list)
    
    data = api.request_data(schema_list, tables_list, columns_list, batch_size=1000)
    data = data.reset_index()
    data = data.drop_duplicates(subset='plot_id')
        
    return data

def get_soilgrids_data():
    data = api.request_data(data_groups=['soilgrids'])
    data = data.reset_index()
    data = data.drop_duplicates(subset='plot_id')
        
    return data

def get_climate_data():
    data = api.request_data(data_groups=['climate_data'])
    data = data.reset_index()
    data = data.drop_duplicates(subset='plot_id')
        
    return data

metsakeskus_data = get_metsakeskus_data()
copernicus_data = get_copernicus_data()
soilgrids_data = get_soilgrids_data()
soilgrids_data = soilgrids_data.dropna() # only 47 rows missing, ok to drop nas
climate_data = get_climate_data()
118/4:
print("NAs in df data:\n", df.isna().sum())
print("NAs in metsakeskus data:\n", metsakeskus_data.isna().sum())
print("NAs in copernicus data:\n", copernicus_data.isna().sum())
print("NAs in soilgrids data:\n", soilgrids_data.isna().sum())
print("NAs in climate data:\n", climate_data.isna().sum())
118/5:
metsakeskus_columns = list(metsakeskus_data.columns)
copernicus_columns = list(copernicus_data.columns)
soilgrids_columns = list(soilgrids_data.columns)
climate_columns = list(climate_data.columns)
df_columns = ["easting", "northing", "elevation", "aspect", "slope", "soil_type", "tree_cover", "leaf_type"]

columns_dict = {
    'base': df_columns,
    'metsakeskus': metsakeskus_columns,
    'copernicus': copernicus_columns,
    'soilgrids': soilgrids_columns,
    'climate': climate_columns
}

full_data = df.merge(metsakeskus_data, on='plot_id').\
merge(copernicus_data, on="plot_id").\
merge(soilgrids_data, on="plot_id").\
merge(climate_data, on="plot_id")

### drop na before or after how to make sure they're the same and such?
#unusable_features = ["geom", "soil_type", "plot_type", "cluster_id", "vol_pine", "vol_spruce", "vol_deciduous",
#                    "vol_total", "measure_date", "measure_year"]
#targets = ["vol_pine"]
#df = df.dropna()
#full_data_features = df.drop(unusable_features, axis=1)
#full_data_targets = df[targets]
118/6:
def test_different_models(data, feature_columns, target_columns = ["vol_pine"]):
    # Test different models with this data (mix different types, eg. soilgrids with metsakeskus and so on)
    
    from xgboost import XGBRegressor
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import mean_squared_error, make_scorer
    from sklearn.model_selection import RandomizedSearchCV
    from scipy.stats import uniform
    
    features = data[feature_columns].drop("plot_id", axis=1)
    print(features.columns)
    targets = data[target_columns]
    # Default XGB
    default_xgb = XGBRegressor(objective='reg:linear', nthread=-1)
    
    # Search best parameters by CV. Note: if all are lists, sampling is done without replacement? which is bad?
    param_distributions = {'max_depth': uniform(loc=3, scale=15),
                           'learning_rate': uniform(loc=0.001, scale=0.1),
                           'n_estimators': uniform(loc=100, scale=600),
                           'min_child_weight': [1, 2, 5],
                           'colsample_bytree': uniform(loc=0.5, scale=1),
                           'reg_alpha': [0, 0.1, 0.2],
                           'reg_lambda': [0.7, 1],
                           'subsample': [0.8, 0.9],
                           'gamma': [0, 0.07]
                       }

    search = RandomizedSearchCV(XGBRegressor(), scoring=make_scorer(mean_squared_error), param_distributions=param_distributions, 
                       n_jobs=-1, cv=5, verbose=True, n_iter=35)

    search.fit(features, targets)

    best_params = search.best_params_
    cv_xgb = XGBRegressor(**best_params)
    
    
    def scorer(estimator, x, y):
        preds = estimator.predict(x)
        error = np.sqrt(mean_squared_error(preds, y))
        return error
    
    scores = cross_val_score(default_xgb, features, targets, cv=5, scoring=scorer)
    print("RMSE mean of 5-fold CV with default XGB: ", np.mean(scores))
    scores = cross_val_score(cv_xgb, features, targets, cv=5, scoring=scorer)
    print("RMSE mean of 5-fold CV with CV optimized XGB: ", np.mean(scores))

# df contains some basic features such as easting and northing, which have repeatedly proven to  be good features
# so add them in all
copernicus = df_columns + copernicus_columns
print("CV 5-fold RMSE using just copernicus data: \n")
test_different_models(full_data, copernicus)

climate = df_columns + climate_columns
print("CV 5-fold RMSE using just climate data: \n")
test_different_models(full_data, climate)

copernicus_and_climate = df_columns + copernicus_columns + climate_columns
print("CV 5-fold RMSE using copernicus and climate data: \n")
test_different_models(full_data, copernicus_and_climate)

soilgrids = df_columns + soilgrids_columns
print("CV 5-fold RMSE using copernicus and climate data: \n")
test_different_models(full_data, soilgrids)

# So many NAs in metsakeskus, not worth?
#metsakeskus = 
#print("CV 5-fold RMSE using copernicus and climate data: \n")
#test_different_models(full_data, metsakeskus)

soilgrids_and_climate = df_columns + soilgrids_columns + climate_columns
print("CV 5-fold RMSE using soilgrids and climate data: \n")
test_different_models(full_data, soilgrids_and_climate)

soilgrids_and_copernicus = df_columns + soilgrids_columns + copernicus_columns
print("CV 5-fold RMSE using soilgrids and copernicus data: \n")
test_different_models(full_data, soilgrids_and_copernicus)

all_data = df_columns + soilgrids_columns + climate_columns + copernicus_columns
print("CV 5-fold RMSE using all data: \n")
test_different_models(full_data, all_data)
118/7:
def test_different_models(data, feature_columns, target_columns = ["vol_pine"]):
    # Test different models with this data (mix different types, eg. soilgrids with metsakeskus and so on)
    
    from xgboost import XGBRegressor
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import mean_squared_error, make_scorer
    from sklearn.model_selection import RandomizedSearchCV
    from scipy.stats import uniform
    
    features = data[feature_columns].drop("plot_id", axis=1)
    print(features.columns)
    targets = data[target_columns]
    # Default XGB
    default_xgb = XGBRegressor(objective='reg:linear', nthread=-1)
    
    # Search best parameters by CV. Note: if all are lists, sampling is done without replacement? which is bad?
    param_distributions = {'max_depth': uniform(loc=3, scale=15),
                           'learning_rate': uniform(loc=0.001, scale=0.1),
                           'n_estimators': uniform(loc=100, scale=600),
                           'min_child_weight': [1, 2, 5],
                           'colsample_bytree': uniform(loc=0.5, scale=1),
                           'reg_alpha': [0, 0.1, 0.2],
                           'reg_lambda': [0.7, 1],
                           'subsample': [0.8, 0.9],
                           'gamma': [0, 0.07]
                       }

    search = RandomizedSearchCV(XGBRegressor(), scoring=make_scorer(mean_squared_error), param_distributions=param_distributions, 
                       n_jobs=-1, cv=5, verbose=True, n_iter=35)

    search.fit(features, targets)

    best_params = search.best_params_
    cv_xgb = XGBRegressor(**best_params)
    
    
    def scorer(estimator, x, y):
        preds = estimator.predict(x)
        error = np.sqrt(mean_squared_error(preds, y))
        return error
    
    scores = cross_val_score(default_xgb, features, targets, cv=5, scoring=scorer)
    print("RMSE mean of 5-fold CV with default XGB: ", np.mean(scores))
    scores = cross_val_score(cv_xgb, features, targets, cv=5, scoring=scorer)
    print("RMSE mean of 5-fold CV with CV optimized XGB: ", np.mean(scores))

# df contains some basic features such as easting and northing, which have repeatedly proven to  be good features
# so add them in all
copernicus = df_columns + copernicus_columns
print("CV 5-fold RMSE using just copernicus data: \n")
test_different_models(full_data, copernicus)

climate = df_columns + climate_columns
print("CV 5-fold RMSE using just climate data: \n")
test_different_models(full_data, climate)

copernicus_and_climate = df_columns + copernicus_columns + climate_columns
print("CV 5-fold RMSE using copernicus and climate data: \n")
test_different_models(full_data, copernicus_and_climate)

soilgrids = df_columns + soilgrids_columns
print("CV 5-fold RMSE using copernicus and climate data: \n")
test_different_models(full_data, soilgrids)

# So many NAs in metsakeskus, not worth?
#metsakeskus = 
#print("CV 5-fold RMSE using copernicus and climate data: \n")
#test_different_models(full_data, metsakeskus)

soilgrids_and_climate = df_columns + soilgrids_columns + climate_columns
print("CV 5-fold RMSE using soilgrids and climate data: \n")
test_different_models(full_data, soilgrids_and_climate)

soilgrids_and_copernicus = df_columns + soilgrids_columns + copernicus_columns
print("CV 5-fold RMSE using soilgrids and copernicus data: \n")
test_different_models(full_data, soilgrids_and_copernicus)

all_data = df_columns + soilgrids_columns + climate_columns + copernicus_columns
print("CV 5-fold RMSE using all data: \n")
test_different_models(full_data, all_data)
118/8:
def test_different_models(data, feature_columns, target_columns = ["vol_pine"]):
    # Test different models with this data (mix different types, eg. soilgrids with metsakeskus and so on)
    
    from xgboost import XGBRegressor
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import mean_squared_error, make_scorer
    from sklearn.model_selection import RandomizedSearchCV
    from scipy.stats import uniform
    
    features = data[feature_columns].drop("plot_id", axis=1)
    print(features.columns)
    targets = data[target_columns]
    # Default XGB
    default_xgb = XGBRegressor(objective='reg:linear', nthread=-1)
    
    # Search best parameters by CV. Note: if all are lists, sampling is done without replacement? which is bad?
    param_distributions = {'max_depth': uniform(loc=3, scale=15),
                           #'learning_rate': uniform(loc=0.001, scale=0.1),
                           'n_estimators': uniform(loc=100, scale=600),
                           'min_child_weight': [1, 2, 5],
                           #'colsample_bytree': uniform(loc=0.5, scale=1),
                           'reg_alpha': [0, 0.1, 0.2],
                           'reg_lambda': [0.7, 1],
                           'subsample': [0.8, 0.9],
                           'gamma': [0, 0.07]
                       }

    search = RandomizedSearchCV(XGBRegressor(), scoring=make_scorer(mean_squared_error), param_distributions=param_distributions, 
                       n_jobs=-1, cv=5, verbose=True, n_iter=35)

    search.fit(features, targets)

    best_params = search.best_params_
    cv_xgb = XGBRegressor(**best_params)
    
    
    def scorer(estimator, x, y):
        preds = estimator.predict(x)
        error = np.sqrt(mean_squared_error(preds, y))
        return error
    
    scores = cross_val_score(default_xgb, features, targets, cv=5, scoring=scorer)
    print("RMSE mean of 5-fold CV with default XGB: ", np.mean(scores))
    scores = cross_val_score(cv_xgb, features, targets, cv=5, scoring=scorer)
    print("RMSE mean of 5-fold CV with CV optimized XGB: ", np.mean(scores))

# df contains some basic features such as easting and northing, which have repeatedly proven to  be good features
# so add them in all
copernicus = df_columns + copernicus_columns
print("CV 5-fold RMSE using just copernicus data: \n")
test_different_models(full_data, copernicus)

climate = df_columns + climate_columns
print("CV 5-fold RMSE using just climate data: \n")
test_different_models(full_data, climate)

copernicus_and_climate = df_columns + copernicus_columns + climate_columns
print("CV 5-fold RMSE using copernicus and climate data: \n")
test_different_models(full_data, copernicus_and_climate)

soilgrids = df_columns + soilgrids_columns
print("CV 5-fold RMSE using copernicus and climate data: \n")
test_different_models(full_data, soilgrids)

# So many NAs in metsakeskus, not worth?
#metsakeskus = 
#print("CV 5-fold RMSE using copernicus and climate data: \n")
#test_different_models(full_data, metsakeskus)

soilgrids_and_climate = df_columns + soilgrids_columns + climate_columns
print("CV 5-fold RMSE using soilgrids and climate data: \n")
test_different_models(full_data, soilgrids_and_climate)

soilgrids_and_copernicus = df_columns + soilgrids_columns + copernicus_columns
print("CV 5-fold RMSE using soilgrids and copernicus data: \n")
test_different_models(full_data, soilgrids_and_copernicus)

all_data = df_columns + soilgrids_columns + climate_columns + copernicus_columns
print("CV 5-fold RMSE using all data: \n")
test_different_models(full_data, all_data)
118/9:
from scipy.stats import uniform

uniform(loc=0.001, scale=0.1).rvs()
118/10:
from scipy.stats import uniform

uniform(loc=0.001, scale=0.1).rvs()
118/11:
from scipy.stats import uniform

uniform(loc=0.001, scale=0.1).rvs()
118/12:
from scipy.stats import uniform

uniform(loc=0.001, scale=0.1).rvs()
118/13:
def test_different_models(data, feature_columns, target_columns = ["vol_pine"]):
    # Test different models with this data (mix different types, eg. soilgrids with metsakeskus and so on)
    
    from xgboost import XGBRegressor
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import mean_squared_error, make_scorer
    from sklearn.model_selection import RandomizedSearchCV
    from scipy.stats import uniform
    
    features = data[feature_columns].drop("plot_id", axis=1)
    print(features.columns)
    targets = data[target_columns]
    # Default XGB
    default_xgb = XGBRegressor(objective='reg:linear', nthread=-1)
    
    # Search best parameters by CV. Note: if all are lists, sampling is done without replacement? which is bad?
    param_distributions = {'max_depth': uniform(loc=3, scale=15),
                           'learning_rate': uniform(loc=0.001, scale=0.1),
                           'n_estimators': uniform(loc=100, scale=600),
                           'min_child_weight': [1, 2, 5],
                           'colsample_bytree': uniform(loc=0.5, scale=1),
                           'reg_alpha': [0, 0.1, 0.2],
                           'reg_lambda': [0.7, 1],
                           'subsample': [0.8, 0.9],
                           'gamma': [0, 0.07]
                       }

    search = RandomizedSearchCV(XGBRegressor(), scoring=make_scorer(mean_squared_error), param_distributions=param_distributions, 
                       n_jobs=-1, cv=5, verbose=True, n_iter=35)

    search.fit(features, targets)

    best_params = search.best_params_
    cv_xgb = XGBRegressor(**best_params)
    
    
    def scorer(estimator, x, y):
        preds = estimator.predict(x)
        error = np.sqrt(mean_squared_error(preds, y))
        return error
    
    scores = cross_val_score(default_xgb, features, targets, cv=5, scoring=scorer)
    print("RMSE mean of 5-fold CV with default XGB: ", np.mean(scores))
    scores = cross_val_score(cv_xgb, features, targets, cv=5, scoring=scorer)
    print("RMSE mean of 5-fold CV with CV optimized XGB: ", np.mean(scores))

# df contains some basic features such as easting and northing, which have repeatedly proven to  be good features
# so add them in all
copernicus = df_columns + copernicus_columns
print("CV 5-fold RMSE using just copernicus data: \n")
test_different_models(full_data, copernicus)

climate = df_columns + climate_columns
print("CV 5-fold RMSE using just climate data: \n")
test_different_models(full_data, climate)

copernicus_and_climate = df_columns + copernicus_columns + climate_columns
print("CV 5-fold RMSE using copernicus and climate data: \n")
test_different_models(full_data, copernicus_and_climate)

soilgrids = df_columns + soilgrids_columns
print("CV 5-fold RMSE using copernicus and climate data: \n")
test_different_models(full_data, soilgrids)

# So many NAs in metsakeskus, not worth?
#metsakeskus = 
#print("CV 5-fold RMSE using copernicus and climate data: \n")
#test_different_models(full_data, metsakeskus)

soilgrids_and_climate = df_columns + soilgrids_columns + climate_columns
print("CV 5-fold RMSE using soilgrids and climate data: \n")
test_different_models(full_data, soilgrids_and_climate)

soilgrids_and_copernicus = df_columns + soilgrids_columns + copernicus_columns
print("CV 5-fold RMSE using soilgrids and copernicus data: \n")
test_different_models(full_data, soilgrids_and_copernicus)

all_data = df_columns + soilgrids_columns + climate_columns + copernicus_columns
print("CV 5-fold RMSE using all data: \n")
test_different_models(full_data, all_data)
118/14:
def test_different_models(data, feature_columns, target_columns = ["vol_pine"]):
    # Test different models with this data (mix different types, eg. soilgrids with metsakeskus and so on)
    
    from xgboost import XGBRegressor
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import mean_squared_error, make_scorer
    from sklearn.model_selection import RandomizedSearchCV
    from scipy.stats import uniform
    
    features = data[feature_columns].drop("plot_id", axis=1)
    print(features.columns)
    targets = data[target_columns]
    # Default XGB
    default_xgb = XGBRegressor(objective='reg:linear', nthread=-1)
    
    # Search best parameters by CV. Note: if all are lists, sampling is done without replacement? which is bad?
    param_distributions = {#'max_depth': uniform(loc=3, scale=15),
                           #'learning_rate': uniform(loc=0.001, scale=0.1),
                           #'n_estimators': uniform(loc=100, scale=600),
                           'min_child_weight': [1, 2, 5],
                           #'colsample_bytree': uniform(loc=0.5, scale=1),
                           'reg_alpha': [0, 0.1, 0.2],
                           'reg_lambda': [0.7, 1],
                           'subsample': [0.8, 0.9],
                           'gamma': [0, 0.07]
                       }

    search = RandomizedSearchCV(XGBRegressor(), scoring=make_scorer(mean_squared_error), param_distributions=param_distributions, 
                       n_jobs=-1, cv=5, verbose=True, n_iter=35)

    search.fit(features, targets)

    best_params = search.best_params_
    cv_xgb = XGBRegressor(**best_params)
    
    
    def scorer(estimator, x, y):
        preds = estimator.predict(x)
        error = np.sqrt(mean_squared_error(preds, y))
        return error
    
    scores = cross_val_score(default_xgb, features, targets, cv=5, scoring=scorer)
    print("RMSE mean of 5-fold CV with default XGB: ", np.mean(scores))
    scores = cross_val_score(cv_xgb, features, targets, cv=5, scoring=scorer)
    print("RMSE mean of 5-fold CV with CV optimized XGB: ", np.mean(scores))

# df contains some basic features such as easting and northing, which have repeatedly proven to  be good features
# so add them in all
copernicus = df_columns + copernicus_columns
print("CV 5-fold RMSE using just copernicus data: \n")
test_different_models(full_data, copernicus)

climate = df_columns + climate_columns
print("CV 5-fold RMSE using just climate data: \n")
test_different_models(full_data, climate)

copernicus_and_climate = df_columns + copernicus_columns + climate_columns
print("CV 5-fold RMSE using copernicus and climate data: \n")
test_different_models(full_data, copernicus_and_climate)

soilgrids = df_columns + soilgrids_columns
print("CV 5-fold RMSE using copernicus and climate data: \n")
test_different_models(full_data, soilgrids)

# So many NAs in metsakeskus, not worth?
#metsakeskus = 
#print("CV 5-fold RMSE using copernicus and climate data: \n")
#test_different_models(full_data, metsakeskus)

soilgrids_and_climate = df_columns + soilgrids_columns + climate_columns
print("CV 5-fold RMSE using soilgrids and climate data: \n")
test_different_models(full_data, soilgrids_and_climate)

soilgrids_and_copernicus = df_columns + soilgrids_columns + copernicus_columns
print("CV 5-fold RMSE using soilgrids and copernicus data: \n")
test_different_models(full_data, soilgrids_and_copernicus)

all_data = df_columns + soilgrids_columns + climate_columns + copernicus_columns
print("CV 5-fold RMSE using all data: \n")
test_different_models(full_data, all_data)
118/15:
def test_different_models(data, feature_columns, target_columns = ["vol_pine"]):
    # Test different models with this data (mix different types, eg. soilgrids with metsakeskus and so on)
    
    from xgboost import XGBRegressor
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import mean_squared_error, make_scorer
    from sklearn.model_selection import RandomizedSearchCV
    from scipy.stats import uniform
    
    features = data[feature_columns].drop("plot_id", axis=1)
    print(features.columns)
    targets = data[target_columns]
    # Default XGB
    default_xgb = XGBRegressor(objective='reg:linear', nthread=-1)
    
    # Search best parameters by CV. Note: if all are lists, sampling is done without replacement? which is bad?
    param_distributions = {'max_depth': uniform(loc=3, scale=15),
                           #'learning_rate': uniform(loc=0.001, scale=0.1),
                           #'n_estimators': uniform(loc=100, scale=600),
                           'min_child_weight': [1, 2, 5],
                           #'colsample_bytree': uniform(loc=0.5, scale=1),
                           'reg_alpha': [0, 0.1, 0.2],
                           'reg_lambda': [0.7, 1],
                           'subsample': [0.8, 0.9],
                           'gamma': [0, 0.07]
                       }

    search = RandomizedSearchCV(XGBRegressor(), scoring=make_scorer(mean_squared_error), param_distributions=param_distributions, 
                       n_jobs=-1, cv=5, verbose=True, n_iter=35)

    search.fit(features, targets)

    best_params = search.best_params_
    cv_xgb = XGBRegressor(**best_params)
    
    
    def scorer(estimator, x, y):
        preds = estimator.predict(x)
        error = np.sqrt(mean_squared_error(preds, y))
        return error
    
    scores = cross_val_score(default_xgb, features, targets, cv=5, scoring=scorer)
    print("RMSE mean of 5-fold CV with default XGB: ", np.mean(scores))
    scores = cross_val_score(cv_xgb, features, targets, cv=5, scoring=scorer)
    print("RMSE mean of 5-fold CV with CV optimized XGB: ", np.mean(scores))

# df contains some basic features such as easting and northing, which have repeatedly proven to  be good features
# so add them in all
copernicus = df_columns + copernicus_columns
print("CV 5-fold RMSE using just copernicus data: \n")
test_different_models(full_data, copernicus)

climate = df_columns + climate_columns
print("CV 5-fold RMSE using just climate data: \n")
test_different_models(full_data, climate)

copernicus_and_climate = df_columns + copernicus_columns + climate_columns
print("CV 5-fold RMSE using copernicus and climate data: \n")
test_different_models(full_data, copernicus_and_climate)

soilgrids = df_columns + soilgrids_columns
print("CV 5-fold RMSE using copernicus and climate data: \n")
test_different_models(full_data, soilgrids)

# So many NAs in metsakeskus, not worth?
#metsakeskus = 
#print("CV 5-fold RMSE using copernicus and climate data: \n")
#test_different_models(full_data, metsakeskus)

soilgrids_and_climate = df_columns + soilgrids_columns + climate_columns
print("CV 5-fold RMSE using soilgrids and climate data: \n")
test_different_models(full_data, soilgrids_and_climate)

soilgrids_and_copernicus = df_columns + soilgrids_columns + copernicus_columns
print("CV 5-fold RMSE using soilgrids and copernicus data: \n")
test_different_models(full_data, soilgrids_and_copernicus)

all_data = df_columns + soilgrids_columns + climate_columns + copernicus_columns
print("CV 5-fold RMSE using all data: \n")
test_different_models(full_data, all_data)
118/16:
def test_different_models(data, feature_columns, target_columns = ["vol_pine"]):
    # Test different models with this data (mix different types, eg. soilgrids with metsakeskus and so on)
    
    from xgboost import XGBRegressor
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import mean_squared_error, make_scorer
    from sklearn.model_selection import RandomizedSearchCV
    from scipy.stats import uniform
    
    features = data[feature_columns].drop("plot_id", axis=1)
    print(features.columns)
    targets = data[target_columns]
    # Default XGB
    default_xgb = XGBRegressor(objective='reg:linear', nthread=-1)
    
    # Search best parameters by CV. Note: if all are lists, sampling is done without replacement? which is bad?
    param_distributions = {'max_depth': int(uniform(loc=3, scale=15)),
                           #'learning_rate': uniform(loc=0.001, scale=0.1),
                           #'n_estimators': uniform(loc=100, scale=600),
                           'min_child_weight': [1, 2, 5],
                           #'colsample_bytree': uniform(loc=0.5, scale=1),
                           'reg_alpha': [0, 0.1, 0.2],
                           'reg_lambda': [0.7, 1],
                           'subsample': [0.8, 0.9],
                           'gamma': [0, 0.07]
                       }

    search = RandomizedSearchCV(XGBRegressor(), scoring=make_scorer(mean_squared_error), param_distributions=param_distributions, 
                       n_jobs=-1, cv=5, verbose=True, n_iter=35)

    search.fit(features, targets)

    best_params = search.best_params_
    cv_xgb = XGBRegressor(**best_params)
    
    
    def scorer(estimator, x, y):
        preds = estimator.predict(x)
        error = np.sqrt(mean_squared_error(preds, y))
        return error
    
    scores = cross_val_score(default_xgb, features, targets, cv=5, scoring=scorer)
    print("RMSE mean of 5-fold CV with default XGB: ", np.mean(scores))
    scores = cross_val_score(cv_xgb, features, targets, cv=5, scoring=scorer)
    print("RMSE mean of 5-fold CV with CV optimized XGB: ", np.mean(scores))

# df contains some basic features such as easting and northing, which have repeatedly proven to  be good features
# so add them in all
copernicus = df_columns + copernicus_columns
print("CV 5-fold RMSE using just copernicus data: \n")
test_different_models(full_data, copernicus)

climate = df_columns + climate_columns
print("CV 5-fold RMSE using just climate data: \n")
test_different_models(full_data, climate)

copernicus_and_climate = df_columns + copernicus_columns + climate_columns
print("CV 5-fold RMSE using copernicus and climate data: \n")
test_different_models(full_data, copernicus_and_climate)

soilgrids = df_columns + soilgrids_columns
print("CV 5-fold RMSE using copernicus and climate data: \n")
test_different_models(full_data, soilgrids)

# So many NAs in metsakeskus, not worth?
#metsakeskus = 
#print("CV 5-fold RMSE using copernicus and climate data: \n")
#test_different_models(full_data, metsakeskus)

soilgrids_and_climate = df_columns + soilgrids_columns + climate_columns
print("CV 5-fold RMSE using soilgrids and climate data: \n")
test_different_models(full_data, soilgrids_and_climate)

soilgrids_and_copernicus = df_columns + soilgrids_columns + copernicus_columns
print("CV 5-fold RMSE using soilgrids and copernicus data: \n")
test_different_models(full_data, soilgrids_and_copernicus)

all_data = df_columns + soilgrids_columns + climate_columns + copernicus_columns
print("CV 5-fold RMSE using all data: \n")
test_different_models(full_data, all_data)
118/17:
def test_different_models(data, feature_columns, target_columns = ["vol_pine"]):
    # Test different models with this data (mix different types, eg. soilgrids with metsakeskus and so on)
    
    from xgboost import XGBRegressor
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import mean_squared_error, make_scorer
    from sklearn.model_selection import RandomizedSearchCV
    from scipy.stats import uniform
    
    features = data[feature_columns].drop("plot_id", axis=1)
    print(features.columns)
    targets = data[target_columns]
    # Default XGB
    default_xgb = XGBRegressor(objective='reg:linear', nthread=-1)
    
    # Search best parameters by CV. Note: if all are lists, sampling is done without replacement? which is bad?
    param_distributions = {'max_depth': np.float64(3),
                           #'learning_rate': uniform(loc=0.001, scale=0.1),
                           #'n_estimators': uniform(loc=100, scale=600),
                           'min_child_weight': [1, 2, 5],
                           #'colsample_bytree': uniform(loc=0.5, scale=1),
                           'reg_alpha': [0, 0.1, 0.2],
                           'reg_lambda': [0.7, 1],
                           'subsample': [0.8, 0.9],
                           'gamma': [0, 0.07]
                       }

    search = RandomizedSearchCV(XGBRegressor(), scoring=make_scorer(mean_squared_error), param_distributions=param_distributions, 
                       n_jobs=-1, cv=5, verbose=True, n_iter=35)

    search.fit(features, targets)

    best_params = search.best_params_
    cv_xgb = XGBRegressor(**best_params)
    
    
    def scorer(estimator, x, y):
        preds = estimator.predict(x)
        error = np.sqrt(mean_squared_error(preds, y))
        return error
    
    scores = cross_val_score(default_xgb, features, targets, cv=5, scoring=scorer)
    print("RMSE mean of 5-fold CV with default XGB: ", np.mean(scores))
    scores = cross_val_score(cv_xgb, features, targets, cv=5, scoring=scorer)
    print("RMSE mean of 5-fold CV with CV optimized XGB: ", np.mean(scores))

# df contains some basic features such as easting and northing, which have repeatedly proven to  be good features
# so add them in all
copernicus = df_columns + copernicus_columns
print("CV 5-fold RMSE using just copernicus data: \n")
test_different_models(full_data, copernicus)

climate = df_columns + climate_columns
print("CV 5-fold RMSE using just climate data: \n")
test_different_models(full_data, climate)

copernicus_and_climate = df_columns + copernicus_columns + climate_columns
print("CV 5-fold RMSE using copernicus and climate data: \n")
test_different_models(full_data, copernicus_and_climate)

soilgrids = df_columns + soilgrids_columns
print("CV 5-fold RMSE using copernicus and climate data: \n")
test_different_models(full_data, soilgrids)

# So many NAs in metsakeskus, not worth?
#metsakeskus = 
#print("CV 5-fold RMSE using copernicus and climate data: \n")
#test_different_models(full_data, metsakeskus)

soilgrids_and_climate = df_columns + soilgrids_columns + climate_columns
print("CV 5-fold RMSE using soilgrids and climate data: \n")
test_different_models(full_data, soilgrids_and_climate)

soilgrids_and_copernicus = df_columns + soilgrids_columns + copernicus_columns
print("CV 5-fold RMSE using soilgrids and copernicus data: \n")
test_different_models(full_data, soilgrids_and_copernicus)

all_data = df_columns + soilgrids_columns + climate_columns + copernicus_columns
print("CV 5-fold RMSE using all data: \n")
test_different_models(full_data, all_data)
118/18:
def test_different_models(data, feature_columns, target_columns = ["vol_pine"]):
    # Test different models with this data (mix different types, eg. soilgrids with metsakeskus and so on)
    
    from xgboost import XGBRegressor
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import mean_squared_error, make_scorer
    from sklearn.model_selection import RandomizedSearchCV
    from scipy.stats import uniform
    
    features = data[feature_columns].drop("plot_id", axis=1)
    print(features.columns)
    targets = data[target_columns]
    # Default XGB
    default_xgb = XGBRegressor(objective='reg:linear', nthread=-1)
    
    # Search best parameters by CV. Note: if all are lists, sampling is done without replacement? which is bad?
    param_distributions = {'max_depth': uniform(loc=3, scale=15),
                           #'learning_rate': uniform(loc=0.001, scale=0.1),
                           #'n_estimators': uniform(loc=100, scale=600),
                           'min_child_weight': [1, 2, 5],
                           #'colsample_bytree': uniform(loc=0.5, scale=1),
                           'reg_alpha': [0, 0.1, 0.2],
                           'reg_lambda': [0.7, 1],
                           'subsample': [0.8, 0.9],
                           'gamma': [0, 0.07]
                       }

    search = RandomizedSearchCV(XGBRegressor(), scoring=make_scorer(mean_squared_error), param_distributions=param_distributions, 
                       n_jobs=-1, cv=5, verbose=True, n_iter=35)

    search.fit(features, targets)

    best_params = search.best_params_
    cv_xgb = XGBRegressor(**best_params)
    
    
    def scorer(estimator, x, y):
        preds = estimator.predict(x)
        error = np.sqrt(mean_squared_error(preds, y))
        return error
    
    scores = cross_val_score(default_xgb, features, targets, cv=5, scoring=scorer)
    print("RMSE mean of 5-fold CV with default XGB: ", np.mean(scores))
    scores = cross_val_score(cv_xgb, features, targets, cv=5, scoring=scorer)
    print("RMSE mean of 5-fold CV with CV optimized XGB: ", np.mean(scores))

# df contains some basic features such as easting and northing, which have repeatedly proven to  be good features
# so add them in all
copernicus = df_columns + copernicus_columns
print("CV 5-fold RMSE using just copernicus data: \n")
test_different_models(full_data, copernicus)

climate = df_columns + climate_columns
print("CV 5-fold RMSE using just climate data: \n")
test_different_models(full_data, climate)

copernicus_and_climate = df_columns + copernicus_columns + climate_columns
print("CV 5-fold RMSE using copernicus and climate data: \n")
test_different_models(full_data, copernicus_and_climate)

soilgrids = df_columns + soilgrids_columns
print("CV 5-fold RMSE using copernicus and climate data: \n")
test_different_models(full_data, soilgrids)

# So many NAs in metsakeskus, not worth?
#metsakeskus = 
#print("CV 5-fold RMSE using copernicus and climate data: \n")
#test_different_models(full_data, metsakeskus)

soilgrids_and_climate = df_columns + soilgrids_columns + climate_columns
print("CV 5-fold RMSE using soilgrids and climate data: \n")
test_different_models(full_data, soilgrids_and_climate)

soilgrids_and_copernicus = df_columns + soilgrids_columns + copernicus_columns
print("CV 5-fold RMSE using soilgrids and copernicus data: \n")
test_different_models(full_data, soilgrids_and_copernicus)

all_data = df_columns + soilgrids_columns + climate_columns + copernicus_columns
print("CV 5-fold RMSE using all data: \n")
test_different_models(full_data, all_data)
118/19:
def test_different_models(data, feature_columns, target_columns = ["vol_pine"]):
    # Test different models with this data (mix different types, eg. soilgrids with metsakeskus and so on)
    
    from xgboost import XGBRegressor
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import mean_squared_error, make_scorer
    from sklearn.model_selection import RandomizedSearchCV
    from scipy.stats import uniform
    
    features = data[feature_columns].drop("plot_id", axis=1)
    print(features.columns)
    targets = data[target_columns]
    # Default XGB
    default_xgb = XGBRegressor(objective='reg:linear', nthread=-1)
    
    # Search best parameters by CV. Note: if all are lists, sampling is done without replacement? which is bad?
    param_distributions = {'max_depth': uniform(loc=3, scale=15),
                           #'learning_rate': uniform(loc=0.001, scale=0.1),
                           #'n_estimators': uniform(loc=100, scale=600),
                           'min_child_weight': [1, 2, 5],
                           #'colsample_bytree': uniform(loc=0.5, scale=1),
                           'reg_alpha': [0, 0.1, 0.2],
                           'reg_lambda': [0.7, 1],
                           'subsample': [0.8, 0.9],
                           'gamma': [0, 0.07]
                       }

    search = RandomizedSearchCV(XGBRegressor(), scoring=make_scorer(mean_squared_error), param_distributions=param_distributions, 
                       n_jobs=1, cv=5, verbose=True, n_iter=35)

    search.fit(features, targets)

    best_params = search.best_params_
    cv_xgb = XGBRegressor(**best_params)
    
    
    def scorer(estimator, x, y):
        preds = estimator.predict(x)
        error = np.sqrt(mean_squared_error(preds, y))
        return error
    
    scores = cross_val_score(default_xgb, features, targets, cv=5, scoring=scorer)
    print("RMSE mean of 5-fold CV with default XGB: ", np.mean(scores))
    scores = cross_val_score(cv_xgb, features, targets, cv=5, scoring=scorer)
    print("RMSE mean of 5-fold CV with CV optimized XGB: ", np.mean(scores))

# df contains some basic features such as easting and northing, which have repeatedly proven to  be good features
# so add them in all
copernicus = df_columns + copernicus_columns
print("CV 5-fold RMSE using just copernicus data: \n")
test_different_models(full_data, copernicus)

climate = df_columns + climate_columns
print("CV 5-fold RMSE using just climate data: \n")
test_different_models(full_data, climate)

copernicus_and_climate = df_columns + copernicus_columns + climate_columns
print("CV 5-fold RMSE using copernicus and climate data: \n")
test_different_models(full_data, copernicus_and_climate)

soilgrids = df_columns + soilgrids_columns
print("CV 5-fold RMSE using copernicus and climate data: \n")
test_different_models(full_data, soilgrids)

# So many NAs in metsakeskus, not worth?
#metsakeskus = 
#print("CV 5-fold RMSE using copernicus and climate data: \n")
#test_different_models(full_data, metsakeskus)

soilgrids_and_climate = df_columns + soilgrids_columns + climate_columns
print("CV 5-fold RMSE using soilgrids and climate data: \n")
test_different_models(full_data, soilgrids_and_climate)

soilgrids_and_copernicus = df_columns + soilgrids_columns + copernicus_columns
print("CV 5-fold RMSE using soilgrids and copernicus data: \n")
test_different_models(full_data, soilgrids_and_copernicus)

all_data = df_columns + soilgrids_columns + climate_columns + copernicus_columns
print("CV 5-fold RMSE using all data: \n")
test_different_models(full_data, all_data)
118/20:
def test_different_models(data, feature_columns, target_columns = ["vol_pine"]):
    # Test different models with this data (mix different types, eg. soilgrids with metsakeskus and so on)
    
    from xgboost import XGBRegressor
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import mean_squared_error, make_scorer
    from sklearn.model_selection import RandomizedSearchCV
    from scipy.stats import uniform
    
    features = data[feature_columns].drop("plot_id", axis=1)
    print(features.columns)
    targets = data[target_columns]
    # Default XGB
    default_xgb = XGBRegressor(objective='reg:linear', nthread=-1)
    
    maxdepth = uniform(loc=3, scale=15)
    # Search best parameters by CV. Note: if all are lists, sampling is done without replacement? which is bad?
    param_distributions = {'max_depth': maxdepth,
                           #'learning_rate': uniform(loc=0.001, scale=0.1),
                           #'n_estimators': uniform(loc=100, scale=600),
                           'min_child_weight': [1, 2, 5],
                           #'colsample_bytree': uniform(loc=0.5, scale=1),
                           'reg_alpha': [0, 0.1, 0.2],
                           'reg_lambda': [0.7, 1],
                           'subsample': [0.8, 0.9],
                           'gamma': [0, 0.07]
                       }

    search = RandomizedSearchCV(XGBRegressor(), scoring=make_scorer(mean_squared_error), param_distributions=param_distributions, 
                       n_jobs=1, cv=5, verbose=True, n_iter=35)

    search.fit(features, targets)

    best_params = search.best_params_
    cv_xgb = XGBRegressor(**best_params)
    
    
    def scorer(estimator, x, y):
        preds = estimator.predict(x)
        error = np.sqrt(mean_squared_error(preds, y))
        return error
    
    scores = cross_val_score(default_xgb, features, targets, cv=5, scoring=scorer)
    print("RMSE mean of 5-fold CV with default XGB: ", np.mean(scores))
    scores = cross_val_score(cv_xgb, features, targets, cv=5, scoring=scorer)
    print("RMSE mean of 5-fold CV with CV optimized XGB: ", np.mean(scores))

# df contains some basic features such as easting and northing, which have repeatedly proven to  be good features
# so add them in all
copernicus = df_columns + copernicus_columns
print("CV 5-fold RMSE using just copernicus data: \n")
test_different_models(full_data, copernicus)

climate = df_columns + climate_columns
print("CV 5-fold RMSE using just climate data: \n")
test_different_models(full_data, climate)

copernicus_and_climate = df_columns + copernicus_columns + climate_columns
print("CV 5-fold RMSE using copernicus and climate data: \n")
test_different_models(full_data, copernicus_and_climate)

soilgrids = df_columns + soilgrids_columns
print("CV 5-fold RMSE using copernicus and climate data: \n")
test_different_models(full_data, soilgrids)

# So many NAs in metsakeskus, not worth?
#metsakeskus = 
#print("CV 5-fold RMSE using copernicus and climate data: \n")
#test_different_models(full_data, metsakeskus)

soilgrids_and_climate = df_columns + soilgrids_columns + climate_columns
print("CV 5-fold RMSE using soilgrids and climate data: \n")
test_different_models(full_data, soilgrids_and_climate)

soilgrids_and_copernicus = df_columns + soilgrids_columns + copernicus_columns
print("CV 5-fold RMSE using soilgrids and copernicus data: \n")
test_different_models(full_data, soilgrids_and_copernicus)

all_data = df_columns + soilgrids_columns + climate_columns + copernicus_columns
print("CV 5-fold RMSE using all data: \n")
test_different_models(full_data, all_data)
118/21:
def test_different_models(data, feature_columns, target_columns = ["vol_pine"]):
    # Test different models with this data (mix different types, eg. soilgrids with metsakeskus and so on)
    
    from xgboost import XGBRegressor
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import mean_squared_error, make_scorer
    from sklearn.model_selection import RandomizedSearchCV
    from scipy.stats import uniform, randint
    
    features = data[feature_columns].drop("plot_id", axis=1)
    print(features.columns)
    targets = data[target_columns]
    # Default XGB
    default_xgb = XGBRegressor(objective='reg:linear', nthread=-1)
    
    maxdepth = uniform(loc=3, scale=15)
    # Search best parameters by CV. Note: if all are lists, sampling is done without replacement? which is bad?
    param_distributions = {'max_depth': randint(loc=3, scale=15),
                           #'learning_rate': uniform(loc=0.001, scale=0.1),
                           #'n_estimators': uniform(loc=100, scale=600),
                           'min_child_weight': [1, 2, 5],
                           #'colsample_bytree': uniform(loc=0.5, scale=1),
                           'reg_alpha': [0, 0.1, 0.2],
                           'reg_lambda': [0.7, 1],
                           'subsample': [0.8, 0.9],
                           'gamma': [0, 0.07]
                       }

    search = RandomizedSearchCV(XGBRegressor(), scoring=make_scorer(mean_squared_error), param_distributions=param_distributions, 
                       n_jobs=1, cv=5, verbose=True, n_iter=35)

    search.fit(features, targets)

    best_params = search.best_params_
    cv_xgb = XGBRegressor(**best_params)
    
    
    def scorer(estimator, x, y):
        preds = estimator.predict(x)
        error = np.sqrt(mean_squared_error(preds, y))
        return error
    
    scores = cross_val_score(default_xgb, features, targets, cv=5, scoring=scorer)
    print("RMSE mean of 5-fold CV with default XGB: ", np.mean(scores))
    scores = cross_val_score(cv_xgb, features, targets, cv=5, scoring=scorer)
    print("RMSE mean of 5-fold CV with CV optimized XGB: ", np.mean(scores))

# df contains some basic features such as easting and northing, which have repeatedly proven to  be good features
# so add them in all
copernicus = df_columns + copernicus_columns
print("CV 5-fold RMSE using just copernicus data: \n")
test_different_models(full_data, copernicus)

climate = df_columns + climate_columns
print("CV 5-fold RMSE using just climate data: \n")
test_different_models(full_data, climate)

copernicus_and_climate = df_columns + copernicus_columns + climate_columns
print("CV 5-fold RMSE using copernicus and climate data: \n")
test_different_models(full_data, copernicus_and_climate)

soilgrids = df_columns + soilgrids_columns
print("CV 5-fold RMSE using copernicus and climate data: \n")
test_different_models(full_data, soilgrids)

# So many NAs in metsakeskus, not worth?
#metsakeskus = 
#print("CV 5-fold RMSE using copernicus and climate data: \n")
#test_different_models(full_data, metsakeskus)

soilgrids_and_climate = df_columns + soilgrids_columns + climate_columns
print("CV 5-fold RMSE using soilgrids and climate data: \n")
test_different_models(full_data, soilgrids_and_climate)

soilgrids_and_copernicus = df_columns + soilgrids_columns + copernicus_columns
print("CV 5-fold RMSE using soilgrids and copernicus data: \n")
test_different_models(full_data, soilgrids_and_copernicus)

all_data = df_columns + soilgrids_columns + climate_columns + copernicus_columns
print("CV 5-fold RMSE using all data: \n")
test_different_models(full_data, all_data)
118/22:
def test_different_models(data, feature_columns, target_columns = ["vol_pine"]):
    # Test different models with this data (mix different types, eg. soilgrids with metsakeskus and so on)
    
    from xgboost import XGBRegressor
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import mean_squared_error, make_scorer
    from sklearn.model_selection import RandomizedSearchCV
    from scipy.stats import uniform, randint
    
    features = data[feature_columns].drop("plot_id", axis=1)
    print(features.columns)
    targets = data[target_columns]
    # Default XGB
    default_xgb = XGBRegressor(objective='reg:linear', nthread=-1)
    
    maxdepth = uniform(loc=3, scale=15)
    # Search best parameters by CV. Note: if all are lists, sampling is done without replacement? which is bad?
    param_distributions = {'max_depth': randint(3, 15),
                           #'learning_rate': uniform(loc=0.001, scale=0.1),
                           #'n_estimators': uniform(loc=100, scale=600),
                           'min_child_weight': [1, 2, 5],
                           #'colsample_bytree': uniform(loc=0.5, scale=1),
                           'reg_alpha': [0, 0.1, 0.2],
                           'reg_lambda': [0.7, 1],
                           'subsample': [0.8, 0.9],
                           'gamma': [0, 0.07]
                       }

    search = RandomizedSearchCV(XGBRegressor(), scoring=make_scorer(mean_squared_error), param_distributions=param_distributions, 
                       n_jobs=1, cv=5, verbose=True, n_iter=35)

    search.fit(features, targets)

    best_params = search.best_params_
    cv_xgb = XGBRegressor(**best_params)
    
    
    def scorer(estimator, x, y):
        preds = estimator.predict(x)
        error = np.sqrt(mean_squared_error(preds, y))
        return error
    
    scores = cross_val_score(default_xgb, features, targets, cv=5, scoring=scorer)
    print("RMSE mean of 5-fold CV with default XGB: ", np.mean(scores))
    scores = cross_val_score(cv_xgb, features, targets, cv=5, scoring=scorer)
    print("RMSE mean of 5-fold CV with CV optimized XGB: ", np.mean(scores))

# df contains some basic features such as easting and northing, which have repeatedly proven to  be good features
# so add them in all
copernicus = df_columns + copernicus_columns
print("CV 5-fold RMSE using just copernicus data: \n")
test_different_models(full_data, copernicus)

climate = df_columns + climate_columns
print("CV 5-fold RMSE using just climate data: \n")
test_different_models(full_data, climate)

copernicus_and_climate = df_columns + copernicus_columns + climate_columns
print("CV 5-fold RMSE using copernicus and climate data: \n")
test_different_models(full_data, copernicus_and_climate)

soilgrids = df_columns + soilgrids_columns
print("CV 5-fold RMSE using copernicus and climate data: \n")
test_different_models(full_data, soilgrids)

# So many NAs in metsakeskus, not worth?
#metsakeskus = 
#print("CV 5-fold RMSE using copernicus and climate data: \n")
#test_different_models(full_data, metsakeskus)

soilgrids_and_climate = df_columns + soilgrids_columns + climate_columns
print("CV 5-fold RMSE using soilgrids and climate data: \n")
test_different_models(full_data, soilgrids_and_climate)

soilgrids_and_copernicus = df_columns + soilgrids_columns + copernicus_columns
print("CV 5-fold RMSE using soilgrids and copernicus data: \n")
test_different_models(full_data, soilgrids_and_copernicus)

all_data = df_columns + soilgrids_columns + climate_columns + copernicus_columns
print("CV 5-fold RMSE using all data: \n")
test_different_models(full_data, all_data)
118/23:
def test_different_models(data, feature_columns, target_columns = ["vol_pine"]):
    # Test different models with this data (mix different types, eg. soilgrids with metsakeskus and so on)
    
    from xgboost import XGBRegressor
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import mean_squared_error, make_scorer
    from sklearn.model_selection import RandomizedSearchCV
    from scipy.stats import uniform, randint
    
    features = data[feature_columns].drop("plot_id", axis=1)
    print(features.columns)
    targets = data[target_columns]
    # Default XGB
    default_xgb = XGBRegressor(objective='reg:linear', nthread=-1)
    
    maxdepth = uniform(loc=3, scale=15)
    # Search best parameters by CV. Note: if all are lists, sampling is done without replacement? which is bad?
    param_distributions = {'max_depth': randint(3, 15),
                           'learning_rate': uniform(loc=0.001, scale=0.1),
                           #'n_estimators': uniform(loc=100, scale=600),
                           'min_child_weight': [1, 2, 5],
                           #'colsample_bytree': uniform(loc=0.5, scale=1),
                           'reg_alpha': [0, 0.1, 0.2],
                           'reg_lambda': [0.7, 1],
                           'subsample': [0.8, 0.9],
                           'gamma': [0, 0.07]
                       }

    search = RandomizedSearchCV(XGBRegressor(), scoring=make_scorer(mean_squared_error), param_distributions=param_distributions, 
                       n_jobs=1, cv=5, verbose=True, n_iter=35)

    search.fit(features, targets)

    best_params = search.best_params_
    cv_xgb = XGBRegressor(**best_params)
    
    
    def scorer(estimator, x, y):
        preds = estimator.predict(x)
        error = np.sqrt(mean_squared_error(preds, y))
        return error
    
    scores = cross_val_score(default_xgb, features, targets, cv=5, scoring=scorer)
    print("RMSE mean of 5-fold CV with default XGB: ", np.mean(scores))
    scores = cross_val_score(cv_xgb, features, targets, cv=5, scoring=scorer)
    print("RMSE mean of 5-fold CV with CV optimized XGB: ", np.mean(scores))

# df contains some basic features such as easting and northing, which have repeatedly proven to  be good features
# so add them in all
copernicus = df_columns + copernicus_columns
print("CV 5-fold RMSE using just copernicus data: \n")
test_different_models(full_data, copernicus)

climate = df_columns + climate_columns
print("CV 5-fold RMSE using just climate data: \n")
test_different_models(full_data, climate)

copernicus_and_climate = df_columns + copernicus_columns + climate_columns
print("CV 5-fold RMSE using copernicus and climate data: \n")
test_different_models(full_data, copernicus_and_climate)

soilgrids = df_columns + soilgrids_columns
print("CV 5-fold RMSE using copernicus and climate data: \n")
test_different_models(full_data, soilgrids)

# So many NAs in metsakeskus, not worth?
#metsakeskus = 
#print("CV 5-fold RMSE using copernicus and climate data: \n")
#test_different_models(full_data, metsakeskus)

soilgrids_and_climate = df_columns + soilgrids_columns + climate_columns
print("CV 5-fold RMSE using soilgrids and climate data: \n")
test_different_models(full_data, soilgrids_and_climate)

soilgrids_and_copernicus = df_columns + soilgrids_columns + copernicus_columns
print("CV 5-fold RMSE using soilgrids and copernicus data: \n")
test_different_models(full_data, soilgrids_and_copernicus)

all_data = df_columns + soilgrids_columns + climate_columns + copernicus_columns
print("CV 5-fold RMSE using all data: \n")
test_different_models(full_data, all_data)
118/24:
def test_different_models(data, feature_columns, target_columns = ["vol_pine"]):
    # Test different models with this data (mix different types, eg. soilgrids with metsakeskus and so on)
    
    from xgboost import XGBRegressor
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import mean_squared_error, make_scorer
    from sklearn.model_selection import RandomizedSearchCV
    from scipy.stats import uniform, randint
    
    features = data[feature_columns].drop("plot_id", axis=1)
    print(features.columns)
    targets = data[target_columns]
    # Default XGB
    default_xgb = XGBRegressor(objective='reg:linear', nthread=-1)
    
    maxdepth = uniform(loc=3, scale=15)
    # Search best parameters by CV. Note: if all are lists, sampling is done without replacement? which is bad?
    param_distributions = {'max_depth': randint(3, 15),
                           'learning_rate': uniform(loc=0.001, scale=0.1),
                           'n_estimators': uniform(loc=100, scale=600),
                           'min_child_weight': [1, 2, 5],
                           'colsample_bytree': uniform(loc=0.5, scale=1),
                           'reg_alpha': [0, 0.1, 0.2],
                           'reg_lambda': [0.7, 1],
                           'subsample': [0.8, 0.9],
                           'gamma': [0, 0.07]
                       }

    search = RandomizedSearchCV(XGBRegressor(), scoring=make_scorer(mean_squared_error), param_distributions=param_distributions, 
                       n_jobs=1, cv=5, verbose=True, n_iter=35)

    search.fit(features, targets)

    best_params = search.best_params_
    cv_xgb = XGBRegressor(**best_params)
    
    
    def scorer(estimator, x, y):
        preds = estimator.predict(x)
        error = np.sqrt(mean_squared_error(preds, y))
        return error
    
    scores = cross_val_score(default_xgb, features, targets, cv=5, scoring=scorer)
    print("RMSE mean of 5-fold CV with default XGB: ", np.mean(scores))
    scores = cross_val_score(cv_xgb, features, targets, cv=5, scoring=scorer)
    print("RMSE mean of 5-fold CV with CV optimized XGB: ", np.mean(scores))

# df contains some basic features such as easting and northing, which have repeatedly proven to  be good features
# so add them in all
copernicus = df_columns + copernicus_columns
print("CV 5-fold RMSE using just copernicus data: \n")
test_different_models(full_data, copernicus)

climate = df_columns + climate_columns
print("CV 5-fold RMSE using just climate data: \n")
test_different_models(full_data, climate)

copernicus_and_climate = df_columns + copernicus_columns + climate_columns
print("CV 5-fold RMSE using copernicus and climate data: \n")
test_different_models(full_data, copernicus_and_climate)

soilgrids = df_columns + soilgrids_columns
print("CV 5-fold RMSE using copernicus and climate data: \n")
test_different_models(full_data, soilgrids)

# So many NAs in metsakeskus, not worth?
#metsakeskus = 
#print("CV 5-fold RMSE using copernicus and climate data: \n")
#test_different_models(full_data, metsakeskus)

soilgrids_and_climate = df_columns + soilgrids_columns + climate_columns
print("CV 5-fold RMSE using soilgrids and climate data: \n")
test_different_models(full_data, soilgrids_and_climate)

soilgrids_and_copernicus = df_columns + soilgrids_columns + copernicus_columns
print("CV 5-fold RMSE using soilgrids and copernicus data: \n")
test_different_models(full_data, soilgrids_and_copernicus)

all_data = df_columns + soilgrids_columns + climate_columns + copernicus_columns
print("CV 5-fold RMSE using all data: \n")
test_different_models(full_data, all_data)
118/25:
def test_different_models(data, feature_columns, target_columns = ["vol_pine"]):
    # Test different models with this data (mix different types, eg. soilgrids with metsakeskus and so on)
    
    from xgboost import XGBRegressor
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import mean_squared_error, make_scorer
    from sklearn.model_selection import RandomizedSearchCV
    from scipy.stats import uniform, randint
    
    features = data[feature_columns].drop("plot_id", axis=1)
    print(features.columns)
    targets = data[target_columns]
    # Default XGB
    default_xgb = XGBRegressor(objective='reg:linear', nthread=-1)
    
    maxdepth = uniform(loc=3, scale=15)
    # Search best parameters by CV. Note: if all are lists, sampling is done without replacement? which is bad?
    param_distributions = {'max_depth': randint(3, 15),
                           'learning_rate': uniform(loc=0.001, scale=0.1),
                           'n_estimators': randint(100, 600),
                           'min_child_weight': [1, 2, 5],
                           'colsample_bytree': uniform(loc=0.5, scale=1),
                           'reg_alpha': [0, 0.1, 0.2],
                           'reg_lambda': [0.7, 1],
                           'subsample': [0.8, 0.9],
                           'gamma': [0, 0.07]
                       }

    search = RandomizedSearchCV(XGBRegressor(), scoring=make_scorer(mean_squared_error), param_distributions=param_distributions, 
                       n_jobs=1, cv=5, verbose=True, n_iter=35)

    search.fit(features, targets)

    best_params = search.best_params_
    cv_xgb = XGBRegressor(**best_params)
    
    
    def scorer(estimator, x, y):
        preds = estimator.predict(x)
        error = np.sqrt(mean_squared_error(preds, y))
        return error
    
    scores = cross_val_score(default_xgb, features, targets, cv=5, scoring=scorer)
    print("RMSE mean of 5-fold CV with default XGB: ", np.mean(scores))
    scores = cross_val_score(cv_xgb, features, targets, cv=5, scoring=scorer)
    print("RMSE mean of 5-fold CV with CV optimized XGB: ", np.mean(scores))

# df contains some basic features such as easting and northing, which have repeatedly proven to  be good features
# so add them in all
copernicus = df_columns + copernicus_columns
print("CV 5-fold RMSE using just copernicus data: \n")
test_different_models(full_data, copernicus)

climate = df_columns + climate_columns
print("CV 5-fold RMSE using just climate data: \n")
test_different_models(full_data, climate)

copernicus_and_climate = df_columns + copernicus_columns + climate_columns
print("CV 5-fold RMSE using copernicus and climate data: \n")
test_different_models(full_data, copernicus_and_climate)

soilgrids = df_columns + soilgrids_columns
print("CV 5-fold RMSE using copernicus and climate data: \n")
test_different_models(full_data, soilgrids)

# So many NAs in metsakeskus, not worth?
#metsakeskus = 
#print("CV 5-fold RMSE using copernicus and climate data: \n")
#test_different_models(full_data, metsakeskus)

soilgrids_and_climate = df_columns + soilgrids_columns + climate_columns
print("CV 5-fold RMSE using soilgrids and climate data: \n")
test_different_models(full_data, soilgrids_and_climate)

soilgrids_and_copernicus = df_columns + soilgrids_columns + copernicus_columns
print("CV 5-fold RMSE using soilgrids and copernicus data: \n")
test_different_models(full_data, soilgrids_and_copernicus)

all_data = df_columns + soilgrids_columns + climate_columns + copernicus_columns
print("CV 5-fold RMSE using all data: \n")
test_different_models(full_data, all_data)
118/26:
def test_different_models(data, feature_columns, target_columns = ["vol_pine"]):
    # Test different models with this data (mix different types, eg. soilgrids with metsakeskus and so on)
    
    from xgboost import XGBRegressor
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import mean_squared_error, make_scorer
    from sklearn.model_selection import RandomizedSearchCV
    from scipy.stats import uniform, randint
    
    features = data[feature_columns].drop("plot_id", axis=1)
    print(features.columns)
    targets = data[target_columns]
    # Default XGB
    default_xgb = XGBRegressor(objective='reg:linear', nthread=-1)
    
    maxdepth = uniform(loc=3, scale=15)
    # Search best parameters by CV. Note: if all are lists, sampling is done without replacement? which is bad?
    param_distributions = {'max_depth': randint(3, 15),
                           'learning_rate': uniform(loc=0.001, scale=0.1),
                           'n_estimators': randint(100, 600),
                           'min_child_weight': [1, 2, 5],
                           'colsample_bytree': uniform(loc=0.5, scale=0.5),
                           'reg_alpha': [0, 0.1, 0.2],
                           'reg_lambda': [0.7, 1],
                           'subsample': [0.8, 0.9],
                           'gamma': [0, 0.07]
                       }

    search = RandomizedSearchCV(XGBRegressor(), scoring=make_scorer(mean_squared_error), param_distributions=param_distributions, 
                       n_jobs=1, cv=5, verbose=True, n_iter=35)

    search.fit(features, targets)

    best_params = search.best_params_
    cv_xgb = XGBRegressor(**best_params)
    
    
    def scorer(estimator, x, y):
        preds = estimator.predict(x)
        error = np.sqrt(mean_squared_error(preds, y))
        return error
    
    scores = cross_val_score(default_xgb, features, targets, cv=5, scoring=scorer)
    print("RMSE mean of 5-fold CV with default XGB: ", np.mean(scores))
    scores = cross_val_score(cv_xgb, features, targets, cv=5, scoring=scorer)
    print("RMSE mean of 5-fold CV with CV optimized XGB: ", np.mean(scores))

# df contains some basic features such as easting and northing, which have repeatedly proven to  be good features
# so add them in all
copernicus = df_columns + copernicus_columns
print("CV 5-fold RMSE using just copernicus data: \n")
test_different_models(full_data, copernicus)

climate = df_columns + climate_columns
print("CV 5-fold RMSE using just climate data: \n")
test_different_models(full_data, climate)

copernicus_and_climate = df_columns + copernicus_columns + climate_columns
print("CV 5-fold RMSE using copernicus and climate data: \n")
test_different_models(full_data, copernicus_and_climate)

soilgrids = df_columns + soilgrids_columns
print("CV 5-fold RMSE using copernicus and climate data: \n")
test_different_models(full_data, soilgrids)

# So many NAs in metsakeskus, not worth?
#metsakeskus = 
#print("CV 5-fold RMSE using copernicus and climate data: \n")
#test_different_models(full_data, metsakeskus)

soilgrids_and_climate = df_columns + soilgrids_columns + climate_columns
print("CV 5-fold RMSE using soilgrids and climate data: \n")
test_different_models(full_data, soilgrids_and_climate)

soilgrids_and_copernicus = df_columns + soilgrids_columns + copernicus_columns
print("CV 5-fold RMSE using soilgrids and copernicus data: \n")
test_different_models(full_data, soilgrids_and_copernicus)

all_data = df_columns + soilgrids_columns + climate_columns + copernicus_columns
print("CV 5-fold RMSE using all data: \n")
test_different_models(full_data, all_data)
118/27:
def test_different_models(data, feature_columns, target_columns = ["vol_pine"]):
    # Test different models with this data (mix different types, eg. soilgrids with metsakeskus and so on)
    
    from xgboost import XGBRegressor
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import mean_squared_error, make_scorer
    from sklearn.model_selection import RandomizedSearchCV
    from scipy.stats import uniform, randint
    
    features = data[feature_columns].drop("plot_id", axis=1)
    print(features.columns)
    targets = data[target_columns]
    # Default XGB
    default_xgb = XGBRegressor(objective='reg:linear', nthread=-1)
    
    maxdepth = uniform(loc=3, scale=15)
    # Search best parameters by CV. Note: if all are lists, sampling is done without replacement? which is bad?
    param_distributions = {'max_depth': randint(3, 15),
                           'learning_rate': uniform(loc=0.001, scale=0.1),
                           'n_estimators': randint(100, 600),
                           'min_child_weight': [1, 2, 5],
                           'colsample_bytree': uniform(loc=0.5, scale=0.5),
                           'reg_alpha': [0, 0.1, 0.2],
                           'reg_lambda': [0.7, 1],
                           'subsample': [0.8, 0.9],
                           'gamma': [0, 0.07]
                       }

    search = RandomizedSearchCV(XGBRegressor(), scoring=make_scorer(mean_squared_error), param_distributions=param_distributions, 
                       n_jobs=-1, cv=5, verbose=True, n_iter=35)

    search.fit(features, targets)

    best_params = search.best_params_
    cv_xgb = XGBRegressor(**best_params)
    
    
    def scorer(estimator, x, y):
        preds = estimator.predict(x)
        error = np.sqrt(mean_squared_error(preds, y))
        return error
    
    scores = cross_val_score(default_xgb, features, targets, cv=5, scoring=scorer)
    print("RMSE mean of 5-fold CV with default XGB: ", np.mean(scores))
    scores = cross_val_score(cv_xgb, features, targets, cv=5, scoring=scorer)
    print("RMSE mean of 5-fold CV with CV optimized XGB: ", np.mean(scores))

# df contains some basic features such as easting and northing, which have repeatedly proven to  be good features
# so add them in all
copernicus = df_columns + copernicus_columns
print("CV 5-fold RMSE using just copernicus data: \n")
test_different_models(full_data, copernicus)

climate = df_columns + climate_columns
print("CV 5-fold RMSE using just climate data: \n")
test_different_models(full_data, climate)

copernicus_and_climate = df_columns + copernicus_columns + climate_columns
print("CV 5-fold RMSE using copernicus and climate data: \n")
test_different_models(full_data, copernicus_and_climate)

soilgrids = df_columns + soilgrids_columns
print("CV 5-fold RMSE using copernicus and climate data: \n")
test_different_models(full_data, soilgrids)

# So many NAs in metsakeskus, not worth?
#metsakeskus = 
#print("CV 5-fold RMSE using copernicus and climate data: \n")
#test_different_models(full_data, metsakeskus)

soilgrids_and_climate = df_columns + soilgrids_columns + climate_columns
print("CV 5-fold RMSE using soilgrids and climate data: \n")
test_different_models(full_data, soilgrids_and_climate)

soilgrids_and_copernicus = df_columns + soilgrids_columns + copernicus_columns
print("CV 5-fold RMSE using soilgrids and copernicus data: \n")
test_different_models(full_data, soilgrids_and_copernicus)

all_data = df_columns + soilgrids_columns + climate_columns + copernicus_columns
print("CV 5-fold RMSE using all data: \n")
test_different_models(full_data, all_data)
119/1:
%load_ext autoreload
%autoreload 2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import sys
import requests
pd.options.display.float_format = '{:,.2f}'.format
# Add path to where utils.py is so metrics can be imported
sys.path.insert(0, r'../regressors')
from data.data_loading import import_data, GeoAPI, split_from_ids
from data import data_loading
119/2:
stand_volumes = pd.read_csv("../../../data/harvester_data/ccgeodb_harvest_v_cell_volumes_smoothed.csv")
stand_polygons = pd.read_csv("../../../data/harvester_data/tblforestands_geom.csv")
119/3: stand_volumes[:2]
119/4: stand_volumes[:2]
119/5: stand_variances = stand_volumes.groupby("stand_id").var()
119/6:
stand_variances = stand_volumes.groupby("stand_id").var()
stand_data = stand_variances.merge(stand_polygons, left_on="stand_id", right_on="placeid")
119/7: stand_data[:2]
119/8:
stand_variances = stand_volumes.groupby("stand_id").var()
stand_data_temp = stand_variances.merge(stand_polygons, left_on="stand_id", right_on="placeid")
119/9: stand_data = stand_data_temp.drop(['fid', 'stand_group_id', 'placeid_parent'])
119/10: stand_data = stand_data_temp.drop(['fid', 'stand_group_id', 'placeid_parent'], axis=1)
119/11: stand_data[:2]
119/12: stand_variances_areas = stand_data_temp.drop(['fid', 'stand_group_id', 'placeid_parent'], axis=1)
119/13: stand_[:2]
119/14: stand_variances_areas[:2]
121/1:
%load_ext autoreload
%autoreload 2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import sys
import requests
pd.options.display.float_format = '{:,.2f}'.format
# Add path to where utils.py is so metrics can be imported
sys.path.insert(0, r'../regressors')
from data.data_loading import import_data, GeoAPI, split_from_ids
from data import data_loading
121/2:
stand_data = pd.read_csv("../../../data/harvest_FI/ccgeodb_harvest_koski_v_stand_level_features.csv")
gridcell_data = pd.read_csv("../../../data/harvest_FI/ccgeodb_harvest_koski_v_gridcell_volumes_with_coords.csv")
121/3: gridcell_data[:2]
121/4:
stand_data = pd.read_csv("../../../data/harvest_FI/ccgeodb_harvest_koski_v_stand_level_features.csv")
gridcell_data = pd.read_csv("../../../data/harvest_FI/ccgeodb_harvest_koski_v_gridcell_volumes_with_coords.csv")
gridcell_data = gridcell_data.drop('hila_polygon')
121/5:
stand_data = pd.read_csv("../../../data/harvest_FI/ccgeodb_harvest_koski_v_stand_level_features.csv")
gridcell_data = pd.read_csv("../../../data/harvest_FI/ccgeodb_harvest_koski_v_gridcell_volumes_with_coords.csv")
gridcell_data = gridcell_data.drop('hila_polygon', axis=1)
121/6: gridcell_data[:2]
121/7: stand_data
121/8: stand_data[:2]
121/9: stand_data.dtypes
121/10: stand_data[:2]
121/11: stand_data.prd_id.unique()
121/12: stand_data.prd_id.unique().len()
121/13: stand_data.prd_id.unique().len
121/14: len(stand_data.prd_id.unique())
121/15: stand_data[:2]
121/16: len(stand_data.stand_polygon_id.unique())
121/17: gridcell_data[:2]
121/18:
api = GeoAPI(default_locations=stand_data[['easting', 'northing']].values.tolist(),
                default_srid=3067, default_plot_ids=stand_data.prd_id.values.tolist())

def get_metsakeskus_data():
    columns_list = [["volumepine","volumespruce","volumedeciduous","volume","soiltype","fertilityclass","laserheight","laserdensity"]]

    schema_list = ['metsakeskus_hila']
    tables_list = ['gridcell']

    data = api.request_data(schema_list, tables_list, columns_list, batch_size=1000)

    # Return plot_ids from index to a column.
    data.reset_index(inplace=True)
    data = data.drop_duplicates(subset='plot_id')
    
    return data

def get_copernicus_data():
    tables_list = ["copernicus_dem", "copernicus_slope", "copernicus_aspect"]
    columns_list = [None]*len(tables_list)
    schema_list = ['physical']*len(tables_list)
    
    data = api.request_data(schema_list, tables_list, columns_list, batch_size=1000)
    data = data.reset_index()
    data = data.drop_duplicates(subset='plot_id')
        
    return data

def get_soilgrids_data():
    data = api.request_data(data_groups=['soilgrids'])
    data = data.reset_index()
    data = data.drop_duplicates(subset='plot_id')
        
    return data

def get_climate_data():
    data = api.request_data(data_groups=['climate_data'])
    data = data.reset_index()
    data = data.drop_duplicates(subset='plot_id')
        
    return data

metsakeskus_data = get_metsakeskus_data()
copernicus_data = get_copernicus_data()
#soilgrids_data = get_soilgrids_data()
#soilgrids_data = soilgrids_data.dropna() # only 47 rows missing, ok to drop nas
climate_data = get_climate_data()
121/19: metsakeskus_data[:2]
121/20: metsakeskus_data.nan()
121/21: metsakeskus_data.isna().sum()
121/22: metsakeskus_data[:2]
121/23:
api = GeoAPI(default_locations=stand_data[['easting', 'northing']].values.tolist(),
                default_srid=3067, default_plot_ids=stand_data.prd_id.values.tolist())

def get_metsakeskus_data():
    columns_list = [["volumepine","volumespruce","volumedeciduous","volume","soiltype","fertilityclass","laserheight","laserdensity"]]

    schema_list = ['metsakeskus_hila']
    tables_list = ['gridcell']

    data = api.request_data(schema_list, tables_list, columns_list, batch_size=1000)

    # Return plot_ids from index to a column.
    data.reset_index(inplace=True)
    data = data.drop_duplicates(subset='plot_id')
    
    return data

def get_copernicus_data():
    tables_list = ["copernicus_dem", "copernicus_slope", "copernicus_aspect"]
    columns_list = [None]*len(tables_list)
    schema_list = ['physical']*len(tables_list)
    
    data = api.request_data(schema_list, tables_list, columns_list, batch_size=1000)
    data = data.reset_index()
    data = data.drop_duplicates(subset='plot_id')
        
    return data

def get_soilgrids_data():
    data = api.request_data(data_groups=['soilgrids'])
    data = data.reset_index()
    data = data.drop_duplicates(subset='plot_id')
        
    return data

def get_climate_data():
    data = api.request_data(data_groups=['climate_data'])
    data = data.reset_index()
    data = data.drop_duplicates(subset='plot_id')
        
    return data

metsakeskus_data = get_metsakeskus_data()
copernicus_data = get_copernicus_data()
soilgrids_data = get_soilgrids_data()
#soilgrids_data = soilgrids_data.dropna() # only 47 rows missing, ok to drop nas
climate_data = get_climate_data()
121/24:
%load_ext autoreload
%autoreload 2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import sys
import seaborn as sns
import requests
pd.options.display.float_format = '{:,.2f}'.format
# Add path to where utils.py is so metrics can be imported
sys.path.insert(0, r'../regressors')
from data.data_loading import import_data, GeoAPI, split_from_ids
from data import data_loading
121/25:
sns.distplot(stand_data.total_m3_ha, label='Stand data Total Volume Distribution')
#sns.distplot(testing.vol_total, label='Test Set Total Volume Distribution')

plt.legend()
126/1:
%load_ext autoreload
%autoreload 2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import sys
import requests
import seaborn as sns
pd.options.display.float_format = '{:,.2f}'.format
# Add path to where some utilities are so they can be imported
sys.path.insert(0, r'../regressors')
from data.data_loading import import_data, GeoAPI, split_from_ids
from data import data_loading
126/2:
# load SLU data

slu_plots_since_2015 = pd.read_csv("../../../data/terramonitor_verification/ccgeodb_se_slu_v_slu_plots_since_2015_terramonitor.csv")
slu_plots_with_distance = pd.read_csv("../../../data/terramonitor_verification/ccgeodb_se_slu_v_slu_plots_since_2015_with_distance.csv")
126/3: slu_plots_since_2015[:2]
126/4:
api = GeoAPI(default_locations=slu_plots_since_2015[['longitude', 'latitude']].values.tolist(),
                default_srid=4326, default_plot_ids=slu_plots_since_2015.plot_id.values.tolist())

def get_terramonitor_predictions():
    tables_list = ["se_volumes_m3_ha", "se_pine_percent", "se_spruce_percent", "se_deciduous_percent"]
    columns_list = [None]*len(tables_list)
    schema_list = ['terramonitor']*len(tables_list)
    
    data = api.request_data(schema_list, tables_list, columns_list, batch_size=1000)
    data = data.reset_index()
    data = data.drop_duplicates(subset='plot_id')
        
    return data

terramonitor_predictions = get_terramonitor_predictions()
126/5: terramonitor_predictions[:2]
126/6: terramonitor_predictions
126/7: terramonitor_predictions.shape
126/8: terramonitor_predictions.isna().sum()
126/9:
api = GeoAPI(default_locations=slu_plots_since_2015[['latitude', 'longitude']].values.tolist(),
                default_srid=4326, default_plot_ids=slu_plots_since_2015.plot_id.values.tolist())

def get_terramonitor_predictions():
    tables_list = ["se_volumes_m3_ha", "se_pine_percent", "se_spruce_percent", "se_deciduous_percent"]
    columns_list = [None]*len(tables_list)
    schema_list = ['terramonitor']*len(tables_list)
    
    data = api.request_data(schema_list, tables_list, columns_list, batch_size=1000)
    data = data.reset_index()
    data = data.drop_duplicates(subset='plot_id')
        
    return data

terramonitor_predictions = get_terramonitor_predictions()
126/10: terramonitor_predictions[:2]
126/11:
api = GeoAPI(default_locations=slu_plots_since_2015[['longitude', 'latitude']].values.tolist(),
                default_srid=4326, default_plot_ids=slu_plots_since_2015.plot_id.values.tolist())

def get_terramonitor_predictions():
    tables_list = ["se_volumes_m3_ha", "se_pine_percent", "se_spruce_percent", "se_deciduous_percent"]
    columns_list = [None]*len(tables_list)
    schema_list = ['terramonitor']*len(tables_list)
    
    data = api.request_data(schema_list, tables_list, columns_list, batch_size=1000)
    data = data.reset_index()
    data = data.drop_duplicates(subset='plot_id')
        
    return data

terramonitor_predictions = get_terramonitor_predictions()
126/12: terramonitor_predictions[:2]
126/13: slu_plots_since_2015.shape
126/14: slu_plots_with_distance[:2]
126/15: slu_plots_with_distance[slu_plots_with_distance['distance_km_from_kastet'] < 100].shape
126/16: slu_plots_since_2015[:2]
126/17: slu_plots_veri = slu_plots_with_distance[slu_plots_with_distance['distance_km_from_kastet'] < 100].shape
126/18:
api = GeoAPI(default_locations=slu_plots_veri[['longitude', 'latitude']].values.tolist(),
                default_srid=4326, default_plot_ids=slu_plots_veri.plot_id.values.tolist())

def get_terramonitor_predictions():
    tables_list = ["se_volumes_m3_ha", "se_pine_percent", "se_spruce_percent", "se_deciduous_percent"]
    columns_list = [None]*len(tables_list)
    schema_list = ['terramonitor']*len(tables_list)
    
    data = api.request_data(schema_list, tables_list, columns_list, batch_size=1000)
    data = data.reset_index()
    data = data.drop_duplicates(subset='plot_id')
        
    return data

terramonitor_predictions = get_terramonitor_predictions()
126/19: slu_plots_veri = slu_plots_with_distance[slu_plots_with_distance['distance_km_from_kastet'] < 100]
126/20:
api = GeoAPI(default_locations=slu_plots_veri[['longitude', 'latitude']].values.tolist(),
                default_srid=4326, default_plot_ids=slu_plots_veri.plot_id.values.tolist())

def get_terramonitor_predictions():
    tables_list = ["se_volumes_m3_ha", "se_pine_percent", "se_spruce_percent", "se_deciduous_percent"]
    columns_list = [None]*len(tables_list)
    schema_list = ['terramonitor']*len(tables_list)
    
    data = api.request_data(schema_list, tables_list, columns_list, batch_size=1000)
    data = data.reset_index()
    data = data.drop_duplicates(subset='plot_id')
        
    return data

terramonitor_predictions = get_terramonitor_predictions()
126/21: slu_plots_with_distance[:2]
126/22:
api = GeoAPI(default_locations=slu_plots_veri[['lon', 'lat']].values.tolist(),
                default_srid=4326, default_plot_ids=slu_plots_veri.plot_id.values.tolist())

def get_terramonitor_predictions():
    tables_list = ["se_volumes_m3_ha", "se_pine_percent", "se_spruce_percent", "se_deciduous_percent"]
    columns_list = [None]*len(tables_list)
    schema_list = ['terramonitor']*len(tables_list)
    
    data = api.request_data(schema_list, tables_list, columns_list, batch_size=1000)
    data = data.reset_index()
    data = data.drop_duplicates(subset='plot_id')
        
    return data

terramonitor_predictions = get_terramonitor_predictions()
126/23: terramonitor_predictions[:2]
126/24: terramonitor_predictions.isna().sum()
126/25: slu_plots_veri[:2]
126/26:
trees = ['se_pine_percent', 'se_spruce_percent', 'se_deciduous_percent']
terramonitor_predictions[trees]
126/27:
trees = ['se_pine_percent', 'se_spruce_percent', 'se_deciduous_percent']
terramonitor_predictions[trees] * terramonitor_predictions[['se_volumes_m3_ha']]
126/28:
trees = ['se_pine_percent', 'se_spruce_percent', 'se_deciduous_percent']
terramonitor_predictions[trees] * terramonitor_predictions['se_volumes_m3_ha']
126/29:
trees = ['se_pine_percent', 'se_spruce_percent', 'se_deciduous_percent']
terramonitor_predictions[trees] * terramonitor_predictions['se_volumes_m3_ha'].vlues
126/30:
trees = ['se_pine_percent', 'se_spruce_percent', 'se_deciduous_percent']
terramonitor_predictions[trees] * terramonitor_predictions['se_volumes_m3_ha'].values
126/31: terramonitor_predictions['se_volumes_m3_ha'].shape
126/32: terramonitor_predictions['se_volumes_m3_ha'].values.shape
126/33:
trees = ['se_pine_percent', 'se_spruce_percent', 'se_deciduous_percent']
terramonitor_predictions[trees] * terramonitor_predictions[['se_volumes_m3_ha']].values
126/34:
trees = ['se_pine_percent', 'se_spruce_percent', 'se_deciduous_percent']
terramonitor_predictions[trees].values * terramonitor_predictions[['se_volumes_m3_ha']].values
126/35: terramonitor_predictions[['se_volumes_m3_ha']].values.shape
126/36: terramonitor_predictions[trees].values.shape
126/37: terramonitor_predictions[['se_volumes_m3_ha']].values
126/38: terramonitor_predictions[trees].values
126/39:
trees = ['se_pine_percent', 'se_spruce_percent', 'se_deciduous_percent']
(terramonitor_predictions[trees].values/100) * terramonitor_predictions[['se_volumes_m3_ha']].values
126/40: slu_plots_veri[:2]
126/41: slu_plots_veri.isna().sum()
126/42: slu_plots_veri = slu_plots_with_distance[slu_plots_with_distance['distance_km_from_kastet'] < 100].dropna()
126/43:
api = GeoAPI(default_locations=slu_plots_veri[['lon', 'lat']].values.tolist(),
                default_srid=4326, default_plot_ids=slu_plots_veri.plot_id.values.tolist())

def get_terramonitor_predictions():
    tables_list = ["se_volumes_m3_ha", "se_pine_percent", "se_spruce_percent", "se_deciduous_percent"]
    columns_list = [None]*len(tables_list)
    schema_list = ['terramonitor']*len(tables_list)
    
    data = api.request_data(schema_list, tables_list, columns_list, batch_size=1000)
    data = data.reset_index()
    data = data.drop_duplicates(subset='plot_id')
        
    return data

terramonitor_predictions = get_terramonitor_predictions()
126/44:
trees = ['se_pine_percent', 'se_spruce_percent', 'se_deciduous_percent']
(terramonitor_predictions[trees].values/100) * terramonitor_predictions[['se_volumes_m3_ha']].values
126/45: slu_plots_veri[:2]
126/46:
trees_slu = ['ratio_pine', 'ratio_spruce', 'ratio_deciduous']
slu_plots_veri[trees_slu].values * slu_plots_veri[['volume']].values
126/47: slu_plots_veri[:2]
126/48:
trees_terra = ['se_pine_percent', 'se_spruce_percent', 'se_deciduous_percent']
terra_volumes = (terramonitor_predictions[trees_terra].values/100) * terramonitor_predictions[['se_volumes_m3_ha']].values
126/49:
trees_slu = ['ratio_pine', 'ratio_spruce', 'ratio_deciduous']
slu_volumes = slu_plots_veri[trees_slu].values * slu_plots_veri[['volume']].values
126/50:
from sklearn.metrics import mean_squared_error

np.sqrt(mean_squared_error(terra_volumes, slu_volumes, multioutput='raw_values'))
126/51: np.mean(terra_volumes, axis=1)
126/52: np.mean(terra_volumes, axis=0)
126/53:
trees_terra = ['se_pine_percent', 'se_spruce_percent', 'se_deciduous_percent']
terra_volumes = (terramonitor_predictions[trees_terra].values/100) * terramonitor_predictions[['se_volumes_m3_ha']].values
terra_volumes = np.hstack(terramonitor_predictions[['se_volumes_m3_ha']].values, terra_volumes)
126/54:
trees_terra = ['se_pine_percent', 'se_spruce_percent', 'se_deciduous_percent']
terra_volumes = (terramonitor_predictions[trees_terra].values/100) * terramonitor_predictions[['se_volumes_m3_ha']].values
terra_volumes = np.hstack([terramonitor_predictions[['se_volumes_m3_ha']].values, terra_volumes])
126/55:
trees_terra = ['se_pine_percent', 'se_spruce_percent', 'se_deciduous_percent']
terra_volumes = (terramonitor_predictions[trees_terra].values/100) * terramonitor_predictions[['se_volumes_m3_ha']].values
terra_volumes = np.hstack([terramonitor_predictions[['se_volumes_m3_ha']].values, terra_volumes])
126/56:
trees_slu = ['ratio_pine', 'ratio_spruce', 'ratio_deciduous']
slu_volumes = slu_plots_veri[trees_slu].values * slu_plots_veri[['volume']].values
terra_volumes = np.hstack([slu_plots_veri[['volume']].values, slu_volumes])
126/57:
from sklearn.metrics import mean_squared_error
terra_means = np.mean(terra_volumes, axis=0)
slu_means = np.mean(slu_volumes, axis=0)
rmse = np.sqrt(mean_squared_error(terra_volumes, slu_volumes, multioutput='raw_values'))
nrmse = rmse / slu_means
126/58:
trees_terra = ['se_pine_percent', 'se_spruce_percent', 'se_deciduous_percent']
terra_volumes = (terramonitor_predictions[trees_terra].values/100) * terramonitor_predictions[['se_volumes_m3_ha']].values
terra_volumes = np.hstack([terramonitor_predictions[['se_volumes_m3_ha']].values, terra_volumes])
126/59:
trees_slu = ['ratio_pine', 'ratio_spruce', 'ratio_deciduous']
slu_volumes = slu_plots_veri[trees_slu].values * slu_plots_veri[['volume']].values
slu_volumes = np.hstack([slu_plots_veri[['volume']].values, slu_volumes])
126/60:
from sklearn.metrics import mean_squared_error
terra_means = np.mean(terra_volumes, axis=0)
slu_means = np.mean(slu_volumes, axis=0)
rmse = np.sqrt(mean_squared_error(terra_volumes, slu_volumes, multioutput='raw_values'))
nrmse = rmse / slu_means
126/61: nrmse
126/62:
from sklearn.metrics import mean_squared_error
terra_means = np.mean(terra_volumes, axis=0)
slu_means = np.mean(slu_volumes, axis=0)
rmse = np.sqrt(mean_squared_error(terra_volumes, slu_volumes, multioutput='raw_values'))
nrmse = rmse / slu_means * 100
126/63: nrmse
126/64: rmse
126/65: nrmse
128/1:
print(slu_plots_since_2015.shape)
print(slu_plots_with_distance.shape)
128/2:
%load_ext autoreload
%autoreload 2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import sys
import requests
import seaborn as sns
pd.options.display.float_format = '{:,.2f}'.format
# Add path to where some utilities are so they can be imported
sys.path.insert(0, r'../regressors')
from data.data_loading import import_data, GeoAPI, split_from_ids
from data import data_loading
128/3:
# load SLU data

slu_plots_since_2015 = pd.read_csv("../../../data/terramonitor_verification/ccgeodb_se_slu_v_slu_plots_since_2015_terramonitor.csv")
slu_plots_with_distance = pd.read_csv("../../../data/terramonitor_verification/ccgeodb_se_slu_v_slu_plots_since_2015_with_distance.csv")
128/4:
print(slu_plots_since_2015.shape)
print(slu_plots_with_distance.shape)
130/1:
%load_ext autoreload
%autoreload 2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import sys
import requests
import seaborn as sns
pd.options.display.float_format = '{:,.2f}'.format
# Add path to where some utilities are so they can be imported
sys.path.insert(0, r'../regressors')
from data.data_loading import import_data, GeoAPI, split_from_ids
from data import data_loading
130/2:
# load SLU data

slu_plots_since_2015 = pd.read_csv("../../../data/terramonitor_verification/ccgeodb_se_slu_v_slu_plots_since_2015_terramonitor.csv")
slu_plots_with_distance = pd.read_csv("../../../data/terramonitor_verification/ccgeodb_se_slu_v_slu_plots_since_2015_with_distance.csv")
130/3:
print(slu_plots_since_2015.shape)
print(slu_plots_with_distance.shape)
130/4:
#TODO: get all values for all rows, filter to test set later.
api = GeoAPI(default_locations=slu_plots_with_distance[['lon', 'lat']].values.tolist(),
                default_srid=4326, default_plot_ids=slu_plots_with_distance.plot_id.values.tolist())

def get_terramonitor_predictions():
    tables_list = ["se_volumes_m3_ha", "se_pine_percent", "se_spruce_percent", "se_deciduous_percent"]
    columns_list = [None]*len(tables_list)
    schema_list = ['terramonitor']*len(tables_list)
    
    data = api.request_data(schema_list, tables_list, columns_list, batch_size=1000)
    data = data.reset_index()
    data = data.drop_duplicates(subset='plot_id')
        
    return data

def get_metsakeskus_data():
    columns_list = [["volumepine","volumespruce","volumedeciduous","volume","soiltype","fertilityclass","laserheight","laserdensity"]]

    schema_list = ['metsakeskus_hila']
    tables_list = ['gridcell']

    data = api.request_data(schema_list, tables_list, columns_list, batch_size=1000)

    # Return plot_ids from index to a column.
    data.reset_index(inplace=True)
    data = data.drop_duplicates(subset='plot_id')
    
    return data

def get_copernicus_data():
    tables_list = ["copernicus_dem", "copernicus_slope", "copernicus_aspect", "copernicus_leaf_type", "copernicus_tree_cover"]
    columns_list = [None]*len(tables_list)
    schema_list = ['physical']*len(tables_list)
    
    data = api.request_data(schema_list, tables_list, columns_list, batch_size=1000)
    data = data.reset_index()
    data = data.drop_duplicates(subset='plot_id')
        
    return data

def get_physical_data():
    tables_list = ["elev_16m_hila_grid", "aspect_16m_hila_grid", "slope_16m_hila_grid"]
    columns_list = [None]*len(tables_list)
    schema_list = ['physical']*len(tables_list)
    
    data = api.request_data(schema_list, tables_list, columns_list, batch_size=1000)
    data = data.reset_index()
    data = data.drop_duplicates(subset='plot_id')
        
    return data

def get_soilgrids_data():
    data = api.request_data(data_groups=['soilgrids'])
    data = data.reset_index()
    data = data.drop_duplicates(subset='plot_id')
        
    return data

def get_climate_data():
    data = api.request_data(data_groups=['climate_data'])
    data = data.reset_index()
    data = data.drop_duplicates(subset='plot_id')
        
    return data

metsakeskus_data = get_metsakeskus_data()
copernicus_data = get_copernicus_data()
physical_data = get_physical_data()
soilgrids_data = get_soilgrids_data()
soilgrids_data = soilgrids_data.dropna()
climate_data = get_climate_data()

terramonitor_predictions = get_terramonitor_predictions()
131/1:
%load_ext autoreload
%autoreload 2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import sys
import requests
import seaborn as sns
pd.options.display.float_format = '{:,.2f}'.format
# Add path to where some utilities are so they can be imported
sys.path.insert(0, r'../regressors')
from data.data_loading import import_data, GeoAPI, split_from_ids
from data import data_loading
131/2:
# load SLU data

slu_plots_since_2015 = pd.read_csv("../../../data/terramonitor_verification/ccgeodb_se_slu_v_slu_plots_since_2015_terramonitor.csv")
slu_plots_with_distance = pd.read_csv("../../../data/terramonitor_verification/ccgeodb_se_slu_v_slu_plots_since_2015_with_distance.csv")
131/3:
print(slu_plots_since_2015.shape)
print(slu_plots_with_distance.shape)
131/4:
#TODO: get all values for all rows, filter to test set later.
api = GeoAPI(default_locations=slu_plots_with_distance[['lon', 'lat']].values.tolist(),
                default_srid=4326, default_plot_ids=slu_plots_with_distance.plot_id.values.tolist())

def get_terramonitor_predictions():
    tables_list = ["se_volumes_m3_ha", "se_pine_percent", "se_spruce_percent", "se_deciduous_percent"]
    columns_list = [None]*len(tables_list)
    schema_list = ['terramonitor']*len(tables_list)
    
    data = api.request_data(schema_list, tables_list, columns_list, batch_size=1000)
    data = data.reset_index()
    data = data.drop_duplicates(subset='plot_id')
        
    return data

def get_copernicus_data():
    tables_list = ["copernicus_dem", "copernicus_slope", "copernicus_aspect", "copernicus_leaf_type", "copernicus_tree_cover"]
    columns_list = [None]*len(tables_list)
    schema_list = ['physical']*len(tables_list)
    
    data = api.request_data(schema_list, tables_list, columns_list, batch_size=1000)
    data = data.reset_index()
    data = data.drop_duplicates(subset='plot_id')
        
    return data

def get_physical_data():
    tables_list = ["elev_16m_hila_grid", "aspect_16m_hila_grid", "slope_16m_hila_grid"]
    columns_list = [None]*len(tables_list)
    schema_list = ['physical']*len(tables_list)
    
    data = api.request_data(schema_list, tables_list, columns_list, batch_size=1000)
    data = data.reset_index()
    data = data.drop_duplicates(subset='plot_id')
        
    return data

def get_soilgrids_data():
    data = api.request_data(data_groups=['soilgrids'])
    data = data.reset_index()
    data = data.drop_duplicates(subset='plot_id')
        
    return data

def get_climate_data():
    data = api.request_data(data_groups=['climate_data'])
    data = data.reset_index()
    data = data.drop_duplicates(subset='plot_id')
        
    return data

copernicus_data = get_copernicus_data()
physical_data = get_physical_data()
soilgrids_data = get_soilgrids_data()
climate_data = get_climate_data()

terramonitor_predictions = get_terramonitor_predictions()
131/5: soilgrids_data[:2]
131/6: soilgrids_data.shape
131/7: soilgrids_data.isna().sum()
131/8: slu_plots_with_distance.shape
131/9: slu_plots_with_distance[2]
131/10: slu_plots_with_distance[:2]
131/11:
# Merge all data into one dataframe
copernicus_columns = list(copernicus_data.columns)
soilgrids_columns = list(soilgrids_data.columns)
physical_columns = list(physical_data.columns)
climate_columns = list(climate_data.columns)

full_data = stand_data.merge(sku_plots_with_distance, on='plot_id').\
merge(copernicus_data, on="plot_id").\
merge(soilgrids_data, on="plot_id").\
merge(physical_data, on="plot_id").\
merge(climate_data, on="plot_id")
131/12:
# Merge all data into one dataframe
copernicus_columns = list(copernicus_data.columns)
soilgrids_columns = list(soilgrids_data.columns)
physical_columns = list(physical_data.columns)
climate_columns = list(climate_data.columns)

full_data = slu_plots_with_distance.merge(copernicus_data, on="plot_id").\
merge(soilgrids_data, on="plot_id").\
merge(physical_data, on="plot_id").\
merge(climate_data, on="plot_id")
131/13: full_data[:2]
131/14: full_data.isna().sum()
131/15: terramonitor_predictions.isna().sum()
131/16:
# Merge all data into one dataframe
copernicus_columns = list(copernicus_data.columns)
soilgrids_columns = list(soilgrids_data.columns)
physical_columns = list(physical_data.columns)
climate_columns = list(climate_data.columns)

full_data = slu_plots_with_distance.merge(copernicus_data, on="plot_id").\
merge(soilgrids_data, on="plot_id").\
merge(physical_data, on="plot_id").\
merge(climate_data, on="plot_id").\
merge(terramonitor_predictions, on="plot_id")
131/17: full_data[full_data['distance_km_from_kastet'] > 100 & full_data['distance_km_from_kastet'] < 300]
131/18: full_data[full_data['distance_km_from_kastet'] > 100 && full_data['distance_km_from_kastet'] < 300]
131/19: full_data[(full_data['distance_km_from_kastet'] > 100) & (full_data['distance_km_from_kastet'] < 300)]
131/20: full_data[(full_data['distance_km_from_kastet'] > 100) & (full_data['distance_km_from_kastet'] < 300)]
131/21: full_data[(full_data['distance_km_from_kastet'] > 100) && (full_data['distance_km_from_kastet'] < 300)]
131/22: full_data[(full_data['distance_km_from_kastet'] > 100) and (full_data['distance_km_from_kastet'] < 300)]
131/23: full_data[(full_data['distance_km_from_kastet'] > 100) & (full_data['distance_km_from_kastet'] < 300)]
131/24:
# Filter data to train and test:
train_set = full_data[(full_data['distance_km_from_kastet'] > 100) & (full_data['distance_km_from_kastet'] < 300)]
test_set = full_data[full_data['distance_km_from_kastet'] < 100]
print("Training set: plots within 300km but outside 100km of Kastet. Number of plots in training: %d" % )
print("Testing set: plots within 100km of Kastet. Number of plots in test: %d" % )
131/25:
# Filter data to train and test:
train_set = full_data[(full_data['distance_km_from_kastet'] > 100) & (full_data['distance_km_from_kastet'] < 300)]
test_set = full_data[full_data['distance_km_from_kastet'] < 100]
print("Training set: plots within 300km but outside 100km of Kastet. Number of plots in training: %d" % len(train_set))
print("Testing set: plots within 100km of Kastet. Number of plots in test: %d" % len(test_set))
131/26: test_set.isna().sum()
131/27:
#TODO: get all values for all rows, filter to test set later.
api = GeoAPI(default_locations=slu_plots_with_distance[['lon', 'lat']].values.tolist(),
                default_srid=4326, default_plot_ids=slu_plots_with_distance.plot_id.values.tolist())

def get_terramonitor_predictions():
    tables_list = ["se_volumes_m3_ha", "se_pine_percent", "se_spruce_percent", "se_deciduous_percent"]
    columns_list = [None]*len(tables_list)
    schema_list = ['terramonitor']*len(tables_list)
    
    data = api.request_data(schema_list, tables_list, columns_list, batch_size=1000)
    data = data.reset_index()
    data = data.drop_duplicates(subset='plot_id')
        
    return data

def get_copernicus_data():
    tables_list = ["copernicus_dem", "copernicus_slope", "copernicus_aspect", "copernicus_leaf_type", "copernicus_tree_cover"]
    columns_list = [None]*len(tables_list)
    schema_list = ['physical']*len(tables_list)
    
    data = api.request_data(schema_list, tables_list, columns_list, batch_size=1000)
    data = data.reset_index()
    data = data.drop_duplicates(subset='plot_id')
        
    return data

def get_soilgrids_data():
    data = api.request_data(data_groups=['soilgrids'])
    data = data.reset_index()
    data = data.drop_duplicates(subset='plot_id')
        
    return data

def get_climate_data():
    data = api.request_data(data_groups=['climate_data'])
    data = data.reset_index()
    data = data.drop_duplicates(subset='plot_id')
        
    return data

copernicus_data = get_copernicus_data()
physical_data = get_physical_data()
soilgrids_data = get_soilgrids_data()
climate_data = get_climate_data()

terramonitor_predictions = get_terramonitor_predictions()
132/1:
%load_ext autoreload
%autoreload 2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import sys
import requests
import seaborn as sns
pd.options.display.float_format = '{:,.2f}'.format
# Add path to where some utilities are so they can be imported
sys.path.insert(0, r'../regressors')
from data.data_loading import import_data, GeoAPI, split_from_ids
from data import data_loading
132/2:
# load SLU data

slu_plots_since_2015 = pd.read_csv("../../../data/terramonitor_verification/ccgeodb_se_slu_v_slu_plots_since_2015_terramonitor.csv")
slu_plots_with_distance = pd.read_csv("../../../data/terramonitor_verification/ccgeodb_se_slu_v_slu_plots_since_2015_with_distance.csv")
132/3:
print(slu_plots_since_2015.shape)
print(slu_plots_with_distance.shape)
132/4:
#TODO: get all values for all rows, filter to test set later.
api = GeoAPI(default_locations=slu_plots_with_distance[['lon', 'lat']].values.tolist(),
                default_srid=4326, default_plot_ids=slu_plots_with_distance.plot_id.values.tolist())

def get_terramonitor_predictions():
    tables_list = ["se_volumes_m3_ha", "se_pine_percent", "se_spruce_percent", "se_deciduous_percent"]
    columns_list = [None]*len(tables_list)
    schema_list = ['terramonitor']*len(tables_list)
    
    data = api.request_data(schema_list, tables_list, columns_list, batch_size=1000)
    data = data.reset_index()
    data = data.drop_duplicates(subset='plot_id')
        
    return data

def get_copernicus_data():
    tables_list = ["copernicus_dem", "copernicus_slope", "copernicus_aspect", "copernicus_leaf_type", "copernicus_tree_cover"]
    columns_list = [None]*len(tables_list)
    schema_list = ['physical']*len(tables_list)
    
    data = api.request_data(schema_list, tables_list, columns_list, batch_size=1000)
    data = data.reset_index()
    data = data.drop_duplicates(subset='plot_id')
        
    return data

def get_soilgrids_data():
    data = api.request_data(data_groups=['soilgrids'])
    data = data.reset_index()
    data = data.drop_duplicates(subset='plot_id')
        
    return data

def get_climate_data():
    data = api.request_data(data_groups=['climate_data'])
    data = data.reset_index()
    data = data.drop_duplicates(subset='plot_id')
        
    return data

copernicus_data = get_copernicus_data()
soilgrids_data = get_soilgrids_data()
climate_data = get_climate_data()

terramonitor_predictions = get_terramonitor_predictions()
132/5:
# Merge all data into one dataframe
copernicus_columns = list(copernicus_data.columns)
soilgrids_columns = list(soilgrids_data.columns)
climate_columns = list(climate_data.columns)

full_data = slu_plots_with_distance.merge(copernicus_data, on="plot_id").\
merge(soilgrids_data, on="plot_id").\
merge(climate_data, on="plot_id").\
merge(terramonitor_predictions, on="plot_id")
132/6:
# Filter data to train and test:
train_set = full_data[(full_data['distance_km_from_kastet'] > 100) & (full_data['distance_km_from_kastet'] < 300)]
test_set = full_data[full_data['distance_km_from_kastet'] < 100]
print("Training set: plots within 300km but outside 100km of Kastet. Number of plots in training: %d" % len(train_set))
print("Testing set: plots within 100km of Kastet. Number of plots in test: %d" % len(test_set))
132/7:
feature_columns = copernicus_columns + soilgrids_columns + climate_columns
gt_target_columns = ['ratio_pine', 'ratio_spruce', 'ratio_deciduous']
132/8:
def test_different_models(data, feature_columns, target_columns = ["vol_pine"]):
    # Test different models with this data (mix different types, eg. soilgrids with metsakeskus and so on)
    
    from xgboost import XGBRegressor
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import mean_squared_error, make_scorer
    from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
    from scipy.stats import uniform, randint
    
    features = data[feature_columns]
    
    targets = data[target_columns]
    # Default XGB
    default_xgb = XGBRegressor(objective='reg:linear', nthread=-1)
    
    # Search best parameters by CV. Note: if all are lists, sampling is done without replacement? which is bad?
    param_distributions_random = {'max_depth': randint(3, 15),
                           'learning_rate': uniform(loc=0.001, scale=0.1),
                           'n_estimators': randint(100, 600),
                           'min_child_weight': [1, 2, 5],
                           'colsample_bytree': uniform(loc=0.5, scale=0.5),
                           'reg_alpha': [0, 0.1, 0.2],
                           'reg_lambda': [0.7, 1],
                           'subsample': [0.8, 0.9],
                           'gamma': [0, 0.07]
                       }
    
    randomsearch = RandomizedSearchCV(XGBRegressor(), scoring=make_scorer(mean_squared_error, greater_is_better=False), 
                                param_distributions=param_distributions_random, 
                                n_jobs=5, cv=5, verbose=True, n_iter=35)

    randomsearch.fit(features, targets)

    best_params = randomsearch.best_params_
    cv_xgb = XGBRegressor(**best_params)
    
    def scorer(estimator, x, y):
        preds = estimator.predict(x)
        error = np.sqrt(mean_squared_error(preds, y))
        return error
    
    def scorer_nrmse(estimator, x, y):
        preds = estimator.predict(x)
        error = (np.sqrt(mean_squared_error(preds, y)) / np.mean(y))*100
        return error
    
    scores = cross_val_score(default_xgb, features, targets, cv=5, scoring=scorer)
    print("RMSE mean of 5-fold CV with default XGB: ", np.mean(scores))
    scores = cross_val_score(cv_xgb, features, targets, cv=5, scoring=scorer_nrmse)
    print("RMSE mean of 5-fold CV with CV optimized XGB: ", np.mean(scores))
    
    return randomsearch

test_different_models(full_data, feature_columns, gt_target_total)
132/9:
feature_columns = copernicus_columns + soilgrids_columns + climate_columns
gt_target_trres = ['ratio_pine', 'ratio_spruce', 'ratio_deciduous']
gt_target_total = ['volume']
# Rescale the target column with the total volume
132/10:
def test_different_models(data, feature_columns, target_columns = ["vol_pine"]):
    # Test different models with this data (mix different types, eg. soilgrids with metsakeskus and so on)
    
    from xgboost import XGBRegressor
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import mean_squared_error, make_scorer
    from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
    from scipy.stats import uniform, randint
    
    features = data[feature_columns]
    
    targets = data[target_columns]
    # Default XGB
    default_xgb = XGBRegressor(objective='reg:linear', nthread=-1)
    
    # Search best parameters by CV. Note: if all are lists, sampling is done without replacement? which is bad?
    param_distributions_random = {'max_depth': randint(3, 15),
                           'learning_rate': uniform(loc=0.001, scale=0.1),
                           'n_estimators': randint(100, 600),
                           'min_child_weight': [1, 2, 5],
                           'colsample_bytree': uniform(loc=0.5, scale=0.5),
                           'reg_alpha': [0, 0.1, 0.2],
                           'reg_lambda': [0.7, 1],
                           'subsample': [0.8, 0.9],
                           'gamma': [0, 0.07]
                       }
    
    randomsearch = RandomizedSearchCV(XGBRegressor(), scoring=make_scorer(mean_squared_error, greater_is_better=False), 
                                param_distributions=param_distributions_random, 
                                n_jobs=5, cv=5, verbose=True, n_iter=35)

    randomsearch.fit(features, targets)

    best_params = randomsearch.best_params_
    cv_xgb = XGBRegressor(**best_params)
    
    def scorer(estimator, x, y):
        preds = estimator.predict(x)
        error = np.sqrt(mean_squared_error(preds, y))
        return error
    
    def scorer_nrmse(estimator, x, y):
        preds = estimator.predict(x)
        error = (np.sqrt(mean_squared_error(preds, y)) / np.mean(y))*100
        return error
    
    scores = cross_val_score(default_xgb, features, targets, cv=5, scoring=scorer)
    print("RMSE mean of 5-fold CV with default XGB: ", np.mean(scores))
    scores = cross_val_score(cv_xgb, features, targets, cv=5, scoring=scorer_nrmse)
    print("RMSE mean of 5-fold CV with CV optimized XGB: ", np.mean(scores))
    
    return randomsearch

test_different_models(full_data, feature_columns, gt_target_total)
132/11:
feature_columns = list(set(copernicus_columns + soilgrids_columns + climate_columns))
gt_target_trres = ['ratio_pine', 'ratio_spruce', 'ratio_deciduous']
gt_target_total = ['volume']
# Rescale the target column with the total volume
132/12:
def test_different_models(data, feature_columns, target_columns = ["vol_pine"]):
    # Test different models with this data (mix different types, eg. soilgrids with metsakeskus and so on)
    
    from xgboost import XGBRegressor
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import mean_squared_error, make_scorer
    from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
    from scipy.stats import uniform, randint
    
    features = data[feature_columns]
    
    targets = data[target_columns]
    # Default XGB
    default_xgb = XGBRegressor(objective='reg:linear', nthread=-1)
    
    # Search best parameters by CV. Note: if all are lists, sampling is done without replacement? which is bad?
    param_distributions_random = {'max_depth': randint(3, 15),
                           'learning_rate': uniform(loc=0.001, scale=0.1),
                           'n_estimators': randint(100, 600),
                           'min_child_weight': [1, 2, 5],
                           'colsample_bytree': uniform(loc=0.5, scale=0.5),
                           'reg_alpha': [0, 0.1, 0.2],
                           'reg_lambda': [0.7, 1],
                           'subsample': [0.8, 0.9],
                           'gamma': [0, 0.07]
                       }
    
    randomsearch = RandomizedSearchCV(XGBRegressor(), scoring=make_scorer(mean_squared_error, greater_is_better=False), 
                                param_distributions=param_distributions_random, 
                                n_jobs=5, cv=5, verbose=True, n_iter=35)

    randomsearch.fit(features, targets)

    best_params = randomsearch.best_params_
    cv_xgb = XGBRegressor(**best_params)
    
    def scorer(estimator, x, y):
        preds = estimator.predict(x)
        error = np.sqrt(mean_squared_error(preds, y))
        return error
    
    def scorer_nrmse(estimator, x, y):
        preds = estimator.predict(x)
        error = (np.sqrt(mean_squared_error(preds, y)) / np.mean(y))*100
        return error
    
    scores = cross_val_score(default_xgb, features, targets, cv=5, scoring=scorer)
    print("RMSE mean of 5-fold CV with default XGB: ", np.mean(scores))
    scores = cross_val_score(cv_xgb, features, targets, cv=5, scoring=scorer_nrmse)
    print("RMSE mean of 5-fold CV with CV optimized XGB: ", np.mean(scores))
    
    return randomsearch

test_different_models(full_data, feature_columns, gt_target_total)
132/13:
def test_different_models(data, feature_columns, target_columns = ["vol_pine"]):
    # Test different models with this data (mix different types, eg. soilgrids with metsakeskus and so on)
    
    from xgboost import XGBRegressor
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import mean_squared_error, make_scorer
    from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
    from scipy.stats import uniform, randint
    
    features = data[feature_columns]
    
    targets = data[target_columns]
    # Default XGB
    default_xgb = XGBRegressor(objective='reg:linear', nthread=-1)
    
    # Search best parameters by CV. Note: if all are lists, sampling is done without replacement? which is bad?
    param_distributions_random = {'max_depth': randint(3, 15),
                           'learning_rate': uniform(loc=0.001, scale=0.1),
                           'n_estimators': randint(100, 600),
                           'min_child_weight': [1, 2, 5],
                           'colsample_bytree': uniform(loc=0.5, scale=0.5),
                           'reg_alpha': [0, 0.1, 0.2],
                           'reg_lambda': [0.7, 1],
                           'subsample': [0.8, 0.9],
                           'gamma': [0, 0.07]
                       }
    
    randomsearch = RandomizedSearchCV(XGBRegressor(), scoring=make_scorer(mean_squared_error, greater_is_better=False), 
                                param_distributions=param_distributions_random, 
                                n_jobs=5, cv=5, verbose=True, n_iter=35)

    randomsearch.fit(features, targets)

    best_params = randomsearch.best_params_
    cv_xgb = XGBRegressor(**best_params)
    
    def scorer(estimator, x, y):
        preds = estimator.predict(x)
        error = np.sqrt(mean_squared_error(preds, y))
        return error
    
    def scorer_nrmse(estimator, x, y):
        preds = estimator.predict(x)
        error = (np.sqrt(mean_squared_error(preds, y)) / np.mean(y))*100
        return error
    
    scores = cross_val_score(default_xgb, features, targets, cv=5, scoring=scorer)
    print("RMSE mean of 5-fold CV with default XGB: ", np.mean(scores))
    scores = cross_val_score(cv_xgb, features, targets, cv=5, scoring=scorer_nrmse)
    print("RMSE mean of 5-fold CV with CV optimized XGB: ", np.mean(scores))
    
    return randomsearch

test_different_models(train_set, feature_columns, gt_target_total)
132/14:
def test_different_models(data, feature_columns, target_columns = ["vol_pine"]):
    # Test different models with this data (mix different types, eg. soilgrids with metsakeskus and so on)
    
    from xgboost import XGBRegressor
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import mean_squared_error, make_scorer
    from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
    from scipy.stats import uniform, randint
    
    features = data[feature_columns]
    
    targets = data[target_columns]
    # Default XGB
    default_xgb = XGBRegressor(objective='reg:linear', nthread=-1)
    
    # Search best parameters by CV. Note: if all are lists, sampling is done without replacement? which is bad?
    param_distributions_random = {'max_depth': randint(3, 15),
                           'learning_rate': uniform(loc=0.001, scale=0.1),
                           'n_estimators': randint(100, 600),
                           'min_child_weight': [1, 2, 5],
                           'colsample_bytree': uniform(loc=0.5, scale=0.5),
                           'reg_alpha': [0, 0.1, 0.2],
                           'reg_lambda': [0.7, 1],
                           'subsample': [0.8, 0.9],
                           'gamma': [0, 0.07]
                       }
    
    randomsearch = RandomizedSearchCV(XGBRegressor(), scoring=make_scorer(mean_squared_error, greater_is_better=False), 
                                param_distributions=param_distributions_random, 
                                n_jobs=5, cv=5, verbose=True, n_iter=35)

    randomsearch.fit(features, targets)

    best_params = randomsearch.best_params_
    cv_xgb = XGBRegressor(**best_params)
    
    def scorer(estimator, x, y):
        preds = estimator.predict(x)
        error = np.sqrt(mean_squared_error(preds, y))
        return error
    
    def scorer_nrmse(estimator, x, y):
        preds = estimator.predict(x)
        error = (np.sqrt(mean_squared_error(preds, y)) / np.mean(y))*100
        return error
    
    scores = cross_val_score(default_xgb, features, targets, cv=5, scoring=scorer)
    print("RMSE mean of 5-fold CV with default XGB: ", np.mean(scores))
    scores = cross_val_score(cv_xgb, features, targets, cv=5, scoring=scorer)
    print("RMSE mean of 5-fold CV with CV optimized XGB: ", np.mean(scores))
    
    return randomsearch

test_different_models(train_set, feature_columns, gt_target_total)
132/15:
def test_different_models(data, feature_columns, target_columns = ["vol_pine"]):
    # Test different models with this data (mix different types, eg. soilgrids with metsakeskus and so on)
    
    from xgboost import XGBRegressor
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import mean_squared_error, make_scorer
    from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
    from scipy.stats import uniform, randint
    
    features = data[feature_columns]
    
    targets = data[target_columns]
    # Default XGB
    default_xgb = XGBRegressor(objective='reg:linear', nthread=-1)
    
    # Search best parameters by CV. Note: if all are lists, sampling is done without replacement? which is bad?
    param_distributions_random = {'max_depth': randint(3, 15),
                           'learning_rate': uniform(loc=0.001, scale=0.1),
                           'n_estimators': randint(100, 600),
                           'min_child_weight': [1, 2, 5],
                           'colsample_bytree': uniform(loc=0.5, scale=0.5),
                           'reg_alpha': [0, 0.1, 0.2],
                           'reg_lambda': [0.7, 1],
                           'subsample': [0.8, 0.9],
                           'gamma': [0, 0.07]
                       }
    
    randomsearch = RandomizedSearchCV(XGBRegressor(), scoring=make_scorer(mean_squared_error, greater_is_better=False), 
                                param_distributions=param_distributions_random, 
                                n_jobs=5, cv=5, verbose=True, n_iter=35)

    randomsearch.fit(features, targets)

    best_params = randomsearch.best_params_
    cv_xgb = XGBRegressor(**best_params)
    
    def scorer(estimator, x, y):
        preds = estimator.predict(x)
        error = np.sqrt(mean_squared_error(preds, y))
        return error
    
    def scorer_nrmse(estimator, x, y):
        preds = estimator.predict(x)
        error = (np.sqrt(mean_squared_error(preds, y)) / np.mean(y))*100
        return error
    
    scores = cross_val_score(default_xgb, features, targets, cv=5, scoring=scorer)
    print("RMSE mean of 5-fold CV with default XGB: ", np.mean(scores))
    scores = cross_val_score(cv_xgb, features, targets, cv=5, scoring=scorer)
    print("RMSE mean of 5-fold CV with CV optimized XGB: ", np.mean(scores))
    
    return default_xgb, cv_xgb

default_xgb, optimized_xgb = test_different_models(train_set, feature_columns, gt_target_total)
132/16:
%load_ext autoreload
%autoreload 2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import sys
import requests
import seaborn as sns
pd.options.display.float_format = '{:,.2f}'.format
# Add path to where some utilities are so they can be imported
sys.path.insert(0, r'../regressors')
from data.data_loading import import_data, GeoAPI, split_from_ids
from data import data_loading

from notebook.services.config import ConfigManager
c = ConfigManager()
c.update('notebook', {"CodeCell": {"cm_config": {"autoCloseBrackets": False}}})
132/17:
def get_metrics(preds, targets)
    from sklearn.metrics import mean_squared_error
    rmse = np.sqrt(mean_squared_error(terra_volumes, slu_volumes, multioutput='raw_values'))
    
print("Terramonitor RMSE with total volume on test set: ")
print(get_metrics(test_set["se_volumes_m3_ha"], test_set["volume"]))
print("Our prediction RMSE with total volume on test set: ")
our_preds = cv_xgb.predict(test_set[feature_columns])
print(get_metrics(our_preds, test_set["volume"]))
132/18:
def get_metrics(preds, targets):
    from sklearn.metrics import mean_squared_error
    rmse = np.sqrt(mean_squared_error(terra_volumes, slu_volumes, multioutput='raw_values'))
    
print("Terramonitor RMSE with total volume on test set: ")
print(get_metrics(test_set["se_volumes_m3_ha"], test_set["volume"]))
print("Our prediction RMSE with total volume on test set: ")
our_preds = cv_xgb.predict(test_set[feature_columns])
print(get_metrics(our_preds, test_set["volume"]))
132/19:
def get_metrics(preds, targets):
    from sklearn.metrics import mean_squared_error
    rmse = np.sqrt(mean_squared_error(preds, targets, multioutput='raw_values'))
    
print("Terramonitor RMSE with total volume on test set: ")
print(get_metrics(test_set["se_volumes_m3_ha"], test_set["volume"]))
print("Our prediction RMSE with total volume on test set: ")
our_preds = cv_xgb.predict(test_set[feature_columns])
print(get_metrics(our_preds, test_set["volume"]))
132/20:
def get_metrics(preds, targets):
    from sklearn.metrics import mean_squared_error
    rmse = np.sqrt(mean_squared_error(preds, targets, multioutput='raw_values'))
    
print("Terramonitor RMSE with total volume on test set: ")
print(get_metrics(test_set["se_volumes_m3_ha"], test_set["volume"]))
print("Our prediction RMSE with total volume on test set: ")
our_preds = optimized.predict(test_set[feature_columns])
print(get_metrics(our_preds, test_set["volume"]))
132/21:
def get_metrics(preds, targets):
    from sklearn.metrics import mean_squared_error
    rmse = np.sqrt(mean_squared_error(preds, targets, multioutput='raw_values'))
    
print("Terramonitor RMSE with total volume on test set: ")
print(get_metrics(test_set["se_volumes_m3_ha"], test_set["volume"]))
print("Our prediction RMSE with total volume on test set: ")
our_preds = optimized_xgb.predict(test_set[feature_columns])
print(get_metrics(our_preds, test_set["volume"]))
132/22:
def get_metrics(preds, targets):
    from sklearn.metrics import mean_squared_error
    rmse = np.sqrt(mean_squared_error(preds, targets, multioutput='raw_values'))
    
print("Terramonitor RMSE with total volume on test set: ")
print(get_metrics(test_set["se_volumes_m3_ha"], test_set["volume"]))
print("Our prediction RMSE with total volume on test set: ")
optimized_xgb.fit(train_set[feature_columns])
our_preds = optimized_xgb.predict(test_set[feature_columns])
print(get_metrics(our_preds, test_set["volume"]))
132/23:
def get_metrics(preds, targets):
    from sklearn.metrics import mean_squared_error
    rmse = np.sqrt(mean_squared_error(preds, targets, multioutput='raw_values'))
    
print("Terramonitor RMSE with total volume on test set: ")
print(get_metrics(test_set["se_volumes_m3_ha"], test_set["volume"]))
print("Our prediction RMSE with total volume on test set: ")
optimized_xgb.fit(train_set[feature_columns], train_set["volume"])
our_preds = optimized_xgb.predict(test_set[feature_columns])
print(get_metrics(our_preds, test_set["volume"]))
132/24: test_set["se_volumes_m3_ha"]
132/25: test_set["volume"]
132/26: test_set["se_volumes_m3_ha"]
132/27: test_set["se_volumes_m3_ha"].isna().sum()
132/28: test_set["volume"].isna().sum()
132/29:
def get_metrics(preds, targets):
    from sklearn.metrics import mean_squared_error
    rmse = np.sqrt(mean_squared_error(preds, targets, multioutput='raw_values'))
    return rmse
print("Terramonitor RMSE with total volume on test set: ")
print(get_metrics(test_set["se_volumes_m3_ha"], test_set["volume"]))
print("Our prediction RMSE with total volume on test set: ")
optimized_xgb.fit(train_set[feature_columns], train_set["volume"])
our_preds = optimized_xgb.predict(test_set[feature_columns])
print(get_metrics(our_preds, test_set["volume"]))
167/1: from data import data_loading
167/2: data_loading.create_test_set("/home/tman/Work/data/harvest_FI/v_stand_level_features", "/home/tman/Work/data/harvest_FI/")
167/3: data_loading.create_test_set("/home/tman/Work/data/harvest_FI/v_stand_level_features.csv", "/home/tman/Work/data/harvest_FI/")
169/1:
%load_ext autoreload
%autoreload 2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import json
import os
from tqdm import tqdm
import re
import sys
from sklearn.model_selection import train_test_split
import requests
pd.options.display.float_format = '{:,.2f}'.format
# Add path to where utils.py is so metrics can be imported
sys.path.insert(0, r'../regressors')
from data.data_loading import import_data, GeoAPI, split_from_ids
from data import data_loading
169/2:
%load_ext autoreload
%autoreload 2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import json
import os
from tqdm import tqdm
import re
import sys
from sklearn.model_selection import train_test_split
import requests
pd.options.display.float_format = '{:,.2f}'.format
# Add path to where utils.py is so metrics can be imported
sys.path.insert(0, r'../../regressors')
from data.data_loading import import_data, GeoAPI, split_from_ids
from data import data_loading
169/3:
def get_metsakeskus_data():
    columns_list = [["volumepine","volumespruce","volumedeciduous","volume","creationtime", "updatetime",
                     "soiltype","fertilityclass","laserheight","laserdensity"]]

    schema_list = ['metsakeskus_hila']
    tables_list = ['gridcell']

    data = api.request_data(schema_list, tables_list, columns_list, batch_size=1000)

    # Return plot_ids from index to a column.
    data.reset_index(inplace=True)
    data = data.drop_duplicates(subset='plot_id')
    
    return data
169/4:
stand_data = pd.read_csv("../../../../data/koskisen/v_stand_level_features.csv")
gridcell_data = pd.read_csv("../../../../data/koskisen/v_gridcell_volumes_with_coords.csv")
gridcell_data = gridcell_data.drop('hila_polygon', axis=1)
169/5:
stand_data = pd.read_csv("/home/tman/Work/data/koskisen/v_stand_level_features.csv")
gridcell_data = pd.read_csv("/home/tman/Work/data/koskisen/v_gridcell_volumes_with_coords.csv")
gridcell_data = gridcell_data.drop('hila_polygon', axis=1)
169/6: gridcell_data[:2
169/7: gridcell_data[:2]
169/8:
api = GeoAPI(default_locations=gridcell_data[['easting', 'northing']].values.tolist(),
                default_srid=3067, default_plot_ids=gridcell_data.hila_gridcellid.values.tolist())

def get_metsakeskus_data():
    columns_list = [["volumepine","volumespruce","volumedeciduous","volume","creationtime", "updatetime",
                     "soiltype","fertilityclass","laserheight","laserdensity"]]

    schema_list = ['metsakeskus_hila']
    tables_list = ['gridcell']

    data = api.request_data(schema_list, tables_list, columns_list, batch_size=1000)

    # Return plot_ids from index to a column.
    data.reset_index(inplace=True)
    data = data.drop_duplicates(subset='plot_id')
    
    return data
169/9:
api = GeoAPI(default_locations=gridcell_data[['easting', 'northing']].values.tolist(),
                default_srid=3067, default_plot_ids=gridcell_data.hila_gridcellid.values.tolist())

def get_metsakeskus_data():
    columns_list = [["volumepine","volumespruce","volumedeciduous","volume","creationtime", "updatetime",
                     "soiltype","fertilityclass","laserheight","laserdensity"]]

    schema_list = ['metsakeskus_hila']
    tables_list = ['gridcell']

    data = api.request_data(schema_list, tables_list, columns_list, batch_size=1000)

    # Return plot_ids from index to a column.
    data.reset_index(inplace=True)
    data = data.drop_duplicates(subset='plot_id')
    
    return data

metsakeskus_data = get_metsakeskus_data()
169/10:
api = GeoAPI(default_locations=gridcell_data[['easting', 'northing']].values.tolist(),
                default_srid=3067, default_plot_ids=gridcell_data.hila_gridcellid.values.tolist())

def get_metsakeskus_data():
    columns_list = [["volumepine","volumespruce","volumedeciduous","volume","creationtime", "updatetime",
                     "soiltype","fertilityclass","laserheight","laserdensity"]]

    schema_list = ['metsakeskus_hila']
    tables_list = ['gridcell']

    data = api.request_data(schema_list, tables_list, columns_list, batch_size=2000)

    # Return plot_ids from index to a column.
    data.reset_index(inplace=True)
    data = data.drop_duplicates(subset='plot_id')
    
    return data

metsakeskus_data = get_metsakeskus_data()
170/1:
%load_ext autoreload
%autoreload 2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import sys
import requests
import seaborn as sns
pd.options.display.float_format = '{:,.2f}'.format
# Add path to where some utilities are so they can be imported
sys.path.insert(0, r'../regressors')
from data.data_loading import import_data, GeoAPI, split_from_ids
from data import data_loading
170/2:
%load_ext autoreload
%autoreload 2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import sys
import requests
import seaborn as sns
pd.options.display.float_format = '{:,.2f}'.format
# Add path to where some utilities are so they can be imported
sys.path.insert(0, r'../../regressors')
from data.data_loading import import_data, GeoAPI, split_from_ids
from data import data_loading
170/3:
# load SLU data

slu_plots_since_2015 = pd.read_csv("../../../data/terramonitor_verification/ccgeodb_se_slu_v_slu_plots_since_2015_terramonitor.csv")
slu_plots_with_distance = pd.read_csv("../../../data/terramonitor_verification/ccgeodb_se_slu_v_slu_plots_since_2015_with_distance.csv")
170/4:
# load SLU data

slu_plots_since_2015 = pd.read_csv("../../../../data/terramonitor_verification/ccgeodb_se_slu_v_slu_plots_since_2015_terramonitor.csv")
slu_plots_with_distance = pd.read_csv("../../../../data/terramonitor_verification/ccgeodb_se_slu_v_slu_plots_since_2015_with_distance.csv")
170/5:
print(slu_plots_since_2015.shape)
print(slu_plots_with_distance.shape)
170/6:
#TODO: get all values for all rows, filter to test set later.
api = GeoAPI(default_locations=slu_plots_with_distance[['lon', 'lat']].values.tolist(),
                default_srid=4326, default_plot_ids=slu_plots_with_distance.plot_id.values.tolist())

def get_terramonitor_predictions():
    tables_list = ["se_volumes_m3_ha", "se_pine_percent", "se_spruce_percent", "se_deciduous_percent"]
    columns_list = [None]*len(tables_list)
    schema_list = ['terramonitor']*len(tables_list)
    
    data = api.request_data(schema_list, tables_list, columns_list, batch_size=1000)
    data = data.reset_index()
    data = data.drop_duplicates(subset='plot_id')
        
    return data

def get_copernicus_data():
    tables_list = ["copernicus_dem", "copernicus_slope", "copernicus_aspect", "copernicus_leaf_type", "copernicus_tree_cover"]
    columns_list = [None]*len(tables_list)
    schema_list = ['physical']*len(tables_list)
    
    data = api.request_data(schema_list, tables_list, columns_list, batch_size=1000)
    data = data.reset_index()
    data = data.drop_duplicates(subset='plot_id')
        
    return data

def get_soilgrids_data():
    data = api.request_data(data_groups=['soilgrids'])
    data = data.reset_index()
    data = data.drop_duplicates(subset='plot_id')
        
    return data

def get_climate_data():
    data = api.request_data(data_groups=['climate_data'])
    data = data.reset_index()
    data = data.drop_duplicates(subset='plot_id')
        
    return data

def get_lidar_data():
    tables_list = ["lidar_vol_m3_ha", "lidar_height_dm", "lidar_diameter_cm"]
    columns_list = [None]*len(tables_list)
    schema_list = ['sweden']*len(tables_list)
    
    data = api.request_data(schema_list, tables_list, columns_list, batch_size=1000)
    data = data.reset_index()
    data = data.drop_duplicates(subset='plot_id')
        
    return data

def get_mineral_data():
    tables_list = ['se_mineral_soil']
    columns_list = [['be', 'cd', 'dy', 'er', 'eu', 'lu', 'mo', 'nb', 'sn', 'tb', 'te', 'tl', 'tm']]
    schema_list = ['physical']*len(tables_list)
    
    data = api.request_data(schema_list, tables_list, columns_list, batch_size=1000)
    data = data.reset_index()
    data = data.drop_duplicates(subset='plot_id')
        
    return data
copernicus_data = get_copernicus_data()
soilgrids_data = get_soilgrids_data()
climate_data = get_climate_data()
lidar_data = get_lidar_data()
mineral_data = get_mineral_data()

terramonitor_predictions = get_terramonitor_predictions()
169/11: metsakeskus_data[:2
169/12: metsakeskus_data[:2]
170/7:
# Merge all data into one dataframe
copernicus_columns = list(copernicus_data.columns)
soilgrids_columns = list(soilgrids_data.columns)
climate_columns = list(climate_data.columns)
lidar_columns = list(lidar_data.columns)
mineral_columns = list(mineral_data.columns)

full_data = slu_plots_with_distance.merge(copernicus_data, on="plot_id").\
merge(soilgrids_data, on="plot_id").\
merge(climate_data, on="plot_id").\
merge(lidar_data, on="plot_id").\
merge(mineral_data, on="plot_id").\
merge(terramonitor_predictions, on="plot_id")
# full_data.to_csv(r"C:\Users\Teemu\Work\data\harvester_SE\terramonitor_data_ting.csv")
170/8:
# Set removes duplicate column names such as plot_id
feature_columns = list(set(copernicus_columns + soilgrids_columns + climate_columns + lidar_columns + mineral_columns))

# Rescale the target column with the total volume
gt_target_trees = ['ratio_pine', 'ratio_spruce', 'ratio_deciduous']
scaled_volumes = ['pine_volume', 'spruce_volume', 'deciduous_volume']
gt_target_total = ['volume']

trees_terra = ['se_pine_percent', 'se_spruce_percent', 'se_deciduous_percent']
terra_total = ['se_volumes_m3_ha']
terra_scaled = ['terra_pine', 'terra_spruce', 'terra_deciduous']

# volumes are NaN when the total volume is 0 (eg. other volumes are also 0), so it's ok to fill with na
full_data[terra_scaled] = (full_data[terra_total].values * (full_data[trees_terra] / 100)).fillna(0)
full_data[scaled_volumes] = (full_data[gt_target_total].values * full_data[gt_target_trees]).fillna(0)
170/9:
# Filter data to train and test:
train_set = full_data[(full_data['distance_km_from_kastet'] > 100) & (full_data['distance_km_from_kastet'] < 300)]
test_set = full_data[full_data['distance_km_from_kastet'] < 100]
print("Training set: plots within 300km but outside 100km of Kastet. Number of plots in training: %d" % len(train_set))
print("Testing set: plots within 100km of Kastet. Number of plots in test: %d" % len(test_set))

train_set.to_csv("/home/tman/data/SEsampletiles/terramonitor_train.csv", index=False)
test_set.to_csv("/home/tman/data/SEsampletiles/terramonitor_test.csv", index=False)
170/10:
# Filter data to train and test:
train_set = full_data[(full_data['distance_km_from_kastet'] > 100) & (full_data['distance_km_from_kastet'] < 300)]
test_set = full_data[full_data['distance_km_from_kastet'] < 100]
print("Training set: plots within 300km but outside 100km of Kastet. Number of plots in training: %d" % len(train_set))
print("Testing set: plots within 100km of Kastet. Number of plots in test: %d" % len(test_set))

train_set.to_csv("/home/tman/Work/data/SEsampletiles/terramonitor_train.csv", index=False)
test_set.to_csv("/home/tman/Work/data/SEsampletiles/terramonitor_test.csv", index=False)
170/11:
dummy_means = np.mean(train_set[targets], axis=0)
dummy_means.shape
170/12:
# Set removes duplicate column names such as plot_id
feature_columns = list(set(copernicus_columns + soilgrids_columns + climate_columns + lidar_columns + mineral_columns))

# Rescale the target column with the total volume
gt_target_trees = ['ratio_pine', 'ratio_spruce', 'ratio_deciduous']
scaled_volumes = ['pine_volume', 'spruce_volume', 'deciduous_volume']
gt_target_total = ['volume']

trees_terra = ['se_pine_percent', 'se_spruce_percent', 'se_deciduous_percent']
terra_total = ['se_volumes_m3_ha']
terra_scaled = ['terra_pine', 'terra_spruce', 'terra_deciduous']

all_columns = gt_target_total + scaled_volumes

# volumes are NaN when the total volume is 0 (eg. other volumes are also 0), so it's ok to fill with na
full_data[terra_scaled] = (full_data[terra_total].values * (full_data[trees_terra] / 100)).fillna(0)
full_data[scaled_volumes] = (full_data[gt_target_total].values * full_data[gt_target_trees]).fillna(0)
170/13:
# Filter data to train and test:
train_set = full_data[(full_data['distance_km_from_kastet'] > 100) & (full_data['distance_km_from_kastet'] < 300)]
test_set = full_data[full_data['distance_km_from_kastet'] < 100]
print("Training set: plots within 300km but outside 100km of Kastet. Number of plots in training: %d" % len(train_set))
print("Testing set: plots within 100km of Kastet. Number of plots in test: %d" % len(test_set))

train_set.to_csv("/home/tman/Work/data/SEsampletiles/terramonitor_train.csv", index=False)
test_set.to_csv("/home/tman/Work/data/SEsampletiles/terramonitor_test.csv", index=False)
170/14:
dummy_means = np.mean(train_set[targets], axis=0)
dummy_means.shape
170/15:
# Set removes duplicate column names such as plot_id
feature_columns = list(set(copernicus_columns + soilgrids_columns + climate_columns + lidar_columns + mineral_columns))

# Rescale the target column with the total volume
gt_target_trees = ['ratio_pine', 'ratio_spruce', 'ratio_deciduous']
scaled_volumes = ['pine_volume', 'spruce_volume', 'deciduous_volume']
gt_target_total = ['volume']

trees_terra = ['se_pine_percent', 'se_spruce_percent', 'se_deciduous_percent']
terra_total = ['se_volumes_m3_ha']
terra_scaled = ['terra_pine', 'terra_spruce', 'terra_deciduous']

all_columns = gt_target_total + scaled_volumes
targets = all_columns

# volumes are NaN when the total volume is 0 (eg. other volumes are also 0), so it's ok to fill with na
full_data[terra_scaled] = (full_data[terra_total].values * (full_data[trees_terra] / 100)).fillna(0)
full_data[scaled_volumes] = (full_data[gt_target_total].values * full_data[gt_target_trees]).fillna(0)
170/16:
# Filter data to train and test:
train_set = full_data[(full_data['distance_km_from_kastet'] > 100) & (full_data['distance_km_from_kastet'] < 300)]
test_set = full_data[full_data['distance_km_from_kastet'] < 100]
print("Training set: plots within 300km but outside 100km of Kastet. Number of plots in training: %d" % len(train_set))
print("Testing set: plots within 100km of Kastet. Number of plots in test: %d" % len(test_set))

train_set.to_csv("/home/tman/Work/data/SEsampletiles/terramonitor_train.csv", index=False)
test_set.to_csv("/home/tman/Work/data/SEsampletiles/terramonitor_test.csv", index=False)
170/17:
dummy_means = np.mean(train_set[targets], axis=0)
dummy_means.shape
170/18:
dummy_means = np.mean(train_set[targets], axis=0)
dummy_means
170/19:
dummy_means = np.mean(train_set[targets], axis=0)
print("Dummy mean of training set used as prediction: ")
print(get_metrics(dummy_means, test_set[targets])
170/20:
dummy_means = np.mean(train_set[targets], axis=0)
print("Dummy mean of training set used as prediction: ")
print(get_metrics(dummy_means, test_set[targets]))
170/21:
def get_metrics(preds, targets):
    from sklearn.metrics import mean_squared_error
    rmse = np.sqrt(mean_squared_error(preds, targets, multioutput='raw_values'))
    return rmse

terra_targets = terra_total + terra_scaled
targets = all_columns
print("Terramonitor RMSE with volumes on test set (total, pine, spruce, deciduous): ")
print(get_metrics(test_set[terra_targets], test_set[targets]))
print("Our prediction RMSE with volumes on test set (total, pine, spruce, deciduous): ")
our_preds = [opt_model.predict(test_set[feature_columns]) for opt_model in opt_models]
print(get_metrics(np.array(our_preds).T, test_set[targets]))
dummy_means = np.mean(train_set[targets], axis=0)
print("Dummy mean of training set used as prediction: ")
print(get_metrics(dummy_means, test_set[
170/22:
dummy_means = np.mean(train_set[targets], axis=0)
print("Dummy mean of training set used as prediction: ")
print(get_metrics(dummy_means, test_set[targets]))
170/23:
def get_metrics(preds, targets):
    from sklearn.metrics import mean_squared_error
    rmse = np.sqrt(mean_squared_error(preds, targets, multioutput='raw_values'))
    return rmse

terra_targets = terra_total + terra_scaled
targets = all_columns
print("Terramonitor RMSE with volumes on test set (total, pine, spruce, deciduous): ")
print(get_metrics(test_set[terra_targets], test_set[targets]))
print("Our prediction RMSE with volumes on test set (total, pine, spruce, deciduous): ")
our_preds = [opt_model.predict(test_set[feature_columns]) for opt_model in opt_models]
print(get_metrics(np.array(our_preds).T, test_set[targets]))
dummy_means = np.mean(train_set[targets], axis=0)
print("Dummy mean of training set used as prediction: ")
#print(get_metrics(dummy_means, test_set[
170/24:
dummy_means = np.mean(train_set[targets], axis=0)
print("Dummy mean of training set used as prediction: ")
print(get_metrics(dummy_means, test_set[targets]))
170/25: test_set[targets]
170/26:
dummy_means = np.mean(train_set[targets], axis=0)
np.replicate(dummy_means, test_set[targets].shape[0], axis=0)
#print("Dummy mean of training set used as prediction: ")
#print(get_metrics(dummy_means, test_set[targets]))
170/27:
dummy_means = np.mean(train_set[targets], axis=0)
np.repeat(dummy_means, test_set[targets].shape[0], axis=0)
#print("Dummy mean of training set used as prediction: ")
#print(get_metrics(dummy_means, test_set[targets]))
170/28:
dummy_means = np.mean(train_set[targets], axis=0)
np.repeat(dummy_means, test_set[targets].shape[0])
#print("Dummy mean of training set used as prediction: ")
#print(get_metrics(dummy_means, test_set[targets]))
170/29:
dummy_means = np.mean(train_set[targets], axis=0)
np.repeat(dummy_means, test_set[targets].shape[0]).shape
#print("Dummy mean of training set used as prediction: ")
#print(get_metrics(dummy_means, test_set[targets]))
170/30:
dummy_means = np.mean(train_set[targets], axis=0).values
np.repeat(dummy_means, test_set[targets].shape[0]), axis=0).shape
#print("Dummy mean of training set used as prediction: ")
#print(get_metrics(dummy_means, test_set[targets]))
170/31:
dummy_means = np.mean(train_set[targets], axis=0).values
np.repeat(dummy_means, test_set[targets].shape[0], axis=0).shape
#print("Dummy mean of training set used as prediction: ")
#print(get_metrics(dummy_means, test_set[targets]))
170/32:
dummy_means = np.mean(train_set[targets], axis=0).values
np.repeat(dummy_means, test_set[targets].shape[0], axis=1).shape
#print("Dummy mean of training set used as prediction: ")
#print(get_metrics(dummy_means, test_set[targets]))
170/33:
dummy_means = np.mean(train_set[targets], axis=0).values
np.tile(dummy_means, test_set[targets].shape[0])
#print("Dummy mean of training set used as prediction: ")
#print(get_metrics(dummy_means, test_set[targets]))
170/34:
dummy_means = np.mean(train_set[targets], axis=0).values
np.tile(dummy_means, test_set[targets].shape[0]).shape
#print("Dummy mean of training set used as prediction: ")
#print(get_metrics(dummy_means, test_set[targets]))
170/35:
dummy_means = np.mean(train_set[targets], axis=0).values
np.tile(dummy_means, test_set[targets].shape[0], axis=0).shape
#print("Dummy mean of training set used as prediction: ")
#print(get_metrics(dummy_means, test_set[targets]))
170/36: dummy_means
170/37: dummy_means.shape
170/38: np.expand_dims(dummy_means, 1)
170/39: np.expand_dims(dummy_means, 1).shape
170/40: np.expand_dims(dummy_means, 0).shape
170/41: np.tile(np.expand_dims(dummy_means, 0), test_set[targets].shape[0])
170/42: np.tile(np.expand_dims(dummy_means, 0), test_set[targets].shape[0]).shape
170/43: np.repeat(np.expand_dims(dummy_means, 0), test_set[targets].shape[0], axis=0).shape
170/44:
dummy_means_temp = np.mean(train_set[targets], axis=0).values
dummy_means = np.repeat(np.expand_dims(dummy_means_temp, 0), test_set[targets].shape[0], axis=0)
print("Dummy mean of training set used as prediction: ")
#print(get_metrics(dummy_means, test_set[targets]))
170/45:
dummy_means_temp = np.mean(train_set[targets], axis=0).values
dummy_means = np.repeat(np.expand_dims(dummy_means_temp, 0), test_set[targets].shape[0], axis=0)
print("Dummy mean of training set used as prediction: ")
print(get_metrics(dummy_means, test_set[targets]))
173/1:
%load_ext autoreload
%autoreload 2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import json
import os
from tqdm import tqdm
import re
import sys
from sklearn.model_selection import train_test_split
import requests
pd.options.display.float_format = '{:,.2f}'.format
# Add path to where utils.py is so metrics can be imported
sys.path.insert(0, r'../../regressors')
from data.data_loading import import_data, GeoAPI, split_from_ids
from data import data_loading
173/2:
stand_data = pd.read_csv("~/Work/data/koskisen/v_stand_level_features.csv")
gridcell_data = pd.read_csv("~/Work/data/koskisen/v_gridcell_volumes_with_coords.csv")
gridcell_data = gridcell_data.drop('hila_polygon', axis=1)
173/3:
columns_from_stand = ['prd_id', 'harvest_year', 'harvest_start']
koskisen_grids = gridcell_data.merge(stand_data[columns_from_stand], left_on="koski_prd_id", right_on="prd_id")
koskisen_grids['harvest_start'] = pd.to_datetime(koskisen_grids['harvest_start'])
173/4:
stand_data = pd.read_csv("~/Work/data/koskisen/v_stand_level_features.csv")
gridcell_data = pd.read_csv("~/Work/data/koskisen/v_gridcell_volumes_with_coords.csv")
gridcell_data = gridcell_data.drop('hila_polygon', axis=1)
173/5:
columns_from_stand = ['prd_id', 'harvest_year', 'harvest_start']
koskisen_grids = gridcell_data.merge(stand_data[columns_from_stand], left_on="koski_prd_id", right_on="prd_id")
koskisen_grids['harvest_start'] = pd.to_datetime(koskisen_grids['harvest_start'])
173/6:
api = GeoAPI(default_locations=gridcell_data[['easting', 'northing']].values.tolist(),
                default_srid=3067, default_plot_ids=gridcell_data.hila_gridcellid.values.tolist())

def get_metsakeskus_data():
    columns_list = [["volumepine","volumespruce","volumedeciduous","volume","creationtime", "updatetime",
                     "soiltype","fertilityclass","laserheight","laserdensity"]]

    schema_list = ['metsakeskus_hila']
    tables_list = ['gridcell']

    data = api.request_data(schema_list, tables_list, columns_list, batch_size=2000)

    # Return plot_ids from index to a column.
    data.reset_index(inplace=True)
    data = data.drop_duplicates(subset='plot_id')
    
    return data

metsakeskus_data = get_metsakeskus_data()
173/7:
api = GeoAPI(default_locations=gridcell_data[['easting', 'northing']].values.tolist(),
                default_srid=3067, default_plot_ids=gridcell_data.hila_gridcellid.values.tolist())

def get_metsakeskus_data():
    columns_list = [["volumepine","volumespruce","volumedeciduous","volume","creationtime", "updatetime",
                     "soiltype","fertilityclass","laserheight","laserdensity"]]

    schema_list = ['metsakeskus_hila']
    tables_list = ['gridcell']

    data = api.request_data(schema_list, tables_list, columns_list, batch_size=2000)

    # Return plot_ids from index to a column.
    data.reset_index(inplace=True)
    data = data.drop_duplicates(subset='plot_id')
    
    return data

metsakeskus_data = get_metsakeskus_data()
173/8:
metsakeskus_data['creationtime'] = pd.to_datetime(metsakeskus_data['creationtime'])
metsakeskus_data['updatetime'] = pd.to_datetime(metsakeskus_data['updatetime'])
173/9:
# GeoAPI adds plot_id to corresponding rows when fetching data. We used hila_gridcellid when fetching data
full_data = koskisen_grids.merge(metsakeskus_data, left_on="hila_gridcellid", right_on="plot_id")
173/10: full_data.to_csv("~/Work/data/koskisen/fulldata_metsakeskus_koskisen.csv")
173/11:
# Remember same volume order in both
metsakeskus_pred_columns = ['volume', 'volumepine', 'volumespruce', 'volumedeciduous']
koskisen_vol_columns = ['total_volume_ha', 'pine_volume_ha', 'spruce_volume_ha', 'deciduous_volume_ha']
time_columns = ['updatetime', 'harvest_start']

# aggregate volume means and times. Take minimum of both times so we can compare and get just the stands where preds were made
# before harvest
stat_dict = {col: "mean" for col in (metsakeskus_pred_columns + koskisen_vol_columns)}
for col in time_columns: stat_dict[col] = 'min'
# get the means of the volumes per stand and minimum of each harvest_start and updatetime per stand.
# OK to take the mean of koskisen gridcells as ground truth as they're all the same anyway
volume_means_times = full_data.groupby("koski_prd_id")[metsakeskus_pred_columns + koskisen_vol_columns + time_columns].agg(stat_dict)

def calculate_metsakeskus_error(df):
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    
    metsakeskus_preds = df[metsakeskus_pred_columns]
    koskisen_vols = df[koskisen_vol_columns]
    rmse = mean_squared_error(metsakeskus_preds, koskisen_vols, multioutput='raw_values')
    print(np.sqrt(rmse))
    

print("Metsakeskus RMSE on all stands (total, pine, spruce, deciduous):")
calculate_metsakeskus_error(volume_means_times)
print("\nMetsakeskus RMSE on stands where all gridcell preds were made before harvest:")
calculate_metsakeskus_error(volume_means_times[volume_means_times['updatetime'] < volume_means_times['harvest_start']])
173/12: volume_means_times[volume_means_times['updatetime'] < volume_means_times['harvest_start']]
173/13: volume_means_times[volume_means_times['updatetime'] < volume_means_times['harvest_start']].koski_prd_id
173/14: volume_means_times[volume_means_times['updatetime'] < volume_means_times['harvest_start']].index
173/15: volume_means_times.shape
173/16: stand_data.shape
173/17: stand_data
173/18: stand_data.prd_id.unique()
173/19: len(stand_data.prd_id.unique())
173/20: gridcell_data
173/21: len(gridcell_data.koski_prd_id.unique())
173/22:
preds_before = volume_means_times[volume_means_times['updatetime'] < volume_means_times['harvest_start']]
preds_before.index.to_csv("~/Work/data/koskisen/testids.csv")
173/23: preds_before.index.to_df()
173/24: preds_before.index.to_frame()
173/25:
preds_before = volume_means_times[volume_means_times['updatetime'] < volume_means_times['harvest_start']]
preds_before.index.to_frame().rename("koski_prd_id", "prd_id").to_csv("~/Work/data/koskisen/testids.csv", index=False)
173/26: preds_before.index.to_frame().rename(columns={'koski_prd_id':'prd_id'}
173/27: preds_before.index.to_frame().rename(columns={'koski_prd_id':'prd_id'})
173/28: preds_before.index.to_frame().rename(columns={'koski_prd_id':'prd_id'}).to_csv("~/Work/data/koskisen/testids.csv", index=False)
173/29: gridcell_data.shape
173/30: len(gridcell_data.hila_gridcellid.unique())
173/31:
# Remember same volume order in both
metsakeskus_pred_columns = ['volume', 'volumepine', 'volumespruce', 'volumedeciduous']
koskisen_vol_columns = ['total_volume_ha', 'pine_volume_ha', 'spruce_volume_ha', 'deciduous_volume_ha']
time_columns = ['updatetime', 'harvest_start']

# aggregate volume means and times. Take minimum of both times so we can compare and get just the stands where preds were made
# before harvest
stat_dict = {col: "mean" for col in (metsakeskus_pred_columns + koskisen_vol_columns)}
for col in time_columns: stat_dict[col] = 'min'
# get the means of the volumes per stand and minimum of each harvest_start and updatetime per stand.
# OK to take the mean of koskisen gridcells as ground truth as they're all the same anyway
volume_means_times = full_data.groupby("koski_prd_id")[metsakeskus_pred_columns + koskisen_vol_columns + time_columns].agg(stat_dict)

def calculate_metsakeskus_error(df):
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    
    metsakeskus_preds = df[metsakeskus_pred_columns]
    koskisen_vols = df[koskisen_vol_columns]
    koskisen_means = np.mean(koskisen_vols, axis=0)
    rmse = mean_squared_error(metsakeskus_preds, koskisen_vols, multioutput='raw_values')
    print(np.sqrt(rmse))
    

print("Metsakeskus RMSE on all stands (total, pine, spruce, deciduous):")
calculate_metsakeskus_error(volume_means_times)
print("\nMetsakeskus RMSE on stands where all gridcell preds were made before harvest:")
calculate_metsakeskus_error(volume_means_times[volume_means_times['updatetime'] < volume_means_times['harvest_start']])
173/32:
# Remember same volume order in both
metsakeskus_pred_columns = ['volume', 'volumepine', 'volumespruce', 'volumedeciduous']
koskisen_vol_columns = ['total_volume_ha', 'pine_volume_ha', 'spruce_volume_ha', 'deciduous_volume_ha']
time_columns = ['updatetime', 'harvest_start']

# aggregate volume means and times. Take minimum of both times so we can compare and get just the stands where preds were made
# before harvest
stat_dict = {col: "mean" for col in (metsakeskus_pred_columns + koskisen_vol_columns)}
for col in time_columns: stat_dict[col] = 'min'
# get the means of the volumes per stand and minimum of each harvest_start and updatetime per stand.
# OK to take the mean of koskisen gridcells as ground truth as they're all the same anyway
volume_means_times = full_data.groupby("koski_prd_id")[metsakeskus_pred_columns + koskisen_vol_columns + time_columns].agg(stat_dict)

def calculate_metsakeskus_error(df):
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    
    metsakeskus_preds = df[metsakeskus_pred_columns]
    koskisen_vols = df[koskisen_vol_columns]
    koskisen_means = np.mean(koskisen_vols, axis=0)
    print(koskisen_means)
    rmse = mean_squared_error(metsakeskus_preds, koskisen_vols, multioutput='raw_values')
    print(np.sqrt(rmse))
    

print("Metsakeskus RMSE on all stands (total, pine, spruce, deciduous):")
calculate_metsakeskus_error(volume_means_times)
print("\nMetsakeskus RMSE on stands where all gridcell preds were made before harvest:")
calculate_metsakeskus_error(volume_means_times[volume_means_times['updatetime'] < volume_means_times['harvest_start']])
173/33:
# Remember same volume order in both
metsakeskus_pred_columns = ['volume', 'volumepine', 'volumespruce', 'volumedeciduous']
koskisen_vol_columns = ['total_volume_ha', 'pine_volume_ha', 'spruce_volume_ha', 'deciduous_volume_ha']
time_columns = ['updatetime', 'harvest_start']

# aggregate volume means and times. Take minimum of both times so we can compare and get just the stands where preds were made
# before harvest
stat_dict = {col: "mean" for col in (metsakeskus_pred_columns + koskisen_vol_columns)}
for col in time_columns: stat_dict[col] = 'min'
# get the means of the volumes per stand and minimum of each harvest_start and updatetime per stand.
# OK to take the mean of koskisen gridcells as ground truth as they're all the same anyway
volume_means_times = full_data.groupby("koski_prd_id")[metsakeskus_pred_columns + koskisen_vol_columns + time_columns].agg(stat_dict)

def calculate_metsakeskus_error(df):
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    
    metsakeskus_preds = df[metsakeskus_pred_columns]
    koskisen_vols = df[koskisen_vol_columns]
    koskisen_means = np.mean(koskisen_vols, axis=0)
    print(koskisen_means)
    rmse = np.sqrt(mean_squared_error(metsakeskus_preds, koskisen_vols, multioutput='raw_values'))
    nrmse = rmse / koskisen_means
    print(rmse)
    print(nrmse)
    

print("Metsakeskus RMSE on all stands (total, pine, spruce, deciduous):")
calculate_metsakeskus_error(volume_means_times)
print("\nMetsakeskus RMSE on stands where all gridcell preds were made before harvest:")
calculate_metsakeskus_error(volume_means_times[volume_means_times['updatetime'] < volume_means_times['harvest_start']])
173/34:
# Remember same volume order in both
metsakeskus_pred_columns = ['volume', 'volumepine', 'volumespruce', 'volumedeciduous']
koskisen_vol_columns = ['total_volume_ha', 'pine_volume_ha', 'spruce_volume_ha', 'deciduous_volume_ha']
time_columns = ['updatetime', 'harvest_start']

# aggregate volume means and times. Take minimum of both times so we can compare and get just the stands where preds were made
# before harvest
stat_dict = {col: "mean" for col in (metsakeskus_pred_columns + koskisen_vol_columns)}
for col in time_columns: stat_dict[col] = 'min'
# get the means of the volumes per stand and minimum of each harvest_start and updatetime per stand.
# OK to take the mean of koskisen gridcells as ground truth as they're all the same anyway
volume_means_times = full_data.groupby("koski_prd_id")[metsakeskus_pred_columns + koskisen_vol_columns + time_columns].agg(stat_dict)

def calculate_metsakeskus_error(df):
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    
    metsakeskus_preds = df[metsakeskus_pred_columns]
    koskisen_vols = df[koskisen_vol_columns]
    koskisen_means = np.mean(koskisen_vols, axis=0)
    print(koskisen_means)
    rmse = np.sqrt(mean_squared_error(metsakeskus_preds, koskisen_vols, multioutput='raw_values'))
    nrmse = (rmse / koskisen_means)*100
    print(rmse)
    print(nrmse)
    

print("Metsakeskus RMSE on all stands (total, pine, spruce, deciduous):")
calculate_metsakeskus_error(volume_means_times)
print("\nMetsakeskus RMSE on stands where all gridcell preds were made before harvest:")
calculate_metsakeskus_error(volume_means_times[volume_means_times['updatetime'] < volume_means_times['harvest_start']])
173/35:
# Remember same volume order in both
metsakeskus_pred_columns = ['volume', 'volumepine', 'volumespruce', 'volumedeciduous']
koskisen_vol_columns = ['total_volume_ha', 'pine_volume_ha', 'spruce_volume_ha', 'deciduous_volume_ha']
time_columns = ['updatetime', 'harvest_start']

# aggregate volume means and times. Take minimum of both times so we can compare and get just the stands where preds were made
# before harvest
stat_dict = {col: "mean" for col in (metsakeskus_pred_columns + koskisen_vol_columns)}
for col in time_columns: stat_dict[col] = 'min'
# get the means of the volumes per stand and minimum of each harvest_start and updatetime per stand.
# OK to take the mean of koskisen gridcells as ground truth as they're all the same anyway
volume_means_times = full_data.groupby("koski_prd_id")[metsakeskus_pred_columns + koskisen_vol_columns + time_columns].agg(stat_dict)

def calculate_metsakeskus_error(df):
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    
    metsakeskus_preds = df[metsakeskus_pred_columns]
    koskisen_vols = df[koskisen_vol_columns]
    koskisen_means = np.mean(koskisen_vols, axis=0)
    rmse = np.sqrt(mean_squared_error(metsakeskus_preds, koskisen_vols, multioutput='raw_values'))
    nrmse = (rmse / koskisen_means)*100
    print(rmse)
    print(nrmse)
    

print("Metsakeskus RMSE on all stands (total, pine, spruce, deciduous):")
calculate_metsakeskus_error(volume_means_times)
print("\nMetsakeskus RMSE on stands where all gridcell preds were made before harvest:")
calculate_metsakeskus_error(volume_means_times[volume_means_times['updatetime'] < volume_means_times['harvest_start']])
173/36:
# Remember same volume order in both
metsakeskus_pred_columns = ['volume', 'volumepine', 'volumespruce', 'volumedeciduous']
koskisen_vol_columns = ['total_volume_ha', 'pine_volume_ha', 'spruce_volume_ha', 'deciduous_volume_ha']
time_columns = ['updatetime', 'harvest_start']

# aggregate volume means and times. Take minimum of both times so we can compare and get just the stands where preds were made
# before harvest
stat_dict = {col: "mean" for col in (metsakeskus_pred_columns + koskisen_vol_columns)}
for col in time_columns: stat_dict[col] = 'min'
# get the means of the volumes per stand and minimum of each harvest_start and updatetime per stand.
# OK to take the mean of koskisen gridcells as ground truth as they're all the same anyway
volume_means_times = full_data.groupby("koski_prd_id")[metsakeskus_pred_columns + koskisen_vol_columns + time_columns].agg(stat_dict)

def calculate_metsakeskus_error(df):
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    
    metsakeskus_preds = df[metsakeskus_pred_columns]
    koskisen_vols = df[koskisen_vol_columns]
    koskisen_means = np.mean(koskisen_vols, axis=0).values
    rmse = np.sqrt(mean_squared_error(metsakeskus_preds, koskisen_vols, multioutput='raw_values'))
    nrmse = (rmse / koskisen_means)*100
    print(rmse)
    print(nrmse)
    

print("Metsakeskus RMSE on all stands (total, pine, spruce, deciduous):")
calculate_metsakeskus_error(volume_means_times)
print("\nMetsakeskus RMSE on stands where all gridcell preds were made before harvest:")
calculate_metsakeskus_error(volume_means_times[volume_means_times['updatetime'] < volume_means_times['harvest_start']])
173/37:
# Remember same volume order in both
metsakeskus_pred_columns = ['volume', 'volumepine', 'volumespruce', 'volumedeciduous']
koskisen_vol_columns = ['total_volume_ha', 'pine_volume_ha', 'spruce_volume_ha', 'deciduous_volume_ha']
time_columns = ['updatetime', 'harvest_start']

# aggregate volume means and times. Take minimum of both times so we can compare and get just the stands where preds were made
# before harvest
stat_dict = {col: "mean" for col in (metsakeskus_pred_columns + koskisen_vol_columns)}
for col in time_columns: stat_dict[col] = 'min'
# get the means of the volumes per stand and minimum of each harvest_start and updatetime per stand.
# OK to take the mean of koskisen gridcells as ground truth as they're all the same anyway
volume_means_times = full_data.groupby("koski_prd_id")[metsakeskus_pred_columns + koskisen_vol_columns + time_columns].agg(stat_dict)

def calculate_metsakeskus_error(df):
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    
    metsakeskus_preds = df[metsakeskus_pred_columns]
    koskisen_vols = df[koskisen_vol_columns]
    koskisen_means = np.mean(koskisen_vols, axis=0).values
    rmse = np.sqrt(mean_squared_error(metsakeskus_preds, koskisen_vols, multioutput='raw_values'))
    nrmse = (rmse / koskisen_means)*100
    print("Order: total, pine, spruce, deciduous")
    print("Groundtruth means:")
    print(koskisen_means)
    print("RMSE:")
    print(rmse)
    print("NRMSE (RMSE divided by the mean of respective species):")
    print(nrmse)
    

print("Metsakeskus RMSE on all stands (total, pine, spruce, deciduous):")
calculate_metsakeskus_error(volume_means_times)
print("\nMetsakeskus RMSE on stands where all gridcell preds were made before harvest:")
calculate_metsakeskus_error(volume_means_times[volume_means_times['updatetime'] < volume_means_times['harvest_start']])
173/38:
# Remember same volume order in both
metsakeskus_pred_columns = ['volume', 'volumepine', 'volumespruce', 'volumedeciduous']
koskisen_vol_columns = ['total_volume_ha', 'pine_volume_ha', 'spruce_volume_ha', 'deciduous_volume_ha']
time_columns = ['updatetime', 'harvest_start']

# aggregate volume means and times. Take minimum of both times so we can compare and get just the stands where preds were made
# before harvest
stat_dict = {col: "mean" for col in (metsakeskus_pred_columns + koskisen_vol_columns)}
for col in time_columns: stat_dict[col] = 'min'
# get the means of the volumes per stand and minimum of each harvest_start and updatetime per stand.
# OK to take the mean of koskisen gridcells as ground truth as they're all the same anyway
volume_means_times = full_data.groupby("koski_prd_id")[metsakeskus_pred_columns + koskisen_vol_columns + time_columns].agg(stat_dict)

def calculate_metsakeskus_error(df):
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    
    metsakeskus_preds = df[metsakeskus_pred_columns]
    koskisen_vols = df[koskisen_vol_columns]
    koskisen_means = np.mean(koskisen_vols, axis=0).values
    rmse = np.sqrt(mean_squared_error(metsakeskus_preds, koskisen_vols, multioutput='raw_values'))
    nrmse = (rmse / koskisen_means)*100
    print("Order: total, pine, spruce, deciduous")
    print("Groundtruth means:")
    print(koskisen_means)
    print("RMSE:")
    print(rmse)
    print("NRMSE (RMSE divided by the mean of respective species):")
    print(nrmse)
    

print("Metsakeskus, all stands:")
calculate_metsakeskus_error(volume_means_times)
print("\nMetsakeskus, on stands where all gridcell preds were made before harvest:")
calculate_metsakeskus_error(volume_means_times[volume_means_times['updatetime'] < volume_means_times['harvest_start']])
176/1: import data_loading
176/2: data_loading.create_test_set_from_ids("/home/tman/Work/data/koskisen/v_stand_level_features.csv", "/home/tman/Work/data/koskisen/", split_name="koskisen", id_column="prd_id")
175/1:
import sys
import os
sys.path.append('../../regressors')

import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from data import data_loading

%load_ext autoreload
%autoreload 2
%aimport data
175/2:
koskisen_folder = '../../../../data/koskisen/'

train = pd.read_csv(os.path.join(koskisen_folder, 'train.csv'))
test = pd.read_csv(os.path.join(koskisen_folder, 'test.csv'))
175/3: train.shape
175/4: test.shape
181/1:
%load_ext autoreload
%autoreload 2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import json
import os
from tqdm import tqdm
import re
import sys
from sklearn.model_selection import train_test_split
import requests
pd.options.display.float_format = '{:,.2f}'.format
# Add path to where utils.py is so metrics can be imported
sys.path.insert(0, r'../../regressors')
from data.data_loading import import_data, GeoAPI, split_from_ids
from data import data_loading
181/2:
stand_data = pd.read_csv("~/Work/data/koskisen/v_stand_level_features.csv")
gridcell_data = pd.read_csv("~/Work/data/koskisen/v_gridcell_volumes_with_coords.csv")
gridcell_data = gridcell_data.drop('hila_polygon', axis=1)
181/3:
columns_from_stand = ['prd_id', 'harvest_year', 'harvest_start']
koskisen_grids = gridcell_data.merge(stand_data[columns_from_stand], left_on="koski_prd_id", right_on="prd_id")
koskisen_grids['harvest_start'] = pd.to_datetime(koskisen_grids['harvest_start'])
181/4:
api = GeoAPI(default_locations=gridcell_data[['easting', 'northing']].values.tolist(),
                default_srid=3067, default_plot_ids=gridcell_data.hila_gridcellid.values.tolist())

def get_metsakeskus_data():
    
    if not os.path.exists(cache):
    columns_list = [["volumepine","volumespruce","volumedeciduous","volume","creationtime", "updatetime",
                     "soiltype","fertilityclass","laserheight","laserdensity"]]

    schema_list = ['metsakeskus_hila']
    tables_list = ['gridcell']

    data = api.request_data(schema_list, tables_list, columns_list, batch_size=2000)

    # Return plot_ids from index to a column.
    data.reset_index(inplace=True)
    data = data.drop_duplicates(subset='plot_id')
    
    return data

cachefile = "~/Work/data/koskisen/fulldata_metsakeskus_koskisen.csv"
if not os.path.exists(cachefile)
    metsakeskus_data = get_metsakeskus_data()
    metsakeskus_data['creationtime'] = pd.to_datetime(metsakeskus_data['creationtime'])
    metsakeskus_data['updatetime'] = pd.to_datetime(metsakeskus_data['updatetime'])
    # GeoAPI adds plot_id to corresponding rows when fetching data. We used hila_gridcellid when fetching data
    full_data = koskisen_grids.merge(metsakeskus_data, left_on="hila_gridcellid", right_on="plot_id")
    full_data.to_csv(cachefile)
else:
    full_data = pd.read_csv(cachefile)
181/5:
api = GeoAPI(default_locations=gridcell_data[['easting', 'northing']].values.tolist(),
                default_srid=3067, default_plot_ids=gridcell_data.hila_gridcellid.values.tolist())

def get_metsakeskus_data():
    columns_list = [["volumepine","volumespruce","volumedeciduous","volume","creationtime", "updatetime",
                     "soiltype","fertilityclass","laserheight","laserdensity"]]

    schema_list = ['metsakeskus_hila']
    tables_list = ['gridcell']

    data = api.request_data(schema_list, tables_list, columns_list, batch_size=2000)

    # Return plot_ids from index to a column.
    data.reset_index(inplace=True)
    data = data.drop_duplicates(subset='plot_id')
    
    return data

cachefile = "~/Work/data/koskisen/fulldata_metsakeskus_koskisen.csv"
if not os.path.exists(cachefile)
    metsakeskus_data = get_metsakeskus_data()
    metsakeskus_data['creationtime'] = pd.to_datetime(metsakeskus_data['creationtime'])
    metsakeskus_data['updatetime'] = pd.to_datetime(metsakeskus_data['updatetime'])
    # GeoAPI adds plot_id to corresponding rows when fetching data. We used hila_gridcellid when fetching data
    full_data = koskisen_grids.merge(metsakeskus_data, left_on="hila_gridcellid", right_on="plot_id")
    full_data.to_csv(cachefile)
else:
    full_data = pd.read_csv(cachefile)
181/6:
api = GeoAPI(default_locations=gridcell_data[['easting', 'northing']].values.tolist(),
                default_srid=3067, default_plot_ids=gridcell_data.hila_gridcellid.values.tolist())

def get_metsakeskus_data():
    columns_list = [["volumepine","volumespruce","volumedeciduous","volume","creationtime", "updatetime",
                     "soiltype","fertilityclass","laserheight","laserdensity"]]

    schema_list = ['metsakeskus_hila']
    tables_list = ['gridcell']

    data = api.request_data(schema_list, tables_list, columns_list, batch_size=2000)

    # Return plot_ids from index to a column.
    data.reset_index(inplace=True)
    data = data.drop_duplicates(subset='plot_id')
    
    return data

cachefile = "~/Work/data/koskisen/fulldata_metsakeskus_koskisen.csv"
if not os.path.exists(cachefile):
    metsakeskus_data = get_metsakeskus_data()
    metsakeskus_data['creationtime'] = pd.to_datetime(metsakeskus_data['creationtime'])
    metsakeskus_data['updatetime'] = pd.to_datetime(metsakeskus_data['updatetime'])
    # GeoAPI adds plot_id to corresponding rows when fetching data. We used hila_gridcellid when fetching data
    full_data = koskisen_grids.merge(metsakeskus_data, left_on="hila_gridcellid", right_on="plot_id")
    full_data.to_csv(cachefile)
else:
    full_data = pd.read_csv(cachefile)
181/7:
api = GeoAPI(default_locations=gridcell_data[['easting', 'northing']].values.tolist(),
                default_srid=3067, default_plot_ids=gridcell_data.hila_gridcellid.values.tolist())

def get_metsakeskus_data():
    columns_list = [["volumepine","volumespruce","volumedeciduous","volume","creationtime", "updatetime",
                     "soiltype","fertilityclass","laserheight","laserdensity"]]

    schema_list = ['metsakeskus_hila']
    tables_list = ['gridcell']

    data = api.request_data(schema_list, tables_list, columns_list, batch_size=2000)

    # Return plot_ids from index to a column.
    data.reset_index(inplace=True)
    data = data.drop_duplicates(subset='plot_id')
    
    return data

cachefile = "~/Work/data/koskisen/fulldata_metsakeskus_koskisen.csv"
print(os.path.exists(cachefile))
if not os.path.exists(cachefile):
    metsakeskus_data = get_metsakeskus_data()
    metsakeskus_data['creationtime'] = pd.to_datetime(metsakeskus_data['creationtime'])
    metsakeskus_data['updatetime'] = pd.to_datetime(metsakeskus_data['updatetime'])
    # GeoAPI adds plot_id to corresponding rows when fetching data. We used hila_gridcellid when fetching data
    full_data = koskisen_grids.merge(metsakeskus_data, left_on="hila_gridcellid", right_on="plot_id")
    full_data.to_csv(cachefile)
else:
    full_data = pd.read_csv(cachefile)
181/8:
api = GeoAPI(default_locations=gridcell_data[['easting', 'northing']].values.tolist(),
                default_srid=3067, default_plot_ids=gridcell_data.hila_gridcellid.values.tolist())

def get_metsakeskus_data():
    columns_list = [["volumepine","volumespruce","volumedeciduous","volume","creationtime", "updatetime",
                     "soiltype","fertilityclass","laserheight","laserdensity"]]

    schema_list = ['metsakeskus_hila']
    tables_list = ['gridcell']

    data = api.request_data(schema_list, tables_list, columns_list, batch_size=2000)

    # Return plot_ids from index to a column.
    data.reset_index(inplace=True)
    data = data.drop_duplicates(subset='plot_id')
    
    return data

cachefile = "~/Work/data/koskisen/fulldata_metsakeskus_koskisen.csv"
print(os.path.exists(cachefile))
full_data = pd.read_csv(cachefile)
if not os.path.exists(cachefile):
    metsakeskus_data = get_metsakeskus_data()
    metsakeskus_data['creationtime'] = pd.to_datetime(metsakeskus_data['creationtime'])
    metsakeskus_data['updatetime'] = pd.to_datetime(metsakeskus_data['updatetime'])
    # GeoAPI adds plot_id to corresponding rows when fetching data. We used hila_gridcellid when fetching data
    full_data = koskisen_grids.merge(metsakeskus_data, left_on="hila_gridcellid", right_on="plot_id")
    full_data.to_csv(cachefile)
else:
    full_data = pd.read_csv(cachefile)
181/9:
api = GeoAPI(default_locations=gridcell_data[['easting', 'northing']].values.tolist(),
                default_srid=3067, default_plot_ids=gridcell_data.hila_gridcellid.values.tolist())

def get_metsakeskus_data():
    columns_list = [["volumepine","volumespruce","volumedeciduous","volume","creationtime", "updatetime",
                     "soiltype","fertilityclass","laserheight","laserdensity"]]

    schema_list = ['metsakeskus_hila']
    tables_list = ['gridcell']

    data = api.request_data(schema_list, tables_list, columns_list, batch_size=2000)

    # Return plot_ids from index to a column.
    data.reset_index(inplace=True)
    data = data.drop_duplicates(subset='plot_id')
    
    return data

cachefile = "/home/tman/Work/data/koskisen/fulldata_metsakeskus_koskisen.csv"
print(os.path.exists(cachefile))
full_data = pd.read_csv(cachefile)
if not os.path.exists(cachefile):
    metsakeskus_data = get_metsakeskus_data()
    metsakeskus_data['creationtime'] = pd.to_datetime(metsakeskus_data['creationtime'])
    metsakeskus_data['updatetime'] = pd.to_datetime(metsakeskus_data['updatetime'])
    # GeoAPI adds plot_id to corresponding rows when fetching data. We used hila_gridcellid when fetching data
    full_data = koskisen_grids.merge(metsakeskus_data, left_on="hila_gridcellid", right_on="plot_id")
    full_data.to_csv(cachefile)
else:
    full_data = pd.read_csv(cachefile)
181/10:
api = GeoAPI(default_locations=gridcell_data[['easting', 'northing']].values.tolist(),
                default_srid=3067, default_plot_ids=gridcell_data.hila_gridcellid.values.tolist())

def get_metsakeskus_data():
    columns_list = [["volumepine","volumespruce","volumedeciduous","volume","creationtime", "updatetime",
                     "soiltype","fertilityclass","laserheight","laserdensity"]]

    schema_list = ['metsakeskus_hila']
    tables_list = ['gridcell']

    data = api.request_data(schema_list, tables_list, columns_list, batch_size=2000)

    # Return plot_ids from index to a column.
    data.reset_index(inplace=True)
    data = data.drop_duplicates(subset='plot_id')
    
    return data

cachefile = "/home/tman/Work/data/koskisen/fulldata_metsakeskus_koskisen.csv"
if not os.path.exists(cachefile):
    metsakeskus_data = get_metsakeskus_data()
    metsakeskus_data['creationtime'] = pd.to_datetime(metsakeskus_data['creationtime'])
    metsakeskus_data['updatetime'] = pd.to_datetime(metsakeskus_data['updatetime'])
    # GeoAPI adds plot_id to corresponding rows when fetching data. We used hila_gridcellid when fetching data
    full_data = koskisen_grids.merge(metsakeskus_data, left_on="hila_gridcellid", right_on="plot_id")
    full_data.to_csv(cachefile)
else:
    full_data = pd.read_csv(cachefile)
181/11: full_data[:2]
181/12: full_data.columns
181/13: full_data.drop('Unnamed: 0')
181/14: full_data.drop('Unnamed: 0', axis=1)
181/15: full_data = full_data.drop('Unnamed: 0', axis=1)
182/1:
from functools import reduce
import os
import sys
sys.path.append('../../regressors/')

import pandas as pd
import numpy as np

from data import data_loading

%load_ext autoreload
%autoreload 2
%aimport data
182/2:
def fetch_if_not_cached(data_group, api):
    cache = os.path.join(koskisen_folder, data_group + '.csv')
    
    if not os.path.exists(cache):
        data = api.request_data(data_groups=[data_group])
        data = data.reset_index()
        data = data.drop_duplicates(subset='plot_id')
        
        data.to_csv(cache, index=False)
    else:
        data = pd.read_csv(cache)
        
    return data
182/3: koskisen_folder = '../../../../data/koskisen/'
182/4: stand_data = pd.read_csv(os.path.join(koskisen_folder, 'v_stand_level_features.csv'))
182/5:
api = data_loading.GeoAPI(default_locations=stand_data[['easting', 'northing']].values.tolist(), 
                     default_srid=3067, default_plot_ids=stand_data.prd_id.values.tolist())

data_groups = ['soilgrids', 'climate_data']
data_frames = [fetch_if_not_cached(data_group, api) for data_group in data_groups]

scalar_df = reduce(lambda x,y: pd.merge(x,y,on='plot_id', how='outer'), data_frames)
182/6:
api = data_loading.GeoAPI(default_locations=stand_data[['easting', 'northing']].values.tolist(), 
                     default_srid=3067, default_plot_ids=stand_data.prd_id.values.tolist())

data_groups = ['soilgrids', 'climate_data', 'copernicus', 'physical']
data_frames = [fetch_if_not_cached(data_group, api) for data_group in data_groups]

scalar_df = reduce(lambda x,y: pd.merge(x,y,on='plot_id', how='outer'), data_frames)
182/7: features = scalar_df.copy()
182/8: features.isna().mean(axis=0)
182/9:
features = scalar_df.dropna()

assert features.isna().sum().sum() == 0
182/10: features.dtypes
182/11:
categorical_columns = ['texture_class_usda_30cm', 'texture_class_usda_200cm', 
                      'usda_2014_suborder_class', 'wrb_2006_subgroup_class']

features.loc[:, categorical_columns] = features[categorical_columns].astype('category')
print(features[categorical_columns].describe())

features = pd.get_dummies(features)
182/12: features.describe().T
182/13:
target_columns = ['total_m3_ha', 'pine_m3_ha', 'spruce_m3_ha', 'deciduous_m3_ha']

X = features.copy()
y = stand_data[target_columns].loc[X.index, :]
182/14:
X_train, X_test = data_loading.split_from_ids(features, split_name='koskisen', id_column='plot_id')

y_train, y_test = y.loc[X_train.index, :], y.loc[X_test.index, :]

X_train = X_train.drop('plot_id', axis=1)
X_test = X_test.drop('plot_id', axis=1)
182/15:
assert X_train.shape[0] == y_train.shape[0]
assert (X_train.index == y_train.index).all()
182/16:
target_column = 'total_m3_ha'

y_train_col, y_test_col = y_train[target_column], y_test[target_column]
182/17:
import GPyOpt
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score

domain = [
    {'name': 'learning_rate', 'type': 'continuous', 'domain': (0, 1)},
    {'name': 'gamma', 'type': 'continuous', 'domain': (0, 5)},
    {'name': 'max_depth', 'type': 'discrete', 'domain': (1, 50)},
    {'name': 'n_estimators', 'type': 'discrete', 'domain': (1, 300)},
    {'name': 'min_child_weight', 'type': 'discrete', 'domain': (1, 10)}
]

def f(params):
    params = params[0]
    estimator = XGBRegressor(learning_rate=params[0],
                           gamma=params[1],
                           max_depth=int(params[2]),
                           n_estimators=int(params[3]),
                           min_child_weight=int(params[4])
                            )
    
    score = -cross_val_score(estimator, X_train, y_train_col, 
                            scoring='neg_mean_squared_error').mean()
    
    return np.array(score)

np.random.seed(42)
optimizer = GPyOpt.methods.BayesianOptimization(f=f, domain=domain, acquisition_type='MPI', 
                                                num_cores=4, exact_feval=True)
183/1:
from functools import reduce
import os
import sys
sys.path.append('../../regressors/')

import pandas as pd
import numpy as np

from data import data_loading

%load_ext autoreload
%autoreload 2
%aimport data
183/2:
def fetch_if_not_cached(data_group, api):
    cache = os.path.join(koskisen_folder, data_group + '.csv')
    
    if not os.path.exists(cache):
        data = api.request_data(data_groups=[data_group])
        data = data.reset_index()
        data = data.drop_duplicates(subset='plot_id')
        
        data.to_csv(cache, index=False)
    else:
        data = pd.read_csv(cache)
        
    return data
183/3: koskisen_folder = '../../../../data/koskisen/'
183/4: stand_data = pd.read_csv(os.path.join(koskisen_folder, 'v_stand_level_features.csv'))
183/5:
api = data_loading.GeoAPI(default_locations=stand_data[['easting', 'northing']].values.tolist(), 
                     default_srid=3067, default_plot_ids=stand_data.prd_id.values.tolist())

data_groups = ['soilgrids', 'climate_data', 'copernicus', 'physical']
data_frames = [fetch_if_not_cached(data_group, api) for data_group in data_groups]

scalar_df = reduce(lambda x,y: pd.merge(x,y,on='plot_id', how='outer'), data_frames)
183/6: features = scalar_df.copy()
183/7: features.isna().mean(axis=0)
183/8:
features = scalar_df.dropna()

assert features.isna().sum().sum() == 0
183/9: features.dtypes
183/10:
categorical_columns = ['texture_class_usda_30cm', 'texture_class_usda_200cm', 
                      'usda_2014_suborder_class', 'wrb_2006_subgroup_class']

features.loc[:, categorical_columns] = features[categorical_columns].astype('category')
print(features[categorical_columns].describe())

features = pd.get_dummies(features)
183/11: features.describe().T
183/12:
target_columns = ['total_m3_ha', 'pine_m3_ha', 'spruce_m3_ha', 'deciduous_m3_ha']

X = features.copy()
y = stand_data[target_columns].loc[X.index, :]
183/13:
X_train, X_test = data_loading.split_from_ids(features, split_name='koskisen', id_column='plot_id')

y_train, y_test = y.loc[X_train.index, :], y.loc[X_test.index, :]

X_train = X_train.drop('plot_id', axis=1)
X_test = X_test.drop('plot_id', axis=1)
183/14: print(X_train.shape)
183/15:
print(X_train.shape)
print(X_test.shape)
183/16:
assert X_train.shape[0] == y_train.shape[0]
assert (X_train.index == y_train.index).all()
183/17:
target_column = 'total_m3_ha'

y_train_col, y_test_col = y_train[target_column], y_test[target_column]
183/18:
import GPyOpt
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score

domain = [
    {'name': 'learning_rate', 'type': 'continuous', 'domain': (0, 1)},
    {'name': 'gamma', 'type': 'continuous', 'domain': (0, 5)},
    {'name': 'max_depth', 'type': 'discrete', 'domain': (1, 50)},
    {'name': 'n_estimators', 'type': 'discrete', 'domain': (1, 300)},
    {'name': 'min_child_weight', 'type': 'discrete', 'domain': (1, 10)}
]

def f(params):
    params = params[0]
    estimator = XGBRegressor(learning_rate=params[0],
                           gamma=params[1],
                           max_depth=int(params[2]),
                           n_estimators=int(params[3]),
                           min_child_weight=int(params[4])
                            )
    
    score = -cross_val_score(estimator, X_train, y_train_col, 
                            scoring='neg_mean_squared_error').mean()
    
    return np.array(score)

np.random.seed(42)
optimizer = GPyOpt.methods.BayesianOptimization(f=f, domain=domain, acquisition_type='MPI', 
                                                num_cores=4, exact_feval=True)
183/19:
max_iter = 70

optimizer.run_optimization(max_iter=max_iter, verbosity=True)
183/20: optimizer.plot_convergence()
183/21: np.sqrt(optimizer.Y.min())
183/22:
parameter_names = ['learning_rate', 'gamma', 'max_depth', 'n_estimators', 'min_child_weight']
best_parameters = dict(zip(parameter_names, optimizer.X[optimizer.Y.argmin()]))
183/23:
best_parameters['max_depth'] = int(best_parameters['max_depth'])
best_parameters['n_estimators'] = int(best_parameters['n_estimators'])
best_parameters['min_child_weight'] = int(best_parameters['min_child_weight'])
183/24:
model = XGBRegressor(**best_parameters)
model.fit(X_train, y_train_col)
183/25:
from sklearn.metrics import mean_squared_error, mean_absolute_error

pred = model.predict(X_test)
mse = mean_squared_error(y_test_col, pred)
rmse = np.sqrt(mse)
nrmse = rmse / np.mean(y_test_col) * 100
mae = mean_absolute_error(y_test_col, pred)

print('MSE: {:.2f}\nRMSE: {:.2f}\nNRMSE: {:.2f} %\nMAE: {:.2f}'.format(mse, rmse, nrmse, mae))
183/26: X_test.shape
184/1:
%load_ext autoreload
%autoreload 2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import json
import os
from tqdm import tqdm
import re
import sys
from sklearn.model_selection import train_test_split
import requests
pd.options.display.float_format = '{:,.2f}'.format
# Add path to where utils.py is so metrics can be imported
sys.path.insert(0, r'../../regressors')
from data.data_loading import import_data, GeoAPI, split_from_ids
from data import data_loading
184/2:
stand_data = pd.read_csv("~/Work/data/koskisen/v_stand_level_features.csv")
gridcell_data = pd.read_csv("~/Work/data/koskisen/v_gridcell_volumes_with_coords.csv")
gridcell_data = gridcell_data.drop('hila_polygon', axis=1)
184/3:
columns_from_stand = ['prd_id', 'harvest_year', 'harvest_start']
koskisen_grids = gridcell_data.merge(stand_data[columns_from_stand], left_on="koski_prd_id", right_on="prd_id")
koskisen_grids['harvest_start'] = pd.to_datetime(koskisen_grids['harvest_start'])
184/4:
api = GeoAPI(default_locations=gridcell_data[['easting', 'northing']].values.tolist(),
                default_srid=3067, default_plot_ids=gridcell_data.hila_gridcellid.values.tolist())

def get_metsakeskus_data():
    columns_list = [["volumepine","volumespruce","volumedeciduous","volume","creationtime", "updatetime",
                     "soiltype","fertilityclass","laserheight","laserdensity"]]

    schema_list = ['metsakeskus_hila']
    tables_list = ['gridcell']

    data = api.request_data(schema_list, tables_list, columns_list, batch_size=2000)

    # Return plot_ids from index to a column.
    data.reset_index(inplace=True)
    data = data.drop_duplicates(subset='plot_id')
    
    return data

cachefile = "/home/tman/Work/data/koskisen/fulldata_metsakeskus_koskisen.csv"
if not os.path.exists(cachefile):
    metsakeskus_data = get_metsakeskus_data()
    metsakeskus_data['creationtime'] = pd.to_datetime(metsakeskus_data['creationtime'])
    metsakeskus_data['updatetime'] = pd.to_datetime(metsakeskus_data['updatetime'])
    # GeoAPI adds plot_id to corresponding rows when fetching data. We used hila_gridcellid when fetching data
    full_data = koskisen_grids.merge(metsakeskus_data, left_on="hila_gridcellid", right_on="plot_id")
    full_data.to_csv(cachefile, index=False)
else:
    full_data = pd.read_csv(cachefile)
184/5: full_data = full_data.drop('Unnamed: 0', axis=1)
184/6:
# Remember same volume order in both
metsakeskus_pred_columns = ['volume', 'volumepine', 'volumespruce', 'volumedeciduous']
koskisen_vol_columns = ['total_volume_ha', 'pine_volume_ha', 'spruce_volume_ha', 'deciduous_volume_ha']
time_columns = ['updatetime', 'harvest_start']

# aggregate volume means and times. Take minimum of both times so we can compare and get just the stands where preds were made
# before harvest
stat_dict = {col: "mean" for col in (metsakeskus_pred_columns + koskisen_vol_columns)}
for col in time_columns: stat_dict[col] = 'min'
# get the means of the volumes per stand and minimum of each harvest_start and updatetime per stand.
# OK to take the mean of koskisen gridcells as ground truth as they're all the same anyway
volume_means_times = full_data.groupby("koski_prd_id")[metsakeskus_pred_columns + koskisen_vol_columns + time_columns].agg(stat_dict)

def calculate_metsakeskus_error(df):
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    
    metsakeskus_preds = df[metsakeskus_pred_columns]
    koskisen_vols = df[koskisen_vol_columns]
    koskisen_means = np.mean(koskisen_vols, axis=0).values
    rmse = np.sqrt(mean_squared_error(metsakeskus_preds, koskisen_vols, multioutput='raw_values'))
    nrmse = (rmse / koskisen_means)*100
    print("Order: total, pine, spruce, deciduous")
    print("Groundtruth means:")
    print(koskisen_means)
    print("RMSE:")
    print(rmse)
    print("NRMSE (RMSE divided by the mean of respective species):")
    print(nrmse)
    

print("Metsakeskus, all stands:")
print(volume_means_times[volume_means_times['updatetime'] < volume_means_times['harvest_start']].shape)
calculate_metsakeskus_error(volume_means_times)
print("\nMetsakeskus, on stands where all gridcell preds were made before harvest:")
calculate_metsakeskus_error(volume_means_times[volume_means_times['updatetime'] < volume_means_times['harvest_start']])
183/27:
from sklearn.metrics import mean_squared_error, mean_absolute_error

pred = model.predict(X_test)
mse = mean_squared_error(y_test_col, pred)
rmse = np.sqrt(mse)
colmean = np.mean(y_test_col)
nrmse = rmse / colmean * 100
mae = mean_absolute_error(y_test_col, pred)

print('MSE: {:.2f}\nRMSE: {:.2f}\nNRMSE: {:.2f} %\nMAE: {:.2f}\GT Mean: {:.2f}'.format(mse, rmse, nrmse, mae, colmean))
183/28:
from sklearn.metrics import mean_squared_error, mean_absolute_error

pred = model.predict(X_test)
mse = mean_squared_error(y_test_col, pred)
rmse = np.sqrt(mse)
colmean = np.mean(y_test_col)
nrmse = rmse / colmean * 100
mae = mean_absolute_error(y_test_col, pred)

print('MSE: {:.2f}\nRMSE: {:.2f}\nNRMSE: {:.2f} %\nMAE: {:.2f}\nGT Mean: {:.2f}'.format(mse, rmse, nrmse, mae, colmean))
184/7: volume_means_times.shape
183/29:
target_column = 'pine_m3_ha'

y_train_col, y_test_col = y_train[target_column], y_test[target_column]
183/30:
import GPyOpt
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score

domain = [
    {'name': 'learning_rate', 'type': 'continuous', 'domain': (0, 1)},
    {'name': 'gamma', 'type': 'continuous', 'domain': (0, 5)},
    {'name': 'max_depth', 'type': 'discrete', 'domain': (1, 50)},
    {'name': 'n_estimators', 'type': 'discrete', 'domain': (1, 300)},
    {'name': 'min_child_weight', 'type': 'discrete', 'domain': (1, 10)}
]

def f(params):
    params = params[0]
    estimator = XGBRegressor(learning_rate=params[0],
                           gamma=params[1],
                           max_depth=int(params[2]),
                           n_estimators=int(params[3]),
                           min_child_weight=int(params[4])
                            )
    
    score = -cross_val_score(estimator, X_train, y_train_col, 
                            scoring='neg_mean_squared_error').mean()
    
    return np.array(score)

np.random.seed(42)
optimizer = GPyOpt.methods.BayesianOptimization(f=f, domain=domain, acquisition_type='MPI', 
                                                num_cores=4, exact_feval=True)
183/31:
max_iter = 70

optimizer.run_optimization(max_iter=max_iter, verbosity=True)
183/32: optimizer.plot_convergence()
183/33: np.sqrt(optimizer.Y.min())
183/34:
parameter_names = ['learning_rate', 'gamma', 'max_depth', 'n_estimators', 'min_child_weight']
best_parameters = dict(zip(parameter_names, optimizer.X[optimizer.Y.argmin()]))
183/35:
best_parameters['max_depth'] = int(best_parameters['max_depth'])
best_parameters['n_estimators'] = int(best_parameters['n_estimators'])
best_parameters['min_child_weight'] = int(best_parameters['min_child_weight'])
183/36:
model = XGBRegressor(**best_parameters)
model.fit(X_train, y_train_col)
183/37:
from sklearn.metrics import mean_squared_error, mean_absolute_error

pred = model.predict(X_test)
mse = mean_squared_error(y_test_col, pred)
rmse = np.sqrt(mse)
colmean = np.mean(y_test_col)
nrmse = rmse / colmean * 100
mae = mean_absolute_error(y_test_col, pred)

print('MSE: {:.2f}\nRMSE: {:.2f}\nNRMSE: {:.2f} %\nMAE: {:.2f}\nGT Mean: {:.2f}'.format(mse, rmse, nrmse, mae, colmean))
183/38:
target_column = 'spruce_m3_ha'

y_train_col, y_test_col = y_train[target_column], y_test[target_column]
183/39:
import GPyOpt
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score

domain = [
    {'name': 'learning_rate', 'type': 'continuous', 'domain': (0, 1)},
    {'name': 'gamma', 'type': 'continuous', 'domain': (0, 5)},
    {'name': 'max_depth', 'type': 'discrete', 'domain': (1, 50)},
    {'name': 'n_estimators', 'type': 'discrete', 'domain': (1, 300)},
    {'name': 'min_child_weight', 'type': 'discrete', 'domain': (1, 10)}
]

def f(params):
    params = params[0]
    estimator = XGBRegressor(learning_rate=params[0],
                           gamma=params[1],
                           max_depth=int(params[2]),
                           n_estimators=int(params[3]),
                           min_child_weight=int(params[4])
                            )
    
    score = -cross_val_score(estimator, X_train, y_train_col, 
                            scoring='neg_mean_squared_error').mean()
    
    return np.array(score)

np.random.seed(42)
optimizer = GPyOpt.methods.BayesianOptimization(f=f, domain=domain, acquisition_type='MPI', 
                                                num_cores=4, exact_feval=True)
183/40:
max_iter = 70

optimizer.run_optimization(max_iter=max_iter, verbosity=True)
183/41: optimizer.plot_convergence()
183/42: np.sqrt(optimizer.Y.min())
183/43:
parameter_names = ['learning_rate', 'gamma', 'max_depth', 'n_estimators', 'min_child_weight']
best_parameters = dict(zip(parameter_names, optimizer.X[optimizer.Y.argmin()]))
183/44:
best_parameters['max_depth'] = int(best_parameters['max_depth'])
best_parameters['n_estimators'] = int(best_parameters['n_estimators'])
best_parameters['min_child_weight'] = int(best_parameters['min_child_weight'])
183/45:
model = XGBRegressor(**best_parameters)
model.fit(X_train, y_train_col)
183/46:
from sklearn.metrics import mean_squared_error, mean_absolute_error

pred = model.predict(X_test)
mse = mean_squared_error(y_test_col, pred)
rmse = np.sqrt(mse)
colmean = np.mean(y_test_col)
nrmse = rmse / colmean * 100
mae = mean_absolute_error(y_test_col, pred)

print('MSE: {:.2f}\nRMSE: {:.2f}\nNRMSE: {:.2f} %\nMAE: {:.2f}\nGT Mean: {:.2f}'.format(mse, rmse, nrmse, mae, colmean))
183/47:
target_column = 'deciduous_m3_ha'

y_train_col, y_test_col = y_train[target_column], y_test[target_column]
183/48:
import GPyOpt
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score

domain = [
    {'name': 'learning_rate', 'type': 'continuous', 'domain': (0, 1)},
    {'name': 'gamma', 'type': 'continuous', 'domain': (0, 5)},
    {'name': 'max_depth', 'type': 'discrete', 'domain': (1, 50)},
    {'name': 'n_estimators', 'type': 'discrete', 'domain': (1, 300)},
    {'name': 'min_child_weight', 'type': 'discrete', 'domain': (1, 10)}
]

def f(params):
    params = params[0]
    estimator = XGBRegressor(learning_rate=params[0],
                           gamma=params[1],
                           max_depth=int(params[2]),
                           n_estimators=int(params[3]),
                           min_child_weight=int(params[4])
                            )
    
    score = -cross_val_score(estimator, X_train, y_train_col, 
                            scoring='neg_mean_squared_error').mean()
    
    return np.array(score)

np.random.seed(42)
optimizer = GPyOpt.methods.BayesianOptimization(f=f, domain=domain, acquisition_type='MPI', 
                                                num_cores=4, exact_feval=True)
183/49:
max_iter = 70

optimizer.run_optimization(max_iter=max_iter, verbosity=True)
183/50: optimizer.plot_convergence()
183/51: np.sqrt(optimizer.Y.min())
183/52:
parameter_names = ['learning_rate', 'gamma', 'max_depth', 'n_estimators', 'min_child_weight']
best_parameters = dict(zip(parameter_names, optimizer.X[optimizer.Y.argmin()]))
183/53:
best_parameters['max_depth'] = int(best_parameters['max_depth'])
best_parameters['n_estimators'] = int(best_parameters['n_estimators'])
best_parameters['min_child_weight'] = int(best_parameters['min_child_weight'])
183/54:
model = XGBRegressor(**best_parameters)
model.fit(X_train, y_train_col)
183/55:
from sklearn.metrics import mean_squared_error, mean_absolute_error

pred = model.predict(X_test)
mse = mean_squared_error(y_test_col, pred)
rmse = np.sqrt(mse)
colmean = np.mean(y_test_col)
nrmse = rmse / colmean * 100
mae = mean_absolute_error(y_test_col, pred)

print('MSE: {:.2f}\nRMSE: {:.2f}\nNRMSE: {:.2f} %\nMAE: {:.2f}\nGT Mean: {:.2f}'.format(mse, rmse, nrmse, mae, colmean))
186/1:
from functools import reduce
import os
from tqdm import tqdm_notebook
import sys
sys.path.append('../../regressors/')

import pandas as pd
import seaborn as sns
import numpy as np

from data import data_loading

%load_ext autoreload
%autoreload 2
%aimport data
186/2:
koskisen_folder = "/home/tman/Work/data/koskisen"
stand_data = pd.read_csv(os.path.join(koskisen_folder, 'v_stand_level_features.csv'))
grid_data = pd.read_csv(os.path.join(koskisen_folder, 'v_gridcell_volumes_with_coords_unique.csv'))
186/3:
def fetch_if_not_cached(data_group, api, output_folder):
    cache = os.path.join(output_folder, data_group + '.csv')
    
    if not os.path.exists(cache):
        data = api.request_data(data_groups=[data_group])
        data = data.reset_index()
        data = data.drop_duplicates(subset='plot_id')
        
        data.to_csv(cache, index=False)
    else:
        data = pd.read_csv(cache)
        
    return data

def fetch_specific_data(api, columns_list, schema_list, tables_list, output_folder, csv_name):
    # Fetch data that is not in a data group
    #columns_list = [["volumepine","volumespruce","volumedeciduous","volume","creationtime", "updatetime",
    #                 "soiltype","fertilityclass","laserheight","laserdensity"]]

    #schema_list = ['metsakeskus_hila']
    #tables_list = ['gridcell']
    cache = os.path.join(output_folder, data_group + '.csv')
    
    if not os.path.exists(cache):
        data = api.request_data(schema_list, tables_list, columns_list, batch_size=2000)
        data = data.reset_index()
        data = data.drop_duplicates(subset='plot_id')
        
        data.to_csv(cache, index=False)
    else:
        data = pd.read_csv(cache)
        
    return data
186/4:
# Get grid data

grid_data_folder = os.path.join(koskisen_folder, 'grid')
os.makedirs(grid_data_folder, exist_ok=True)

api = data_loading.GeoAPI(default_locations=grid_data[['easting', 'northing']].values.tolist(), 
                     default_srid=3067, default_plot_ids=grid_data.hila_gridcellid.values.tolist())

data_groups = ['soilgrids', 'climate_data', 'copernicus', 'physical']
data_frames = [fetch_if_not_cached(data_group, api, grid_data_folder) for data_group in data_groups]

#scalar_grid_df = reduce(lambda x,y: pd.merge(x,y,on='plot_id', how='outer'), data_frames)
186/5:
# Get grid data with data groups

grid_data_folder = os.path.join(koskisen_folder, 'grid')
os.makedirs(grid_data_folder, exist_ok=True)

api = data_loading.GeoAPI(default_locations=grid_data[['easting', 'northing']].values.tolist(), 
                     default_srid=3067, default_plot_ids=grid_data.hila_gridcellid.values.tolist())

data_groups = ['soilgrids', 'climate_data', 'copernicus', 'physical']
data_frames = [fetch_if_not_cached(data_group, api, grid_data_folder) for data_group in data_groups]

#scalar_grid_df = reduce(lambda x,y: pd.merge(x,y,on='plot_id', how='outer'), data_frames)
tables_list = ['lidar_p10', 'lidar_p75', 'lidar_p80', 'lidar_vol_cov', 'lidar_pct_r1_above_mean', 'lidar_z_mean_sq']
schema_list = ['finland'] * len(tables_list)
columns_list = [None] * len(tables_list)
lidar_data = fetch_specific_data(api, columns_list, schema_list, tables_list, grid_data_folder, "lidar_data.csv")
186/6:
def fetch_if_not_cached(data_group, api, output_folder):
    cache = os.path.join(output_folder, data_group + '.csv')
    
    if not os.path.exists(cache):
        data = api.request_data(data_groups=[data_group])
        data = data.reset_index()
        data = data.drop_duplicates(subset='plot_id')
        
        data.to_csv(cache, index=False)
    else:
        data = pd.read_csv(cache)
        
    return data

def fetch_specific_data(api, columns_list, schema_list, tables_list, output_folder, csv_name):
    # Fetch data that is not in a data group
    #columns_list = [["volumepine","volumespruce","volumedeciduous","volume","creationtime", "updatetime",
    #                 "soiltype","fertilityclass","laserheight","laserdensity"]]

    #schema_list = ['metsakeskus_hila']
    #tables_list = ['gridcell']
    cache = os.path.join(output_folder, csv_name)
    
    if not os.path.exists(cache):
        data = api.request_data(schema_list, tables_list, columns_list, batch_size=2000)
        data = data.reset_index()
        data = data.drop_duplicates(subset='plot_id')
        
        data.to_csv(cache, index=False)
    else:
        data = pd.read_csv(cache)
        
    return data
186/7:
# Get grid data with data groups

grid_data_folder = os.path.join(koskisen_folder, 'grid')
os.makedirs(grid_data_folder, exist_ok=True)

api = data_loading.GeoAPI(default_locations=grid_data[['easting', 'northing']].values.tolist(), 
                     default_srid=3067, default_plot_ids=grid_data.hila_gridcellid.values.tolist())

data_groups = ['soilgrids', 'climate_data', 'copernicus', 'physical']
data_frames = [fetch_if_not_cached(data_group, api, grid_data_folder) for data_group in data_groups]

#scalar_grid_df = reduce(lambda x,y: pd.merge(x,y,on='plot_id', how='outer'), data_frames)
tables_list = ['lidar_p10', 'lidar_p75', 'lidar_p80', 'lidar_vol_cov', 'lidar_pct_r1_above_mean', 'lidar_z_mean_sq']
schema_list = ['finland'] * len(tables_list)
columns_list = [None] * len(tables_list)
lidar_data = fetch_specific_data(api, columns_list, schema_list, tables_list, grid_data_folder, "lidar_data.csv")
186/8:
# Get grid data with data groups

grid_data_folder = os.path.join(koskisen_folder, 'grid')
os.makedirs(grid_data_folder, exist_ok=True)

api = data_loading.GeoAPI(default_locations=grid_data[['easting', 'northing']].values.tolist(), 
                     default_srid=3067, default_plot_ids=grid_data.hila_gridcellid.values.tolist())

data_groups = ['soilgrids', 'climate_data', 'copernicus', 'physical']
data_frames = [fetch_if_not_cached(data_group, api, grid_data_folder) for data_group in data_groups]

#scalar_grid_df = reduce(lambda x,y: pd.merge(x,y,on='plot_id', how='outer'), data_frames)
tables_list = ['lidar_p10', 'lidar_p75', 'lidar_p80', 'lidar_vol_cov', 'lidar_pct_r1_above_mean', 'lidar_z_mean_sq']
schema_list = ['finland'] * len(tables_list)
columns_list = [[None]] * len(tables_list)
lidar_data = fetch_specific_data(api, columns_list, schema_list, tables_list, grid_data_folder, "lidar_data.csv")
186/9:
# Get grid data with data groups

grid_data_folder = os.path.join(koskisen_folder, 'grid')
os.makedirs(grid_data_folder, exist_ok=True)

api = data_loading.GeoAPI(default_locations=grid_data[['easting', 'northing']].values.tolist(), 
                     default_srid=3067, default_plot_ids=grid_data.hila_gridcellid.values.tolist())

data_groups = ['soilgrids', 'climate_data', 'copernicus', 'physical']
data_frames = [fetch_if_not_cached(data_group, api, grid_data_folder) for data_group in data_groups]

#scalar_grid_df = reduce(lambda x,y: pd.merge(x,y,on='plot_id', how='outer'), data_frames)
tables_list = ['lidar_p10', 'lidar_p75', 'lidar_p80', 'lidar_vol_cov', 'lidar_pct_r1_above_mean', 'lidar_z_mean_sq']
schema_list = ['finland'] * len(tables_list)
columns_list = ['null'] * len(tables_list)
lidar_data = fetch_specific_data(api, columns_list, schema_list, tables_list, grid_data_folder, "lidar_data.csv")
186/10:
# Get grid data with data groups

grid_data_folder = os.path.join(koskisen_folder, 'grid')
os.makedirs(grid_data_folder, exist_ok=True)

api = data_loading.GeoAPI(default_locations=grid_data[['easting', 'northing']].values.tolist(), 
                     default_srid=3067, default_plot_ids=grid_data.hila_gridcellid.values.tolist())

data_groups = ['soilgrids', 'climate_data', 'copernicus', 'physical']
data_frames = [fetch_if_not_cached(data_group, api, grid_data_folder) for data_group in data_groups]

#scalar_grid_df = reduce(lambda x,y: pd.merge(x,y,on='plot_id', how='outer'), data_frames)
tables_list = ['lidar_p10', 'lidar_p75', 'lidar_p80', 'lidar_vol_cov', 'lidar_pct_r1_above_mean', 'lidar_z_mean_sq']
schema_list = ['finland'] * len(tables_list)
columns_list = [None] * len(tables_list)
lidar_data = fetch_specific_data(api, columns_list, schema_list, tables_list, grid_data_folder, "lidar_data.csv")
186/11:
# Get grid data with data groups

grid_data_folder = os.path.join(koskisen_folder, 'grid')
os.makedirs(grid_data_folder, exist_ok=True)

api = data_loading.GeoAPI(default_locations=grid_data[['easting', 'northing']].values.tolist(), 
                     default_srid=3067, default_plot_ids=grid_data.hila_gridcellid.values.tolist())

data_groups = ['soilgrids', 'climate_data', 'copernicus', 'physical']
data_frames = [fetch_if_not_cached(data_group, api, grid_data_folder) for data_group in data_groups]

#scalar_grid_df = reduce(lambda x,y: pd.merge(x,y,on='plot_id', how='outer'), data_frames)
tables_list = ['lidar_p_10', 'lidar_p_75', 'lidar_p_80', 'lidar_vol_cov', 'lidar_pct_r1_above_mean', 'lidar_z_mean_sq']
schema_list = ['finland'] * len(tables_list)
columns_list = [None] * len(tables_list)
lidar_data = fetch_specific_data(api, columns_list, schema_list, tables_list, grid_data_folder, "lidar_data.csv")
187/1: %run koskisen_grid_data_creation.py
187/2: %run koskisen_grid_data_creation.py --dir c
187/3: %run koskisen_grid_data_creation.py --dir /home/tman/Work/data/koskisen --output koskisen_grid_with_lidar.csv
187/4: %run koskisen_grid_data_creation.py --dir /home/tman/Work/data/koskisen --output koskisen_grid_with_lidar.csv
187/5: scalar_grid_df.head()
187/6: scalar_grid_df.agg("mean")
187/7: scalar_grid_df.groupby("stand_id")agg("mean")
187/8: scalar_grid_df.groupby("stand_id").agg("mean")
187/9: scalar_grid_df.groupby("stand_id").agg("mean").reset_index()
187/10: scalar_grid_df.columns
187/11: [col for col in scalar_grid_df.columns if "Unnamed" in col]
187/12: grid_and_stand_ids
187/13: grid_and_stand_ids.columns
188/1: %run koskisen_grid_data_creation.py
188/2: %run koskisen_grid_data_creation.py --dir /home/tman/Work/data/koskisen --output koskisen_grid_with_lidar.csv
188/3: scalar_grid_df['stand_id']
188/4: grid_data['stand_id
188/5: grid_data['stand_id'}
188/6: grid_data['stand_id']
188/7: grid_and_stand_ids['stand_id']
188/8: data_frames[0]
188/9: data_frames[1]
188/10: data_frames[0]
188/11: data_frames[2]
188/12: data_frames[3]
188/13: grid_and_stand_ids.shape
189/1:
# Adapted from 'Koskisen Modelling with Bayesian Hyperparameter Optimization.ipynb'
from functools import reduce
from tqdm import tqdm_notebook
import os
import sys
sys.path.append('../../regressors/')

import pandas as pd
import numpy as np

from data import data_loading

%load_ext autoreload
%autoreload 2
%aimport data
189/2:
import GPyOpt
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score

def optimize_xgboost(X_train, y_train_col, max_iter=30, random_state=42):
    domain = [
        {'name': 'learning_rate', 'type': 'continuous', 'domain': (0, 1)},
        {'name': 'gamma', 'type': 'continuous', 'domain': (0, 5)},
        {'name': 'max_depth', 'type': 'discrete', 'domain': (1, 50)},
        {'name': 'n_estimators', 'type': 'discrete', 'domain': (1, 300)},
        {'name': 'min_child_weight', 'type': 'discrete', 'domain': (1, 10)}
    ]

    def f(params):
        params = params[0]
        estimator = XGBRegressor(learning_rate=params[0],
                               gamma=params[1],
                               max_depth=int(params[2]),
                               n_estimators=int(params[3]),
                               min_child_weight=int(params[4])
                                )

        score = -cross_val_score(estimator, X_train, y_train_col, cv=5,
                                scoring='neg_mean_squared_error').mean()

        return np.array(score)

    np.random.seed(random_state)
    optimizer = GPyOpt.methods.BayesianOptimization(f=f, domain=domain, acquisition_type='MPI', 
                                                    num_cores=4, exact_feval=True)


    optimizer.run_optimization(max_iter=max_iter, verbosity=True)
    optimizer.plot_convergence()
    
    print("Best RMSE on CV: {:.2f}".format(np.sqrt(optimizer.Y.min())))
    print("Best NRMSE on CV: {:.2f} %".format(np.sqrt(optimizer.Y.min()) / y_train_col.mean() * 100))
    
    parameter_names = ['learning_rate', 'gamma', 'max_depth', 'n_estimators', 'min_child_weight']
    best_parameters = dict(zip(parameter_names, optimizer.X[optimizer.Y.argmin()]))
    
    best_parameters['max_depth'] = int(best_parameters['max_depth'])
    best_parameters['n_estimators'] = int(best_parameters['n_estimators'])
    best_parameters['min_child_weight'] = int(best_parameters['min_child_weight'])
    
    return optimizer, best_parameters

from sklearn.model_selection import KFold

def get_95_ci(X_train, y_train_col, best_parameters, normalization_mean=None, random_state=42):
    cv_scores = np.concatenate(
        [-cross_val_score(XGBRegressor(**best_parameters), X_train, y_train_col, 
                          cv=KFold(n_splits=5, shuffle=True, random_state=random_state), 
                          n_jobs=1, scoring='neg_mean_squared_error', verbose=1)
        for i in tqdm_notebook(range(10))]
    )

    cv_rmse = np.sqrt(cv_scores)
    mu = cv_rmse.mean()
    
    normalization_mean = y_train_col.mean() if normalization_mean is None else normalization_mean
    mu_nrmse = mu / normalization_mean * 100

    se = cv_rmse.std()

    me = 1.96*se
    me_nrmse = 1.96*se / normalization_mean * 100
    
    rmse_ci = '{:.2f} +/- {:.2f}'.format(mu, me)
    nrmse_ci = '{:.2f} +/- {:.2f}'.format(mu_nrmse, me_nrmse)
    
    print('CV RMSE 95% confidence interval: {}'.format(rmse_ci))
    print('CV NRMSE 95% confidence interval: {}'.format(nrmse_ci))
    
    return {'cv_rmse_ci': rmse_ci, 'cv_nrmse_ci': nrmse_ci}

from sklearn.metrics import mean_squared_error, mean_absolute_error

def get_test_metrics(model, X_test, y_test_col, normalization_mean=None):
    pred = model.predict(X_test)
    mse = mean_squared_error(y_test_col, pred)
    rmse = np.sqrt(mse)
    
    normalization_mean = np.mean(y_test_col) if normalization_mean is None else normalization_mean 
    nrmse = rmse / normalization_mean * 100
    mae = mean_absolute_error(y_test_col, pred)

    print('Test Results: \n')
    print('MSE: {:.2f}\nRMSE: {:.2f}\nNRMSE: {:.2f} %\nMAE: {:.2f}'.format(mse, rmse, nrmse, mae))
    
    return {'test_mse': mse, 
            'test_rmse': rmse, 
            'test_nrmse': nrmse, 
            'test_mae': mae}

def model_target(X_train, y_train, X_test, y_test, target_column, random_state=42, default=False):
    y_train_col, y_test_col = y_train[target_column], y_test[target_column]
    
    print('Optimizing model for {}...'.format(target_column))
    
    optimizer, best_parameters = optimize_xgboost(X_train, y_train_col, random_state=random_state)
    
    print('Training with best hyperparameters found for {}...'.format(target_column))
    if default == False:
        model = XGBRegressor(**best_parameters)
    else:
        model = XGBRegressor()
    model.fit(X_train, y_train_col)
    
    print('Evaluating the model for {}...'.format(target_column))
    cv_metrics = get_95_ci(X_train, y_train_col, best_parameters, random_state=random_state)
    test_metrics = get_test_metrics(model, X_test, y_test_col)
    
    all_results = {**cv_metrics, **test_metrics}
    
    return all_results, best_parameters

def run_experiment(X_train, y_train, X_test, y_test, target_columns, default):
    all_results = []
    best_parameters = []
    for target_column in target_columns:
        target_results, target_best_parameters = model_target(X_train, y_train, X_test, y_test, target_column, default)
        
        all_results.append(target_results)
        best_parameters.append(target_best_parameters)
        
    all_results = pd.DataFrame(all_results, index=target_columns)
    best_parameters = dict(zip(target_columns, best_parameters))
    
    return all_results, best_parameters
189/3: koskisen_folder = '../../../../data/koskisen/'
189/4:
stand_data = pd.read_csv(os.path.join(koskisen_folder, "stand", 'koskisen_stands_aggregated.csv'))
#stand_data = pd.read_csv(os.path.join(koskisen_folder, "stand", 'koskisen_stand_data.csv'))
189/5:
stand_data = stand_data.drop(['harvest_year', 'harvest_start', 'easting', 'northing', 'area_ha', 'unknown_m3_ha', 
                              'check_volume_diff'], axis=1)
stand_data.isna().mean(axis=0)
189/6:
#stand_data = stand_data.drop(['harvest_year', 'harvest_start', 'easting', 'northing', 'area_ha', 'unknown_m3_ha', 
#                              'check_volume_diff'], axis=1)
stand_data.isna().mean(axis=0)
189/7:
#stand_data = stand_data.drop(['harvest_year', 'harvest_start', 'easting', 'northing', 'area_ha', 'unknown_m3_ha', 
#                              'check_volume_diff'], axis=1)
stand_data.shape
189/8:
#stand_data = stand_data.drop(['harvest_year', 'harvest_start', 'easting', 'northing', 'area_ha', 'unknown_m3_ha', 
#                              'check_volume_diff'], axis=1)
stand_data.head()
189/9: koskisen_folder = '../../../../data/koskisen/'
189/10:
#stand_data = pd.read_csv(os.path.join(koskisen_folder, "stand", 'koskisen_stands_aggregated.csv'))
stand_data = pd.read_csv(os.path.join(koskisen_folder, "stand", 'koskisen_stand_data.csv'))
189/11:
stand_data = stand_data.drop(['harvest_year', 'harvest_start', 'easting', 'northing', 'area_ha', 'unknown_m3_ha', 
                              'check_volume_diff'], axis=1)
#stand_data.head()
189/12:
stand_data = stand_data.drop(['harvest_year', 'harvest_start', 'easting', 'northing', 'area_ha', 'unknown_m3_ha', 
                              'check_volume_diff'], axis=1)
stand_data.head()
189/13:
# Drop unneeded columns
stand_data = stand_data.drop(['harvest_year', 'harvest_start', 'easting', 'northing', 'area_ha', 'unknown_m3_ha', 
                              'check_volume_diff'], axis=1)
189/14: stand_data.head()
190/1:
# Adapted from 'Koskisen Modelling with Bayesian Hyperparameter Optimization.ipynb'
from functools import reduce
from tqdm import tqdm_notebook
import os
import sys
sys.path.append('../../regressors/')

import pandas as pd
import numpy as np

from data import data_loading

%load_ext autoreload
%autoreload 2
%aimport data
190/2:
import GPyOpt
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score

def optimize_xgboost(X_train, y_train_col, max_iter=30, random_state=42):
    domain = [
        {'name': 'learning_rate', 'type': 'continuous', 'domain': (0, 1)},
        {'name': 'gamma', 'type': 'continuous', 'domain': (0, 5)},
        {'name': 'max_depth', 'type': 'discrete', 'domain': (1, 50)},
        {'name': 'n_estimators', 'type': 'discrete', 'domain': (1, 300)},
        {'name': 'min_child_weight', 'type': 'discrete', 'domain': (1, 10)}
    ]

    def f(params):
        params = params[0]
        estimator = XGBRegressor(learning_rate=params[0],
                               gamma=params[1],
                               max_depth=int(params[2]),
                               n_estimators=int(params[3]),
                               min_child_weight=int(params[4])
                                )

        score = -cross_val_score(estimator, X_train, y_train_col, cv=5,
                                scoring='neg_mean_squared_error').mean()

        return np.array(score)

    np.random.seed(random_state)
    optimizer = GPyOpt.methods.BayesianOptimization(f=f, domain=domain, acquisition_type='MPI', 
                                                    num_cores=4, exact_feval=True)


    optimizer.run_optimization(max_iter=max_iter, verbosity=True)
    optimizer.plot_convergence()
    
    print("Best RMSE on CV: {:.2f}".format(np.sqrt(optimizer.Y.min())))
    print("Best NRMSE on CV: {:.2f} %".format(np.sqrt(optimizer.Y.min()) / y_train_col.mean() * 100))
    
    parameter_names = ['learning_rate', 'gamma', 'max_depth', 'n_estimators', 'min_child_weight']
    best_parameters = dict(zip(parameter_names, optimizer.X[optimizer.Y.argmin()]))
    
    best_parameters['max_depth'] = int(best_parameters['max_depth'])
    best_parameters['n_estimators'] = int(best_parameters['n_estimators'])
    best_parameters['min_child_weight'] = int(best_parameters['min_child_weight'])
    
    return optimizer, best_parameters

from sklearn.model_selection import KFold

def get_95_ci(X_train, y_train_col, best_parameters, normalization_mean=None, random_state=42):
    cv_scores = np.concatenate(
        [-cross_val_score(XGBRegressor(**best_parameters), X_train, y_train_col, 
                          cv=KFold(n_splits=5, shuffle=True, random_state=random_state), 
                          n_jobs=1, scoring='neg_mean_squared_error', verbose=1)
        for i in tqdm_notebook(range(10))]
    )

    cv_rmse = np.sqrt(cv_scores)
    mu = cv_rmse.mean()
    
    normalization_mean = y_train_col.mean() if normalization_mean is None else normalization_mean
    mu_nrmse = mu / normalization_mean * 100

    se = cv_rmse.std()

    me = 1.96*se
    me_nrmse = 1.96*se / normalization_mean * 100
    
    rmse_ci = '{:.2f} +/- {:.2f}'.format(mu, me)
    nrmse_ci = '{:.2f} +/- {:.2f}'.format(mu_nrmse, me_nrmse)
    
    print('CV RMSE 95% confidence interval: {}'.format(rmse_ci))
    print('CV NRMSE 95% confidence interval: {}'.format(nrmse_ci))
    
    return {'cv_rmse_ci': rmse_ci, 'cv_nrmse_ci': nrmse_ci}

from sklearn.metrics import mean_squared_error, mean_absolute_error

def get_test_metrics(model, X_test, y_test_col, normalization_mean=None):
    pred = model.predict(X_test)
    mse = mean_squared_error(y_test_col, pred)
    rmse = np.sqrt(mse)
    
    normalization_mean = np.mean(y_test_col) if normalization_mean is None else normalization_mean 
    nrmse = rmse / normalization_mean * 100
    mae = mean_absolute_error(y_test_col, pred)

    print('Test Results: \n')
    print('MSE: {:.2f}\nRMSE: {:.2f}\nNRMSE: {:.2f} %\nMAE: {:.2f}'.format(mse, rmse, nrmse, mae))
    
    return {'test_mse': mse, 
            'test_rmse': rmse, 
            'test_nrmse': nrmse, 
            'test_mae': mae}

def model_target(X_train, y_train, X_test, y_test, target_column, random_state=42, default=False):
    y_train_col, y_test_col = y_train[target_column], y_test[target_column]
    
    print('Optimizing model for {}...'.format(target_column))
    
    optimizer, best_parameters = optimize_xgboost(X_train, y_train_col, random_state=random_state)
    
    print('Training with best hyperparameters found for {}...'.format(target_column))
    if default == False:
        model = XGBRegressor(**best_parameters)
    else:
        model = XGBRegressor()
    model.fit(X_train, y_train_col)
    
    print('Evaluating the model for {}...'.format(target_column))
    cv_metrics = get_95_ci(X_train, y_train_col, best_parameters, random_state=random_state)
    test_metrics = get_test_metrics(model, X_test, y_test_col)
    
    all_results = {**cv_metrics, **test_metrics}
    
    return all_results, best_parameters

def run_experiment(X_train, y_train, X_test, y_test, target_columns, default):
    all_results = []
    best_parameters = []
    for target_column in target_columns:
        target_results, target_best_parameters = model_target(X_train, y_train, X_test, y_test, target_column, default)
        
        all_results.append(target_results)
        best_parameters.append(target_best_parameters)
        
    all_results = pd.DataFrame(all_results, index=target_columns)
    best_parameters = dict(zip(target_columns, best_parameters))
    
    return all_results, best_parameters
190/3: koskisen_folder = '../../../../data/koskisen/'
190/4:
#stand_data = pd.read_csv(os.path.join(koskisen_folder, "stand", 'koskisen_stands_aggregated.csv'))
stand_data = pd.read_csv(os.path.join(koskisen_folder, "stand", 'koskisen_stand_data.csv'))
190/5:
# Drop unneeded columns
stand_data = stand_data.drop(['harvest_year', 'harvest_start', 'easting', 'northing', 'area_ha', 'unknown_m3_ha', 
                              'check_volume_diff'], axis=1)
190/6: stand_data.head()
190/7: grid_data.head()
190/8:
grid_data = pd.read_csv(os.path.join(koskisen_folder, "grid", 'koskisen_stands_aggregated.csv'))
stand_data_aggregated = 
stand_data = pd.read_csv(os.path.join(koskisen_folder, "stand", 'koskisen_stand_data.csv'))
190/9:
grid_data = pd.read_csv(os.path.join(koskisen_folder, "grid", 'koskisen_grid_data.csv'))
#stand_data_aggregated = 
stand_data = pd.read_csv(os.path.join(koskisen_folder, "stand", 'koskisen_stand_data.csv'))
190/10: grid_data.head()
190/11: grid_data.columns
190/12:
stand_data_aggregated = grid_data.groupby('prd_id').agg('mean')
stand_data_aggregated.head()
190/13:
stand_data_aggregated = grid_data.groupby('prd_id').agg('mean').reset_index()
stand_data_aggregated.head()
190/14:
stand_data_aggregated = grid_data.groupby('prd_id').agg('mean').reset_index()
stand_data_aggregated..shape
190/15:
stand_data_aggregated = grid_data.groupby('prd_id').agg('mean').reset_index()
stand_data_aggregated.shape
190/16:
stand_data_aggregated = grid_data.groupby('prd_id').agg('mean').reset_index()
stand_data_aggregated['lidared_before_harvest']
190/17: stand_data_aggregated.columns
190/18:
col_aggregations = {k: 'mode' if k in ['copernicus_leaf_type'] else k: 'mean' for k in stand_data_aggregated.columns}
stand_data_aggregated = grid_data.groupby('prd_id').agg(col_aggregations).reset_index()
stand_data_aggregated = stand_data_aggregated.drop(['plot_id', 'lon', 'lat']
190/19:
col_aggregations = {k: 'mode' if k in ['copernicus_leaf_type'] else k:'mean' for k in stand_data_aggregated.columns}
stand_data_aggregated = grid_data.groupby('prd_id').agg(col_aggregations).reset_index()
stand_data_aggregated = stand_data_aggregated.drop(['plot_id', 'lon', 'lat']
190/20:
col_aggregations = {k: 'mode' if k in {'copernicus_leaf_type'} else k: 'mean' for k in stand_data_aggregated.columns}
stand_data_aggregated = grid_data.groupby('prd_id').agg(col_aggregations).reset_index()
stand_data_aggregated = stand_data_aggregated.drop(['plot_id', 'lon', 'lat']
190/21:
col_aggregations = {k:'mode' if k in {'copernicus_leaf_type'} else k:'mean' for k in stand_data_aggregated.columns}
stand_data_aggregated = grid_data.groupby('prd_id').agg(col_aggregations).reset_index()
stand_data_aggregated = stand_data_aggregated.drop(['plot_id', 'lon', 'lat']
190/22:
col_aggregations = {k:str('mode' if k in {'copernicus_leaf_type'} else k:str('mean') for k in stand_data_aggregated.columns}
stand_data_aggregated = grid_data.groupby('prd_id').agg(col_aggregations).reset_index()
stand_data_aggregated = stand_data_aggregated.drop(['plot_id', 'lon', 'lat']
190/23:
col_aggregations = {k:"mode" if k in {'copernicus_leaf_type'} else k:"mean" for k in stand_data_aggregated.columns}
stand_data_aggregated = grid_data.groupby('prd_id').agg(col_aggregations).reset_index()
stand_data_aggregated = stand_data_aggregated.drop(['plot_id', 'lon', 'lat']
190/24:
col_aggregations = {k:"mode" if k in {'copernicus_leaf_type'} else: k:"mean" for k in stand_data_aggregated.columns}
stand_data_aggregated = grid_data.groupby('prd_id').agg(col_aggregations).reset_index()
stand_data_aggregated = stand_data_aggregated.drop(['plot_id', 'lon', 'lat']
190/25:
col_aggregations = {k:("mode") if k in {'copernicus_leaf_type'} else k:("mean") for k in stand_data_aggregated.columns}
stand_data_aggregated = grid_data.groupby('prd_id').agg(col_aggregations).reset_index()
stand_data_aggregated = stand_data_aggregated.drop(['plot_id', 'lon', 'lat']
190/26:
col_aggregations = {k:'mode' if k in {'copernicus_leaf_type'} for k in stand_data_aggregated.columns}
stand_data_aggregated = grid_data.groupby('prd_id').agg(col_aggregations).reset_index()
stand_data_aggregated = stand_data_aggregated.drop(['plot_id', 'lon', 'lat']
190/27:
col_aggregations = {k:'mode' if k in ['copernicus_leaf_type'] for k in stand_data_aggregated.columns}
stand_data_aggregated = grid_data.groupby('prd_id').agg(col_aggregations).reset_index()
stand_data_aggregated = stand_data_aggregated.drop(['plot_id', 'lon', 'lat']
190/28:
col_aggregations = {k:'mode' if k in ['copernicus_leaf_type'] for k in stand_data_aggregated.columns}
stand_data_aggregated = grid_data.groupby('prd_id').agg(col_aggregations).reset_index()
stand_data_aggregated = stand_data_aggregated.drop(['plot_id', 'lon', 'lat']
190/29:
col_aggregations = {k:'mode' if k in ['copernicus_leaf_type'] else k for k in stand_data_aggregated.columns}
stand_data_aggregated = grid_data.groupby('prd_id').agg(col_aggregations).reset_index()
stand_data_aggregated = stand_data_aggregated.drop(['plot_id', 'lon', 'lat']
190/30:
col_aggregations = {k:'mode' if k in ['copernicus_leaf_type'] else k for k in stand_data_aggregated.columns}
print(col_aggregations)
stand_data_aggregated = grid_data.groupby('prd_id').agg(col_aggregations).reset_index()
stand_data_aggregated = stand_data_aggregated.drop(['plot_id', 'lon', 'lat']
190/31:
col_aggregations = {col:'mode' if col in ['copernicus_leaf_type'] else col for col in stand_data_aggregated.columns}
print(col_aggregations)
stand_data_aggregated = grid_data.groupby('prd_id').agg(col_aggregations).reset_index()
stand_data_aggregated = stand_data_aggregated.drop(['plot_id', 'lon', 'lat']
190/32:
col_aggregations = {col:'mode' if col in ['copernicus_leaf_type'] else col for col in stand_data_aggregated.columns}
print(col_aggregations)
stand_data_aggregated = grid_data.groupby('prd_id').agg(col_aggregations).reset_index()
stand_data_aggregated = stand_data_aggregated.drop(['plot_id', 'lon', 'lat'])
190/33:
col_aggregations = {col:'mode' if col in ['copernicus_leaf_type'] else col:'mean' for col in stand_data_aggregated.columns}
print(col_aggregations)
stand_data_aggregated = grid_data.groupby('prd_id').agg(col_aggregations).reset_index()
stand_data_aggregated = stand_data_aggregated.drop(['plot_id', 'lon', 'lat'])
190/34:
col_aggregations = {'copernicus_leaf_type':'mode'}
[col_aggregations[col] = 'mean' for col in grid_data.columns]
print(col_aggregations)
stand_data_aggregated = grid_data.groupby('prd_id').agg(col_aggregations).reset_index()
stand_data_aggregated = stand_data_aggregated.drop(['plot_id', 'lon', 'lat'])
190/35:
col_aggregations = {'copernicus_leaf_type':'mode'}
for col in grid_data.columns:
    col_aggregations[col] = 'mean'
print(col_aggregations)
stand_data_aggregated = grid_data.groupby('prd_id').agg(col_aggregations).reset_index()
stand_data_aggregated = stand_data_aggregated.drop(['plot_id', 'lon', 'lat'])
190/36:
col_aggregations = {}
for col in grid_data.columns:
    col_aggregations[col] = 'mean'

col_aggregations['copernicus_leaf_type'] = 'mode'
print(col_aggregations)
stand_data_aggregated = grid_data.groupby('prd_id').agg(col_aggregations).reset_index()
stand_data_aggregated = stand_data_aggregated.drop(['plot_id', 'lon', 'lat'])
190/37:
col_aggregations = {}
for col in grid_data.columns:
    col_aggregations[col] = 'mean'

col_aggregations['copernicus_leaf_type'] = 'mode'
stand_data_aggregated = grid_data.groupby('prd_id').agg(col_aggregations).reset_index()
stand_data_aggregated = stand_data_aggregated.drop(['plot_id', 'lon', 'lat'])
190/38: grid_data.columns
190/39: grid_data.columns
190/40:
col_aggregations = {}
for col in grid_data.columns:
    col_aggregations[col] = 'mean'

col_aggregations['copernicus_leaf_type'] = 'mode'
stand_data_aggregated = grid_data.groupby('prd_id').agg(col_aggregations).reset_index()
stand_data_aggregated = stand_data_aggregated.drop(['plot_id', 'lon', 'lat'])
190/41:
col_aggregations['copernicus_leaf_type'] = 'mode'
stand_data_aggregated = grid_data.groupby('prd_id').agg('mean').reset_index()
stand_data_aggregated = stand_data_aggregated.drop(['plot_id', 'lon', 'lat'])
190/42:
col_aggregations['copernicus_leaf_type'] = 'mode'
stand_data_aggregated = grid_data.groupby('prd_id').agg('mean').reset_index()
stand_data_aggregated.head()
190/43: stand_data_aggregated.head()
190/44: stand_data_aggregated.isna().mean(axis=0)
190/45:
features = stand_data_aggregated.dropna()

assert features.isna().sum().sum() == 0
190/46:
# Drop rows where lidar was done after harvesting and the column after filtering
features = features[features['lidared_before_harvest']].drop('lidared_before_harvest', axis=1)

target_columns = ['total_volume_ha', 'pine_volume_ha', 'spruce_volume_ha', 'deciduous_volume_ha']
non_feature_columns = ['prd_id']

X = features.drop(target_columns, axis=1)
y = features[target_columns]
192/1:
# Adapted from 'Koskisen Modelling with Bayesian Hyperparameter Optimization.ipynb'
from functools import reduce
from tqdm import tqdm_notebook
import os
import sys
sys.path.append('../../regressors/')

import pandas as pd
import numpy as np

from data import data_loading

%load_ext autoreload
%autoreload 2
%aimport data
192/2:
import GPyOpt
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score

def optimize_xgboost(X_train, y_train_col, max_iter=30, random_state=42):
    domain = [
        {'name': 'learning_rate', 'type': 'continuous', 'domain': (0, 1)},
        {'name': 'gamma', 'type': 'continuous', 'domain': (0, 5)},
        {'name': 'max_depth', 'type': 'discrete', 'domain': (1, 50)},
        {'name': 'n_estimators', 'type': 'discrete', 'domain': (1, 300)},
        {'name': 'min_child_weight', 'type': 'discrete', 'domain': (1, 10)}
    ]

    def f(params):
        params = params[0]
        estimator = XGBRegressor(learning_rate=params[0],
                               gamma=params[1],
                               max_depth=int(params[2]),
                               n_estimators=int(params[3]),
                               min_child_weight=int(params[4])
                                )

        score = -cross_val_score(estimator, X_train, y_train_col, cv=5,
                                scoring='neg_mean_squared_error').mean()

        return np.array(score)

    np.random.seed(random_state)
    optimizer = GPyOpt.methods.BayesianOptimization(f=f, domain=domain, acquisition_type='MPI', 
                                                    num_cores=4, exact_feval=True)


    optimizer.run_optimization(max_iter=max_iter, verbosity=True)
    optimizer.plot_convergence()
    
    print("Best RMSE on CV: {:.2f}".format(np.sqrt(optimizer.Y.min())))
    print("Best NRMSE on CV: {:.2f} %".format(np.sqrt(optimizer.Y.min()) / y_train_col.mean() * 100))
    
    parameter_names = ['learning_rate', 'gamma', 'max_depth', 'n_estimators', 'min_child_weight']
    best_parameters = dict(zip(parameter_names, optimizer.X[optimizer.Y.argmin()]))
    
    best_parameters['max_depth'] = int(best_parameters['max_depth'])
    best_parameters['n_estimators'] = int(best_parameters['n_estimators'])
    best_parameters['min_child_weight'] = int(best_parameters['min_child_weight'])
    
    return optimizer, best_parameters

from sklearn.model_selection import KFold

def get_95_ci(X_train, y_train_col, best_parameters, normalization_mean=None, random_state=42):
    cv_scores = np.concatenate(
        [-cross_val_score(XGBRegressor(**best_parameters), X_train, y_train_col, 
                          cv=KFold(n_splits=5, shuffle=True, random_state=random_state), 
                          n_jobs=1, scoring='neg_mean_squared_error', verbose=1)
        for i in tqdm_notebook(range(10))]
    )

    cv_rmse = np.sqrt(cv_scores)
    mu = cv_rmse.mean()
    
    normalization_mean = y_train_col.mean() if normalization_mean is None else normalization_mean
    mu_nrmse = mu / normalization_mean * 100

    se = cv_rmse.std()

    me = 1.96*se
    me_nrmse = 1.96*se / normalization_mean * 100
    
    rmse_ci = '{:.2f} +/- {:.2f}'.format(mu, me)
    nrmse_ci = '{:.2f} +/- {:.2f}'.format(mu_nrmse, me_nrmse)
    
    print('CV RMSE 95% confidence interval: {}'.format(rmse_ci))
    print('CV NRMSE 95% confidence interval: {}'.format(nrmse_ci))
    
    return {'cv_rmse_ci': rmse_ci, 'cv_nrmse_ci': nrmse_ci}

from sklearn.metrics import mean_squared_error, mean_absolute_error

def get_test_metrics(model, X_test, y_test_col, normalization_mean=None):
    pred = model.predict(X_test)
    mse = mean_squared_error(y_test_col, pred)
    rmse = np.sqrt(mse)
    
    normalization_mean = np.mean(y_test_col) if normalization_mean is None else normalization_mean 
    nrmse = rmse / normalization_mean * 100
    mae = mean_absolute_error(y_test_col, pred)

    print('Test Results: \n')
    print('MSE: {:.2f}\nRMSE: {:.2f}\nNRMSE: {:.2f} %\nMAE: {:.2f}'.format(mse, rmse, nrmse, mae))
    
    return {'test_mse': mse, 
            'test_rmse': rmse, 
            'test_nrmse': nrmse, 
            'test_mae': mae}

def model_target(X_train, y_train, X_test, y_test, target_column, random_state=42, default=False):
    y_train_col, y_test_col = y_train[target_column], y_test[target_column]
    
    print('Optimizing model for {}...'.format(target_column))
    
    optimizer, best_parameters = optimize_xgboost(X_train, y_train_col, random_state=random_state)
    
    print('Training with best hyperparameters found for {}...'.format(target_column))
    if default == False:
        model = XGBRegressor(**best_parameters)
    else:
        model = XGBRegressor()
    model.fit(X_train, y_train_col)
    
    print('Evaluating the model for {}...'.format(target_column))
    cv_metrics = get_95_ci(X_train, y_train_col, best_parameters, random_state=random_state)
    test_metrics = get_test_metrics(model, X_test, y_test_col)
    
    all_results = {**cv_metrics, **test_metrics}
    
    return all_results, best_parameters

def run_experiment(X_train, y_train, X_test, y_test, target_columns, default):
    all_results = []
    best_parameters = []
    for target_column in target_columns:
        target_results, target_best_parameters = model_target(X_train, y_train, X_test, y_test, target_column, default)
        
        all_results.append(target_results)
        best_parameters.append(target_best_parameters)
        
    all_results = pd.DataFrame(all_results, index=target_columns)
    best_parameters = dict(zip(target_columns, best_parameters))
    
    return all_results, best_parameters
192/3: koskisen_folder = '../../../../data/koskisen/'
192/4:
grid_data = pd.read_csv(os.path.join(koskisen_folder, "grid", 'koskisen_grid_data.csv'))
#stand_data_aggregated = 
stand_data = pd.read_csv(os.path.join(koskisen_folder, "stand", 'koskisen_stand_data.csv'))
192/5: grid_data.columns
192/6:
col_aggregations['copernicus_leaf_type'] = 'mode'
stand_data_aggregated = grid_data.groupby('prd_id').agg('mean').reset_index()
stand_data_aggregated.head()
192/7:
stand_data_aggregated = grid_data.groupby('prd_id').agg('mean').reset_index()
stand_data_aggregated.head()
192/8: stand_data_aggregated.isna().mean(axis=0)
192/9:
features = stand_data_aggregated.dropna()

assert features.isna().sum().sum() == 0
192/10:
# Drop rows where lidar was done after harvesting and the column after filtering
features = features[features['lidared_before_harvest']].drop('lidared_before_harvest', axis=1)

target_columns = ['total_volume_ha', 'pine_volume_ha', 'spruce_volume_ha', 'deciduous_volume_ha']
non_feature_columns = ['prd_id']

X = features.drop(target_columns, axis=1)
y = features[target_columns]
192/11:
X_train, X_test = data_loading.split_from_ids(X, split_name='koskisen', id_column='prd_id')
X_train, X_test = X_train.drop("prd_id", axis=1), X_test.drop("prd_id", axis=1)
y_train, y_test = y.loc[X_train.index, :], y.loc[X_test.index, :]
192/12: X_train.head()
192/13:
drop_cols = ["prd_id", "plot_id", "lon", "lat"]
X_train, X_test = data_loading.split_from_ids(X, split_name='koskisen', id_column='prd_id')
X_train, X_test = X_train.drop(drop_cols, axis=1), X_test.drop(drop_cols, axis=1)
y_train, y_test = y.loc[X_train.index, :], y.loc[X_test.index, :]
192/14: X_train.head()
192/15:
assert X_train.shape[0] == y_train.shape[0]
assert (X_train.index == y_train.index).all()
192/16:
all_results, all_best_parameters = run_experiment(X_train, y_train, X_test, y_test, target_columns, default=False)
all_results
194/1:
%load_ext autoreload
%autoreload 2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import json
import os
from tqdm import tqdm
import re
import sys
from sklearn.model_selection import train_test_split
import requests
pd.options.display.float_format = '{:,.2f}'.format
# Add path to where utils.py is so metrics can be imported
sys.path.insert(0, r'../../regressors')
from data.data_loading import import_data, GeoAPI, split_from_ids
from data import data_loading
194/2:
stand_data = pd.read_csv("~/Work/data/koskisen/v_stand_level_features.csv")
gridcell_data = pd.read_csv("~/Work/data/koskisen/v_gridcell_volumes_with_coords.csv")
gridcell_data = gridcell_data.drop('hila_polygon', axis=1)
194/3:
stand_data = pd.read_csv("~/Work/data/koskisen/stand/v_stand_level_features.csv")
gridcell_data = pd.read_csv("~/Work/data/koskisen/grid/v_gridcell_volumes_with_coords.csv")
gridcell_data = gridcell_data.drop('hila_polygon', axis=1)
194/4:
stand_data = pd.read_csv("~/Work/data/koskisen/rawdata/v_stand_level_features.csv")
gridcell_data = pd.read_csv("~/Work/data/koskisen/rawdata/v_gridcell_volumes_with_coords.csv")
gridcell_data = gridcell_data.drop('hila_polygon', axis=1)
194/5:
columns_from_stand = ['prd_id', 'harvest_year', 'harvest_start']
koskisen_grids = gridcell_data.merge(stand_data[columns_from_stand], left_on="koski_prd_id", right_on="prd_id")
koskisen_grids['harvest_start'] = pd.to_datetime(koskisen_grids['harvest_start'])
194/6:
columns_from_stand = ['prd_id', 'harvest_year', 'harvest_start']
koskisen_grids = gridcell_data.merge(stand_data[columns_from_stand], left_on="koski_prd_id", right_on="prd_id")
koskisen_grids['harvest_start'] = pd.to_datetime(koskisen_grids['harvest_start'])
194/7:
stand_data = pd.read_csv("~/Work/data/koskisen/rawdata/v_stand_level_features.csv")
gridcell_data = pd.read_csv("~/Work/data/koskisen/rawdata/v_gridcell_volumes_with_coords.csv")
gridcell_data = gridcell_data.drop('hila_polygon', axis=1)
194/8:
columns_from_stand = ['prd_id', 'harvest_year', 'harvest_start']
koskisen_grids = gridcell_data.merge(stand_data[columns_from_stand], left_on="koski_prd_id", right_on="prd_id")
koskisen_grids['harvest_start'] = pd.to_datetime(koskisen_grids['harvest_start'])
194/9:
api = GeoAPI(default_locations=gridcell_data[['easting', 'northing']].values.tolist(),
                default_srid=3067, default_plot_ids=gridcell_data.hila_gridcellid.values.tolist())

def get_metsakeskus_data():
    columns_list = [["volumepine","volumespruce","volumedeciduous","volume","creationtime", "updatetime",
                     "soiltype","fertilityclass","laserheight","laserdensity"]]

    schema_list = ['metsakeskus_hila']
    tables_list = ['gridcell']

    data = api.request_data(schema_list, tables_list, columns_list, batch_size=2000)

    # Return plot_ids from index to a column.
    data.reset_index(inplace=True)
    data = data.drop_duplicates(subset='plot_id')
    
    return data

metsakeskus_data = get_metsakeskus_data()
194/10:
metsakeskus_data['creationtime'] = pd.to_datetime(metsakeskus_data['creationtime'])
metsakeskus_data['updatetime'] = pd.to_datetime(metsakeskus_data['updatetime'])
194/11:
# GeoAPI adds plot_id to corresponding rows when fetching data. We used hila_gridcellid when fetching data
full_data = koskisen_grids.merge(metsakeskus_data, left_on="hila_gridcellid", right_on="plot_id")
194/12:
# Remember same volume order in both
metsakeskus_pred_columns = ['volume', 'volumepine', 'volumespruce', 'volumedeciduous']
koskisen_vol_columns = ['total_volume_ha', 'pine_volume_ha', 'spruce_volume_ha', 'deciduous_volume_ha']
time_columns = ['updatetime', 'harvest_start']

# aggregate volume means and times. Take minimum of both times so we can compare and get just the stands where preds were made
# before harvest
stat_dict = {col: "mean" for col in (metsakeskus_pred_columns + koskisen_vol_columns)}
for col in time_columns: stat_dict[col] = 'min'
# get the means of the volumes per stand and minimum of each harvest_start and updatetime per stand.
# OK to take the mean of koskisen gridcells as ground truth as they're all the same anyway
volume_means_times = full_data.groupby("koski_prd_id")[metsakeskus_pred_columns + koskisen_vol_columns + time_columns].agg(stat_dict)

def calculate_metsakeskus_error(df):
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    
    metsakeskus_preds = df[metsakeskus_pred_columns]
    koskisen_vols = df[koskisen_vol_columns]
    koskisen_means = np.mean(koskisen_vols, axis=0).values
    rmse = np.sqrt(mean_squared_error(metsakeskus_preds, koskisen_vols, multioutput='raw_values'))
    nrmse = (rmse / koskisen_means)*100
    print("Order: total, pine, spruce, deciduous")
    print("Groundtruth means:")
    print(koskisen_means)
    print("RMSE:")
    print(rmse)
    print("NRMSE (RMSE divided by the mean of respective species):")
    print(nrmse)
    

print("Metsakeskus, all stands:")
#calculate_metsakeskus_error(volume_means_times)
print("\nMetsakeskus, on stands where all gridcell preds were made before harvest:")
#calculate_metsakeskus_error(volume_means_times[volume_means_times['updatetime'] < volume_means_times['harvest_start']])
194/13: volume_means_times[volume_means_times['updatetime'] < volume_means_times['harvest_start']].shape
194/14:
# Remember same volume order in both
metsakeskus_pred_columns = ['volume', 'volumepine', 'volumespruce', 'volumedeciduous']
koskisen_vol_columns = ['total_volume_ha', 'pine_volume_ha', 'spruce_volume_ha', 'deciduous_volume_ha']
time_columns = ['updatetime', 'harvest_start']

# aggregate volume means and times. Take minimum of both times so we can compare and get just the stands where preds were made
# before harvest
stat_dict = {col: "mean" for col in (metsakeskus_pred_columns + koskisen_vol_columns)}
for col in time_columns: stat_dict[col] = 'max'
# get the means of the volumes per stand and minimum of each harvest_start and updatetime per stand.
# OK to take the mean of koskisen gridcells as ground truth as they're all the same anyway
volume_means_times = full_data.groupby("koski_prd_id")[metsakeskus_pred_columns + koskisen_vol_columns + time_columns].agg(stat_dict)

def calculate_metsakeskus_error(df):
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    
    metsakeskus_preds = df[metsakeskus_pred_columns]
    koskisen_vols = df[koskisen_vol_columns]
    koskisen_means = np.mean(koskisen_vols, axis=0).values
    rmse = np.sqrt(mean_squared_error(metsakeskus_preds, koskisen_vols, multioutput='raw_values'))
    nrmse = (rmse / koskisen_means)*100
    print("Order: total, pine, spruce, deciduous")
    print("Groundtruth means:")
    print(koskisen_means)
    print("RMSE:")
    print(rmse)
    print("NRMSE (RMSE divided by the mean of respective species):")
    print(nrmse)
    

print("Metsakeskus, all stands:")
#calculate_metsakeskus_error(volume_means_times)
print("\nMetsakeskus, on stands where all gridcell preds were made before harvest:")
#calculate_metsakeskus_error(volume_means_times[volume_means_times['updatetime'] < volume_means_times['harvest_start']])
194/15: volume_means_times[volume_means_times['updatetime'] < volume_means_times['harvest_start']].shape
194/16:
# Remember same volume order in both
metsakeskus_pred_columns = ['volume', 'volumepine', 'volumespruce', 'volumedeciduous']
koskisen_vol_columns = ['total_volume_ha', 'pine_volume_ha', 'spruce_volume_ha', 'deciduous_volume_ha']
time_columns = ['updatetime', 'harvest_start']

# aggregate volume means and times. Take minimum of both times so we can compare and get just the stands where preds were made
# before harvest
stat_dict = {col: "mean" for col in (metsakeskus_pred_columns + koskisen_vol_columns)}
for col in time_columns: stat_dict[col] = 'max'
# get the means of the volumes per stand and minimum of each harvest_start and updatetime per stand.
# OK to take the mean of koskisen gridcells as ground truth as they're all the same anyway
volume_means_times = full_data.groupby("koski_prd_id")[metsakeskus_pred_columns + koskisen_vol_columns + time_columns].agg(stat_dict)

def calculate_metsakeskus_error(df):
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    
    metsakeskus_preds = df[metsakeskus_pred_columns]
    koskisen_vols = df[koskisen_vol_columns]
    koskisen_means = np.mean(koskisen_vols, axis=0).values
    rmse = np.sqrt(mean_squared_error(metsakeskus_preds, koskisen_vols, multioutput='raw_values'))
    nrmse = (rmse / koskisen_means)*100
    print("Order: total, pine, spruce, deciduous")
    print("Groundtruth means:")
    print(koskisen_means)
    print("RMSE:")
    print(rmse)
    print("NRMSE (RMSE divided by the mean of respective species):")
    print(nrmse)
    

print("Metsakeskus, all stands:")
calculate_metsakeskus_error(volume_means_times)
print("\nMetsakeskus, on stands where all gridcell preds were made before harvest:")
calculate_metsakeskus_error(volume_means_times[volume_means_times['updatetime'] < volume_means_times['harvest_start']])
194/17:
# Remember same volume order in both
metsakeskus_pred_columns = ['volume', 'volumepine', 'volumespruce', 'volumedeciduous']
koskisen_vol_columns = ['total_volume_ha', 'pine_volume_ha', 'spruce_volume_ha', 'deciduous_volume_ha']
time_columns = ['updatetime', 'harvest_start']

# aggregate volume means and times. Take minimum of both times so we can compare and get just the stands where preds were made
# before harvest
stat_dict = {col: "mean" for col in (metsakeskus_pred_columns + koskisen_vol_columns)}
stat_dict['harvest_start'] = 'min'
stat_dict['updatetime'] = 'max'
# get the means of the volumes per stand and minimum of each harvest_start and updatetime per stand.
# OK to take the mean of koskisen gridcells as ground truth as they're all the same anyway
volume_means_times = full_data.groupby("koski_prd_id")[metsakeskus_pred_columns + koskisen_vol_columns + time_columns].agg(stat_dict)

def calculate_metsakeskus_error(df):
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    
    metsakeskus_preds = df[metsakeskus_pred_columns]
    koskisen_vols = df[koskisen_vol_columns]
    koskisen_means = np.mean(koskisen_vols, axis=0).values
    rmse = np.sqrt(mean_squared_error(metsakeskus_preds, koskisen_vols, multioutput='raw_values'))
    nrmse = (rmse / koskisen_means)*100
    print("Order: total, pine, spruce, deciduous")
    print("Groundtruth means:")
    print(koskisen_means)
    print("RMSE:")
    print(rmse)
    print("NRMSE (RMSE divided by the mean of respective species):")
    print(nrmse)
    

print("Metsakeskus, all stands:")
calculate_metsakeskus_error(volume_means_times)
print("\nMetsakeskus, on stands where all gridcell preds were made before harvest:")
calculate_metsakeskus_error(volume_means_times[volume_means_times['updatetime'] < volume_means_times['harvest_start']])
194/18: volume_means_times[volume_means_times['updatetime'] < volume_means_times['harvest_start']].shape
194/19:
# Remember same volume order in both
metsakeskus_pred_columns = ['volume', 'volumepine', 'volumespruce', 'volumedeciduous']
koskisen_vol_columns = ['total_volume_ha', 'pine_volume_ha', 'spruce_volume_ha', 'deciduous_volume_ha']
time_columns = ['updatetime', 'harvest_start']

# aggregate volume means and times. Take minimum of both times so we can compare and get just the stands where preds were made
# before harvest
stat_dict = {col: "mean" for col in (metsakeskus_pred_columns + koskisen_vol_columns)}
stat_dict['harvest_start'] = 'min'
stat_dict['updatetime'] = 'max'
# get the means of the volumes per stand and minimum of each harvest_start and updatetime per stand.
# OK to take the mean of koskisen gridcells as ground truth as they're all the same anyway
volume_means_times = full_data.groupby("koski_prd_id")[metsakeskus_pred_columns + koskisen_vol_columns + time_columns].agg(stat_dict)

def calculate_metsakeskus_error(df):
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    
    metsakeskus_preds = df[metsakeskus_pred_columns]
    koskisen_vols = df[koskisen_vol_columns]
    koskisen_means = np.mean(koskisen_vols, axis=0).values
    rmse = np.sqrt(mean_squared_error(metsakeskus_preds, koskisen_vols, multioutput='raw_values'))
    nrmse = (rmse / koskisen_means)*100
    print("Order: total, pine, spruce, deciduous")
    print("Groundtruth means:")
    print(koskisen_means)
    print("RMSE:")
    print(rmse)
    print("NRMSE (RMSE divided by the mean of respective species):")
    print(nrmse)
    

print("Metsakeskus, all stands:")
calculate_metsakeskus_error(volume_means_times)
print("\nMetsakeskus, on stands where all gridcell preds were made before harvest:")
test_set = volume_means_times[volume_means_times['updatetime'] < volume_means_times['harvest_start']].shape
calculate_metsakeskus_error(volume_means_times[volume_means_times['updatetime'] < volume_means_times['harvest_start']])
194/20:
# Remember same volume order in both
metsakeskus_pred_columns = ['volume', 'volumepine', 'volumespruce', 'volumedeciduous']
koskisen_vol_columns = ['total_volume_ha', 'pine_volume_ha', 'spruce_volume_ha', 'deciduous_volume_ha']
time_columns = ['updatetime', 'harvest_start']

# aggregate volume means and times. Take minimum of both times so we can compare and get just the stands where preds were made
# before harvest
stat_dict = {col: "mean" for col in (metsakeskus_pred_columns + koskisen_vol_columns)}
stat_dict['harvest_start'] = 'min'
stat_dict['updatetime'] = 'max'
# get the means of the volumes per stand and minimum of each harvest_start and updatetime per stand.
# OK to take the mean of koskisen gridcells as ground truth as they're all the same anyway
volume_means_times = full_data.groupby("koski_prd_id")[metsakeskus_pred_columns + koskisen_vol_columns + time_columns].agg(stat_dict)

def calculate_metsakeskus_error(df):
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    
    metsakeskus_preds = df[metsakeskus_pred_columns]
    koskisen_vols = df[koskisen_vol_columns]
    koskisen_means = np.mean(koskisen_vols, axis=0).values
    rmse = np.sqrt(mean_squared_error(metsakeskus_preds, koskisen_vols, multioutput='raw_values'))
    nrmse = (rmse / koskisen_means)*100
    print("Order: total, pine, spruce, deciduous")
    print("Groundtruth means:")
    print(koskisen_means)
    print("RMSE:")
    print(rmse)
    print("NRMSE (RMSE divided by the mean of respective species):")
    print(nrmse)
    

print("Metsakeskus, all stands:")
calculate_metsakeskus_error(volume_means_times)
print("\nMetsakeskus, on stands where all gridcell preds were made before harvest:")
test_set = volume_means_times[volume_means_times['updatetime'] < volume_means_times['harvest_start']].shape
calculate_metsakeskus_error(test_set)
194/21:
# Remember same volume order in both
metsakeskus_pred_columns = ['volume', 'volumepine', 'volumespruce', 'volumedeciduous']
koskisen_vol_columns = ['total_volume_ha', 'pine_volume_ha', 'spruce_volume_ha', 'deciduous_volume_ha']
time_columns = ['updatetime', 'harvest_start']

# aggregate volume means and times. Take minimum of both times so we can compare and get just the stands where preds were made
# before harvest
stat_dict = {col: "mean" for col in (metsakeskus_pred_columns + koskisen_vol_columns)}
stat_dict['harvest_start'] = 'min'
stat_dict['updatetime'] = 'max'
# get the means of the volumes per stand and minimum of each harvest_start and updatetime per stand.
# OK to take the mean of koskisen gridcells as ground truth as they're all the same anyway
volume_means_times = full_data.groupby("koski_prd_id")[metsakeskus_pred_columns + koskisen_vol_columns + time_columns].agg(stat_dict)

def calculate_metsakeskus_error(df):
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    
    metsakeskus_preds = df[metsakeskus_pred_columns]
    koskisen_vols = df[koskisen_vol_columns]
    koskisen_means = np.mean(koskisen_vols, axis=0).values
    rmse = np.sqrt(mean_squared_error(metsakeskus_preds, koskisen_vols, multioutput='raw_values'))
    nrmse = (rmse / koskisen_means)*100
    print("Order: total, pine, spruce, deciduous")
    print("Groundtruth means:")
    print(koskisen_means)
    print("RMSE:")
    print(rmse)
    print("NRMSE (RMSE divided by the mean of respective species):")
    print(nrmse)
    

print("Metsakeskus, all stands:")
calculate_metsakeskus_error(volume_means_times)
print("\nMetsakeskus, on stands where all gridcell preds were made before harvest:")
test_set = volume_means_times[volume_means_times['updatetime'] < volume_means_times['harvest_start']]
calculate_metsakeskus_error(test_set)
194/22: test_set.head()
194/23: test_set.shape
193/1:
import sys
import os
sys.path.append('../../regressors')

import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from data import data_loading

%load_ext autoreload
%autoreload 2
%aimport data
193/2:
import sys
import os
sys.path.append('../../regressors')

import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from data import data_loading
from models import models_definition

%load_ext autoreload
%autoreload 2
%aimport data
193/3:
import sys
import os
sys.path.append('../../regressors')

import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from data import data_loading
from models import models_definition

%load_ext autoreload
%autoreload 2
%aimport data
193/4: pip install dill
197/1:
import sys
import os
sys.path.append('../../regressors')

import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from data import data_loading
from models import models_definition

%load_ext autoreload
%autoreload 2
%aimport data
198/1:
import sys
import os
sys.path.append('../../regressors')

import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from data import data_loading
from models import models_definition

%load_ext autoreload
%autoreload 2
%aimport data
194/24: test_set['prd_id']
194/25: test_set.columns
194/26: test_set.reset_index()
194/27: prd_ids = test_set.reset_index()['koski_prd_id']
194/28: prd_ids
194/29: prd_ids.rename('prd_id')
194/30: prd_ids = test_set.reset_index()['koski_prd_id'].rename('prd_id')
194/31: prd_ids.to_csv("/Home/tman/koskisen_testids.csv", index=False)
194/32: prd_ids.to_csv("/home/tman/koskisen_testids.csv", index=False)
194/33: prd_ids = test_set.reset_index()[['koski_prd_id']].rename('prd_id')
194/34: prd_ids = test_set.reset_index()[['koski_prd_id']]
194/35: prd_ids
194/36: prd_ids.rename({'koski_prd_id':'prd_id'})
194/37: prd_ids.rename({'koski_prd_id':'prd_id'}, axis=1)
194/38:
prd_ids = test_set.reset_index()[['koski_prd_id']]
prd_ids.rename({'koski_prd_id':'prd_id'}, axis=1)
194/39:
prd_ids = test_set.reset_index()[['koski_prd_id']]
prd_ids = prd_ids.rename({'koski_prd_id':'prd_id'}, axis=1)
194/40: prd_ids.to_csv("/home/tman/koskisen_testids.csv", index=False)
194/41: prd_ids.shape
194/42: prd_ids.to_csv("/home/tman/Work/linda-forestry-ml/species_prediction/regressors/data/koskisen_testids.csv", index=False)
198/2:
koskisen_folder = "/home/tman/Work/data/koskisen"
#stand_data = pd.read_csv(os.path.join(koskisen_folder, 'v_stand_level_features.csv'))
grid_data = pd.read_csv(os.path.join(koskisen_folder, 'grid', 'koskisen_grid_data.csv'))
198/3:
koskisen_folder = "/home/tman/Work/data/koskisen"
#stand_data = pd.read_csv(os.path.join(koskisen_folder, 'v_stand_level_features.csv'))
grid_data = pd.read_csv(os.path.join(koskisen_folder, 'grid', 'koskisen_grid_data.csv'))

stand_data_aggregated = grid_data.groupby('prd_id').agg('mean').reset_index()
stand_data_aggregated.head()


stand_data_aggregated.isna().mean(axis=0)
features = stand_data_aggregated.dropna()

assert features.isna().sum().sum() == 0

# Drop rows where lidar was done after harvesting and the column after filtering
features = features[features['lidared_before_harvest']].drop('lidared_before_harvest', axis=1)

target_columns = ['total_volume_ha', 'pine_volume_ha', 'spruce_volume_ha', 'deciduous_volume_ha']
non_feature_columns = ['prd_id']

X = features.drop(target_columns, axis=1)
y = features[target_columns]
198/4: features.shape
198/5: stand_data_aggregated.shape
198/6: features.columns
198/7:
koskisen_folder = "/home/tman/Work/data/koskisen"
#stand_data = pd.read_csv(os.path.join(koskisen_folder, 'v_stand_level_features.csv'))
grid_data = pd.read_csv(os.path.join(koskisen_folder, 'grid', 'koskisen_grid_data.csv'))

stand_data_aggregated = grid_data.groupby('prd_id').agg('mean').reset_index()
stand_data_aggregated.head()


stand_data_aggregated.isna().mean(axis=0)
features = stand_data_aggregated.dropna()

assert features.isna().sum().sum() == 0

# Drop rows where lidar was done after harvesting and the column after filtering
features = features[features['lidared_before_harvest']].drop('lidared_before_harvest', axis=1)

target_columns = ['total_volume_ha', 'pine_volume_ha', 'spruce_volume_ha', 'deciduous_volume_ha']
non_feature_columns = ['prd_id']

X = features.drop(target_columns, axis=1)
y = features[target_columns]

drop_cols = ["prd_id", "plot_id", "lon", "lat"]
X_train, X_test = data_loading.split_from_ids(X, split_name='koskisen', id_column='prd_id')
X_train, X_test = X_train.drop(drop_cols, axis=1), X_test.drop(drop_cols, axis=1)
y_train, y_test = y.loc[X_train.index, :], y.loc[X_test.index, :]
198/8: y.shape
198/9:
koskisen_folder = "/home/tman/Work/data/koskisen"
#stand_data = pd.read_csv(os.path.join(koskisen_folder, 'v_stand_level_features.csv'))
grid_data = pd.read_csv(os.path.join(koskisen_folder, 'grid', 'koskisen_grid_data.csv'))

stand_data_aggregated = grid_data.groupby('prd_id').agg('mean').reset_index()
stand_data_aggregated.head()


stand_data_aggregated.isna().mean(axis=0)
features = stand_data_aggregated.dropna()

assert features.isna().sum().sum() == 0

# Drop rows where lidar was done after harvesting and the column after filtering
features = features[features['lidared_before_harvest']].drop('lidared_before_harvest', axis=1)

target_columns = ['pine_volume_ha', 'spruce_volume_ha', 'deciduous_volume_ha']
non_feature_columns = ['prd_id']

X = features.drop(target_columns, axis=1)
y = features[target_columns]
transformed_y = transform_targets(y)

drop_cols = ["prd_id", "plot_id", "lon", "lat"]
X_train, X_test = data_loading.split_from_ids(X, split_name='koskisen', id_column='prd_id')
X_train, X_test = X_train.drop(drop_cols, axis=1), X_test.drop(drop_cols, axis=1)
y_train, y_test = y.loc[X_train.index, :], y.loc[X_test.index, :]
198/10:
koskisen_folder = "/home/tman/Work/data/koskisen"
#stand_data = pd.read_csv(os.path.join(koskisen_folder, 'v_stand_level_features.csv'))
grid_data = pd.read_csv(os.path.join(koskisen_folder, 'grid', 'koskisen_grid_data.csv'))

stand_data_aggregated = grid_data.groupby('prd_id').agg('mean').reset_index()
stand_data_aggregated.head()


stand_data_aggregated.isna().mean(axis=0)
features = stand_data_aggregated.dropna()

assert features.isna().sum().sum() == 0

# Drop rows where lidar was done after harvesting and the column after filtering
features = features[features['lidared_before_harvest']].drop('lidared_before_harvest', axis=1)

target_columns = ['pine_volume_ha', 'spruce_volume_ha', 'deciduous_volume_ha']
non_feature_columns = ['prd_id']

X = features.drop(target_columns, axis=1)
y = features[target_columns]
#transformed_y = transform_targets(y)

drop_cols = ["prd_id", "plot_id", "lon", "lat"]
X_train, X_test = data_loading.split_from_ids(X, split_name='koskisen', id_column='prd_id')
X_train, X_test = X_train.drop(drop_cols, axis=1), X_test.drop(drop_cols, axis=1)
y_train, y_test = y.loc[X_train.index, :], y.loc[X_test.index, :]
198/11: y.shape
198/12: np.mean(y, axis=0).shape
198/13: np.mean(y, axis=1).shape
198/14: np.mean(y, axis=1)[:5]
198/15: np.sum(y, axis=1)[:5]
198/16: y / np.sum(y, axis=1)
198/17: y / np.sum(y, axis=1).values
198/18: y.values / np.sum(y, axis=1).values
198/19: y.values / np.sum(y, axis=1).values[:,np.newaxis]
198/20: (y.values / np.sum(y, axis=1).values[:,np.newaxis]).sum(axis=1)
198/21:
koskisen_folder = "/home/tman/Work/data/koskisen"
#stand_data = pd.read_csv(os.path.join(koskisen_folder, 'v_stand_level_features.csv'))
grid_data = pd.read_csv(os.path.join(koskisen_folder, 'grid', 'koskisen_grid_data.csv'))

stand_data_aggregated = grid_data.groupby('prd_id').agg('mean').reset_index()
stand_data_aggregated.head()


stand_data_aggregated.isna().mean(axis=0)
features = stand_data_aggregated.dropna()

assert features.isna().sum().sum() == 0

# Drop rows where lidar was done after harvesting and the column after filtering
features = features[features['lidared_before_harvest']].drop('lidared_before_harvest', axis=1)

target_columns = ['pine_volume_ha', 'spruce_volume_ha', 'deciduous_volume_ha']
non_feature_columns = ['prd_id']

X = features.drop(target_columns, axis=1)
y = features[target_columns]
transformed_y = transform_targets(y)

drop_cols = ["prd_id", "plot_id", "lon", "lat"]
X_train, X_test = data_loading.split_from_ids(X, split_name='koskisen', id_column='prd_id')
X_train, X_test = X_train.drop(drop_cols, axis=1), X_test.drop(drop_cols, axis=1)
y_train, y_test = y.loc[X_train.index, :], y.loc[X_test.index, :]
198/22:
def transform_targets(targets):
    # Transform from regression targets to relative targets for softmax
    return (targets.values / np.sum(targets, axis=1).values[:,np.newaxis])
198/23:
koskisen_folder = "/home/tman/Work/data/koskisen"
#stand_data = pd.read_csv(os.path.join(koskisen_folder, 'v_stand_level_features.csv'))
grid_data = pd.read_csv(os.path.join(koskisen_folder, 'grid', 'koskisen_grid_data.csv'))

stand_data_aggregated = grid_data.groupby('prd_id').agg('mean').reset_index()
stand_data_aggregated.head()


stand_data_aggregated.isna().mean(axis=0)
features = stand_data_aggregated.dropna()

assert features.isna().sum().sum() == 0

# Drop rows where lidar was done after harvesting and the column after filtering
features = features[features['lidared_before_harvest']].drop('lidared_before_harvest', axis=1)

target_columns = ['pine_volume_ha', 'spruce_volume_ha', 'deciduous_volume_ha']
non_feature_columns = ['prd_id']

X = features.drop(target_columns, axis=1)
y = features[target_columns]
transformed_y = transform_targets(y)

drop_cols = ["prd_id", "plot_id", "lon", "lat"]
X_train, X_test = data_loading.split_from_ids(X, split_name='koskisen', id_column='prd_id')
X_train, X_test = X_train.drop(drop_cols, axis=1), X_test.drop(drop_cols, axis=1)
y_train, y_test = y.loc[X_train.index, :], y.loc[X_test.index, :]
198/24:
dense = models_definition.create_dense(X_train.shape[1], y_train.shape[1], 
                                       n_units=128, n_layers=4, final_activation='softmax')
198/25:
dense = models_definition.create_dense((X_train.shape[1],), y_train.shape[1], 
                                       n_units=128, n_layers=4, final_activation='softmax')
198/26:
koskisen_folder = "/home/tman/Work/data/koskisen"
#stand_data = pd.read_csv(os.path.join(koskisen_folder, 'v_stand_level_features.csv'))
grid_data = pd.read_csv(os.path.join(koskisen_folder, 'grid', 'koskisen_grid_data.csv'))

stand_data_aggregated = grid_data.groupby('prd_id').agg('mean').reset_index()
stand_data_aggregated.head()


stand_data_aggregated.isna().mean(axis=0)
features = stand_data_aggregated.dropna()

assert features.isna().sum().sum() == 0

# Drop rows where lidar was done after harvesting and the column after filtering
features = features[features['lidared_before_harvest']].drop('lidared_before_harvest', axis=1)

target_columns = ['pine_volume_ha', 'spruce_volume_ha', 'deciduous_volume_ha']
non_feature_columns = ['prd_id']

X = features.drop(target_columns, axis=1)
y = features[target_columns]
transformed_y = transform_targets(y)

drop_cols = ["prd_id", "plot_id", "lon", "lat"]
X_train, X_test = data_loading.split_from_ids(X, split_name='koskisen', id_column='prd_id')
X_train, X_test = X_train.drop(drop_cols, axis=1), X_test.drop(drop_cols, axis=1)
y_train, y_test = transformed_y.loc[X_train.index, :], transformed_y.loc[X_test.index, :]
198/27:
koskisen_folder = "/home/tman/Work/data/koskisen"
#stand_data = pd.read_csv(os.path.join(koskisen_folder, 'v_stand_level_features.csv'))
grid_data = pd.read_csv(os.path.join(koskisen_folder, 'grid', 'koskisen_grid_data.csv'))

stand_data_aggregated = grid_data.groupby('prd_id').agg('mean').reset_index()
stand_data_aggregated.head()


stand_data_aggregated.isna().mean(axis=0)
features = stand_data_aggregated.dropna()

assert features.isna().sum().sum() == 0

# Drop rows where lidar was done after harvesting and the column after filtering
features = features[features['lidared_before_harvest']].drop('lidared_before_harvest', axis=1)

target_columns = ['pine_volume_ha', 'spruce_volume_ha', 'deciduous_volume_ha']
non_feature_columns = ['prd_id']

X = features.drop(target_columns, axis=1)
y = features[target_columns]
#transformed_y = transform_targets(y)

drop_cols = ["prd_id", "plot_id", "lon", "lat"]
X_train, X_test = data_loading.split_from_ids(X, split_name='koskisen', id_column='prd_id')
X_train, X_test = X_train.drop(drop_cols, axis=1), X_test.drop(drop_cols, axis=1)
y_train, y_test = y.loc[X_train.index, :], y.loc[X_test.index, :]
y_train, y_test = transform_targets(y_train), transform_targets(y_test)
198/28:
koskisen_folder = "/home/tman/Work/data/koskisen"
#stand_data = pd.read_csv(os.path.join(koskisen_folder, 'v_stand_level_features.csv'))
grid_data = pd.read_csv(os.path.join(koskisen_folder, 'grid', 'koskisen_grid_data.csv'))

stand_data_aggregated = grid_data.groupby('prd_id').agg('mean').reset_index()
stand_data_aggregated.head()


stand_data_aggregated.isna().mean(axis=0)
features = stand_data_aggregated.dropna()

assert features.isna().sum().sum() == 0

# Drop rows where lidar was done after harvesting and the column after filtering
features = features[features['lidared_before_harvest']].drop('lidared_before_harvest', axis=1)

target_columns = ['pine_volume_ha', 'spruce_volume_ha', 'deciduous_volume_ha']
non_feature_columns = ['prd_id']

X = features.drop(target_columns, axis=1)
y = features[target_columns]
#transformed_y = transform_targets(y)

drop_cols = ["prd_id", "plot_id", "lon", "lat"]
X_train, X_test = data_loading.split_from_ids(X, split_name='koskisen', id_column='prd_id')
X_train, X_test = X_train.drop(drop_cols, axis=1), X_test.drop(drop_cols, axis=1)
y_train, y_test = y.loc[X_train.index, :], y.loc[X_test.index, :]
y_train_transformed, y_test_transformed = transform_targets(y_train), transform_targets(y_test)
198/29:
dense = models_definition.create_dense((X_train.shape[1],), y_train_transformed.shape[1], 
                                       n_units=128, n_layers=4, final_activation='softmax')
198/30: dense.fit(X_train, y_train_transformed)
198/31:
dense = models_definition.create_dense((X_train.shape[1],), y_train_transformed.shape[1], 
                                       n_units=128, n_layers=4, final_activation='softmax')
dense.compile(loss='categorical_crossentropy', optimizer='adam')
198/32: dense.fit(X_train, y_train_transformed)
198/33: dense.fit(X_train, y_train_transformed, max_epochs=50)
198/34: dense.fit(X_train, y_train_transformed, epochs=50)
198/35: X_train[:5]
198/36:
dense = models_definition.create_dense((X_train.shape[1],), y_train_transformed.shape[1], 
                                       n_units=128, n_layers=4, final_activation='softmax')
dense.compile(loss='mean_squred_error', optimizer='adam')
198/37:
dense = models_definition.create_dense((X_train.shape[1],), y_train_transformed.shape[1], 
                                       n_units=128, n_layers=4, final_activation='softmax')
dense.compile(loss='mean_squared_error', optimizer='adam')
198/38: X_train[:5]
198/39: dense.fit(X_train, y_train_transformed, epochs=50)
198/40: y_train_transformed[:4)
198/41: y_train_transformed[:4]
198/42:
dense = models_definition.create_dense((X_train.shape[1],), y_train_transformed.shape[1], 
                                       n_units=512, n_layers=4, final_activation='softmax')
dense.compile(loss='mean_squared_error', optimizer='adam')
198/43: dense.fit(X_train, y_train_transformed, epochs=50)
198/44: dense.summary()
198/45:
dense = models_definition.create_dense((X_train.shape[1],), y_train_transformed.shape[1], 
                                       n_units=512, n_layers=4, final_activation='softmhjvjh,ax')
dense.compile(loss='mean_squared_error', optimizer='adam')
198/46:
from keras.optimizers import Adam

dense = models_definition.create_dense((X_train.shape[1],), y_train_transformed.shape[1], 
                                       n_units=512, n_layers=4, final_activation='softmax')
opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
dense.compile(loss='mean_squared_error', optimizer=opt)
198/47: dense.fit(X_train, y_train_transformed, epochs=50)
198/48:
from keras.optimizers import Adam

dense = models_definition.create_dense((X_train.shape[1],), y_train_transformed.shape[1], 
                                       n_units=512, n_layers=4, final_activation='softmax')
opt = Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
dense.compile(loss='mean_squared_error', optimizer=opt)
198/49: dense.fit(X_train, y_train_transformed, epochs=50)
198/50:
from keras.optimizers import Adam

dense = models_definition.create_dense((X_train.shape[1],), y_train_transformed.shape[1], 
                                       n_units=512, n_layers=4, final_activation='softmax')
opt = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
dense.compile(loss='mean_squared_error', optimizer=opt)
198/51: dense.fit(X_train, y_train_transformed, epochs=50)
198/52: dense.fit(X_train, y_train_transformed, epochs=50, batch_size=64)
198/53: dense.summary()
198/54:
from keras.optimizers import Adam

dense = models_definition.create_dense((X_train.shape[1],), y_train_transformed.shape[1], dropout_probability=0.4,
                                       n_units=512, n_layers=4, final_activation='softmax')
opt = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
dense.compile(loss='mean_squared_error', optimizer=opt)
198/55: dense.fit(X_train, y_train_transformed, epochs=50, batch_size=64)
198/56:
from keras.optimizers import Adam

dense = models_definition.create_dense((X_train.shape[1],), y_train_transformed.shape[1], dropout_probability=0.4,
                                       n_units=512, n_layers=4, final_activation='softmax')
opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
dense.compile(loss='mean_squared_error', optimizer=opt)
198/57: dense.fit(X_train, y_train_transformed, epochs=50, batch_size=64)
198/58:
from keras.optimizers import Adam

dense = models_definition.create_dense((X_train.shape[1],), y_train_transformed.shape[1], dropout_probability=0.4,
                                       n_units=512, n_layers=4, final_activation='softmax')
opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
dense.compile(loss='mean_squared_error', optimizer=opt)
198/59: dense.fit(X_train, y_train_transformed, epochs=50, batch_size=64)
198/60:
from keras.optimizers import Adam

dense = models_definition.create_dense((X_train.shape[1],), y_train_transformed.shape[1], dropout_probability=0.4,
                                       n_units=512, n_layers=4, final_activation='softmax')
opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
dense.compile(loss='categorical_crossentropy', optimizer=opt)
198/61: dense.fit(X_train, y_train_transformed, epochs=50, batch_size=64)
198/62:
from keras.optimizers import Adam

dense = models_definition.create_dense((X_train.shape[1],), y_train_transformed.shape[1], dropout_probability=0.4,
                                       n_units=512, n_layers=4, final_activation='softmax')
opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
dense.compile(loss='categorical_crossentropy', optimizer=opt)
198/63: dense.fit(X_train, y_train_transformed, epochs=50, batch_size=64)
198/64:
from keras.optimizers import Adam

dense = models_definition.create_dense((X_train.shape[1],), y_train_transformed.shape[1], dropout_probability=0.4,
                                       n_units=512, n_layers=4, final_activation='softmax')
opt = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
dense.compile(loss='categorical_crossentropy', optimizer=opt)
198/65: dense.fit(X_train, y_train_transformed, epochs=50, batch_size=64)
198/66:
from keras.optimizers import Adam

dense = models_definition.create_dense((X_train.shape[1],), y_train_transformed.shape[1], dropout_probability=0.4,
                                       n_units=512, n_layers=4, final_activation='softmax')
opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
dense.compile(loss='categorical_crossentropy', optimizer=opt)
198/67: dense.fit(X_train, y_train_transformed, epochs=50, batch_size=64)
198/68: dense.fit(X_train, y_train_transformed, epochs=50, batch_size=64)
198/69:
from keras.optimizers import Adam

dense = models_definition.create_dense((X_train.shape[1],), y_train_transformed.shape[1], dropout_probability=0.4,
                                       n_units=512, n_layers=4, final_activation='softmax')
opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
dense.compile(loss='categorical_crossentropy', optimizer=opt)
198/70: dense.fit(X_train, y_train_transformed, epochs=50, batch_size=64)
198/71:
from keras.optimizers import Adam

dense = models_definition.create_dense((X_train.shape[1],), y_train_transformed.shape[1], dropout_probability=0.4,
                                       n_units=512, n_layers=4, final_activation='softmax')
opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
dense.compile(loss='categorical_crossentropy', optimizer=opt)
198/72: dense.fit(X_train, y_train_transformed, epochs=50, batch_size=64)
198/73:
from keras.optimizers import Adam

dense = models_definition.create_dense((X_train.shape[1],), y_train_transformed.shape[1], dropout_probability=0.4,
                                       n_units=512, n_layers=4, final_activation='softmax')
opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
dense.compile(loss='mean_squared_error', optimizer=opt)
198/74: dense.fit(X_train, y_train_transformed, epochs=50, batch_size=64)
198/75:
from keras.optimizers import Adam

dense = models_definition.create_dense((X_train.shape[1],), y_train_transformed.shape[1], dropout_probability=0.4,
                                       n_units=512, n_layers=4, final_activation='softmax')
opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
dense.compile(loss='mean_squared_error', optimizer=opt)
198/76: dense.fit(X_train, y_train_transformed, epochs=50, batch_size=64)
198/77:
from keras.optimizers import Adam

dense = models_definition.create_dense((X_train.shape[1],), y_train_transformed.shape[1], dropout_probability=0.4,
                                       n_units=512, n_layers=4, final_activation='softmax')
opt = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
dense.compile(loss='mean_squared_error', optimizer=opt)
198/78: dense.fit(X_train, y_train_transformed, epochs=50, batch_size=64)
198/79:
from keras.optimizers import Adam

dense = models_definition.create_dense((X_train.shape[1],), y_train_transformed.shape[1], dropout_probability=0.4,
                                       n_units=512, n_layers=4, final_activation='softmax')
opt = Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
dense.compile(loss='mean_squared_error', optimizer=opt)
198/80: dense.fit(X_train, y_train_transformed, epochs=50, batch_size=64)
198/81: dense.predict(X_test)
198/82:
from keras.optimizers import Adam

dense = models_definition.create_dense((X_train.shape[1],), y_train_transformed.shape[1], dropout_probability=0.4,
                                       n_units=512, n_layers=4, final_activation='softmax')
opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
dense.compile(loss='categorical_crossentropy', optimizer=opt)
198/83: dense.fit(X_train, y_train_transformed, epochs=50, batch_size=64)
198/84:
from keras.optimizers import Adam

dense = models_definition.create_dense((X_train.shape[1],), y_train_transformed.shape[1], dropout_probability=0.4,
                                       n_units=512, n_layers=4, final_activation='softmax')
opt = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
dense.compile(loss='categorical_crossentropy', optimizer=opt)
198/85: dense.fit(X_train, y_train_transformed, epochs=50, batch_size=64)
198/86: dense.predict(X_test)
198/87:
from keras.optimizers import Adam

dense = models_definition.create_dense((X_train.shape[1],), y_train_transformed.shape[1], dropout_probability=0.4,
                                       n_units=512, n_layers=4, final_activation='softmax')
opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
dense.compile(loss='categorical_crossentropy', optimizer=opt)
198/88: dense.fit(X_train, y_train_transformed, epochs=50, batch_size=64)
198/89: dense.predict(X_test)
198/90:
from keras.optimizers import Adam

dense = models_definition.create_dense((X_train.shape[1],), y_train_transformed.shape[1], dropout_probability=0.4,
                                       n_units=512, n_layers=4, final_activation='softmax')
opt = Adam(lr=0.000001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
dense.compile(loss='categorical_crossentropy', optimizer=opt)
198/91: dense.fit(X_train, y_train_transformed, epochs=50, batch_size=64)
198/92: dense.predict(X_test)
198/93:
from keras.optimizers import Adam

dense = models_definition.create_dense((X_train.shape[1],), y_train_transformed.shape[1], dropout_probability=0.4,
                                       n_units=512, n_layers=2, final_activation='softmax')
opt = Adam(lr=0.000001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
dense.compile(loss='categorical_crossentropy', optimizer=opt)
198/94: dense.fit(X_train, y_train_transformed, epochs=50, batch_size=64)
198/95:
from keras.optimizers import Adam

dense = models_definition.create_dense((X_train.shape[1],), y_train_transformed.shape[1], dropout_probability=0.4,
                                       n_units=512, n_layers=2, final_activation='softmax')
opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
dense.compile(loss='categorical_crossentropy', optimizer=opt)
198/96: dense.fit(X_train, y_train_transformed, epochs=50, batch_size=64)
198/97:
from keras.optimizers import Adam

dense = models_definition.create_dense((X_train.shape[1],), y_train_transformed.shape[1], dropout_probability=0.4,
                                       n_units=512, n_layers=1, final_activation='softmax')
opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
dense.compile(loss='categorical_crossentropy', optimizer=opt)
198/98: dense.fit(X_train, y_train_transformed, epochs=50, batch_size=64)
198/99: dense.predict(X_test)
198/100:
from keras.optimizers import Adam

dense = models_definition.create_dense((X_train.shape[1],), y_train_transformed.shape[1], dropout_probability=0.4,
                                       n_units=512, n_layers=1, final_activation='softmax')
opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
dense.compile(loss='mean_squared_error', optimizer=opt)
198/101: dense.fit(X_train, y_train_transformed, epochs=50, batch_size=64)
198/102: dense.predict(X_test)
198/103:
from keras.optimizers import Adam

dense = models_definition.create_dense((X_train.shape[1],), y_train_transformed.shape[1], dropout_probability=0.4,
                                       n_units=512, n_layers=1, final_activation='softmax')
opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
dense.compile(loss='mean_squared_error', optimizer=opt)
198/104: dense.predict(X_test)
198/105: np.sum(dense.predict(X_test) != 1)
198/106: np.sum(dense.predict(X_test) == 1)
198/107:
from keras.optimizers import Adam

dense = models_definition.create_dense((X_train.shape[1],), y_train_transformed.shape[1], dropout_probability=0.4,
                                       n_units=128, n_layers=2, final_activation='softmax')
opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
dense.compile(loss='mean_squared_error', optimizer=opt)
198/108: dense.fit(X_train, y_train_transformed, epochs=50, batch_size=64)
198/109: np.sum(dense.predict(X_test) != 1)
198/110: np.sum(dense.predict(X_test) == 1)
198/111: X_train[:5]
198/112: dense.fit(X_train.values, y_train_transformed, epochs=50, batch_size=64)
198/113: np.sum(dense.predict(X_test) == 1)
198/114: dense.predict(X_test)
198/115: print(dense.predict(X_test))
198/116:
from keras.optimizers import Adam

dense = models_definition.create_dense((X_train.shape[1],), y_train_transformed.shape[1],
                                       n_units=128, n_layers=2, final_activation='softmax')
opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
dense.compile(loss='mean_squared_error', optimizer=opt)
198/117: dense.fit(X_train, y_train_transformed, epochs=50, batch_size=128)
198/118:
from keras.optimizers import Adam
from models import models_definition

dense = models_definition.create_dense((X_train.shape[1],), y_train_transformed.shape[1],
                                       n_units=128, n_layers=2, final_activation='softmax')
opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
dense.compile(loss='mean_squared_error', optimizer=opt)
198/119: dense.summary()
198/120:
from keras.optimizers import Adam
from models import models_definition

dense = models_definition.create_dense((X_train.shape[1],), y_train_transformed.shape[1],
                                       n_units=128, n_layers=2, final_activation='softmax')
opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
dense.compile(loss='mean_squared_error', optimizer=opt)
198/121: dense.fit(X_train, y_train_transformed, epochs=50, batch_size=128)
198/122: dense.fit(X_train.values, y_train_transformed, epochs=50, batch_size=128)
198/123: print(dense.predict(X_test))
198/124: X_test[:5]
198/125: y_test[:5]
198/126:
from keras.optimizers import Adam
from models import models_definition

dense = models_definition.create_dense((X_train.shape[1],), y_train_transformed.shape[1],
                                       n_units=128, n_layers=2, final_activation='softmax')
opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
dense.compile(loss='mean_squared_error', optimizer=opt)
198/127: dense.fit(X_train.values, y_train_transformed, epochs=50, batch_size=128, validation_data=(X_test, y_test_transformed))
198/128: dense.fit(X_train.values, y_train_transformed, epochs=200, batch_size=128, validation_data=(X_test, y_test_transformed))
198/129: print(dense.predict(X_test))
198/130: y_test[:5]
198/131:
from keras.optimizers import Adam
from models import models_definition

dense = models_definition.create_dense((X_train.shape[1],), y_train_transformed.shape[1],
                                       n_units=128, n_layers=2, final_activation='softmax')
opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
dense.compile(loss='mean_squared_error', optimizer=opt)
198/132: dense.fit(X_train.values, y_train_transformed, epochs=200, batch_size=128, validation_data=(X_test, y_test_transformed))
198/133:
from keras.optimizers import Adam
from models import models_definition

dense = models_definition.create_dense((X_train.shape[1],), y_train_transformed.shape[1],
                                       n_units=128, n_layers=2, final_activation='softmax')
opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
dense.compile(loss='mean_squared_error', optimizer=opt)
198/134: dense.fit(X_train.values, y_train_transformed, epochs=200, batch_size=128, validation_data=(X_test, y_test_transformed))
198/135: print(dense.predict(X_test))
198/136:
from keras.optimizers import Adam
from models import models_definition

dense = models_definition.create_dense((X_train.shape[1],), y_train_transformed.shape[1],
                                       n_units=128, n_layers=2, final_activation='softmax')
opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
dense.compile(loss='categorical_crossentropy', optimizer=opt)
198/137: dense.fit(X_train.values, y_train_transformed, epochs=200, batch_size=128, validation_data=(X_test, y_test_transformed))
198/138: print(dense.predict(X_test))
198/139: transformed_back = preds * X_train[['total_volume_ha']]
198/140:
def metrics(y_pred, y_true):
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    rmse = np.sqrt(mean_squared_error(y_pred, y_true, multioutput='raw_values'))
    print(rmse)

preds = dense.predict(X_test)
transformed_back = preds * X_test[['total_volume_ha']]
198/141: preds.shape
198/142: X_test[['total_volume_ha']]
198/143: X_test[['total_volume_ha']].shape
198/144:
def metrics(y_pred, y_true):
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    rmse = np.sqrt(mean_squared_error(y_pred, y_true, multioutput='raw_values'))
    print(rmse)

preds = dense.predict(X_test)
transformed_back = preds * X_test[['total_volume_ha']].values
198/145: X_test[['total_volume_ha']].values.shape
198/146:
def metrics(y_pred, y_true):
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    rmse = np.sqrt(mean_squared_error(y_pred, y_true, multioutput='raw_values'))
    print(rmse)

preds = dense.predict(X_test)
transformed_back = preds * X_test[['total_volume_ha']].values
metrics(preds, y_test)
198/147:
from keras.optimizers import Adam
from models import models_definition

dense = models_definition.create_dense((X_train.shape[1],), y_train_transformed.shape[1],
                                       n_units=128, n_layers=2, final_activation='softmax')
opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
dense.compile(loss='mean_squared_error', optimizer=opt)
198/148: dense.fit(X_train.values, y_train_transformed, epochs=200, batch_size=128, validation_data=(X_test, y_test_transformed))
198/149:
def metrics(y_pred, y_true):
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    rmse = np.sqrt(mean_squared_error(y_pred, y_true, multioutput='raw_values'))
    print(rmse)

preds = dense.predict(X_test)
transformed_back = preds * X_test[['total_volume_ha']].values
metrics(preds, y_test)
198/150:
from keras.optimizers import Adam
from models import models_definition

dense = models_definition.create_dense((X_train.shape[1],), y_train_transformed.shape[1],
                                       n_units=128, n_layers=2, final_activation='softmax')
opt = Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
dense.compile(loss='mean_squared_error', optimizer=opt)
198/151: dense.fit(X_train.values, y_train_transformed, epochs=200, batch_size=128, validation_data=(X_test, y_test_transformed))
198/152:
from keras.optimizers import Adam
from models import models_definition

dense = models_definition.create_dense((X_train.shape[1],), y_train_transformed.shape[1],
                                       n_units=128, n_layers=2, final_activation='softmax')
opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
dense.compile(loss='mean_squared_error', optimizer=opt)
198/153: dense.fit(X_train.values, y_train_transformed, epochs=200, batch_size=128, validation_data=(X_test, y_test_transformed))
198/154:
def metrics(y_pred, y_true):
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    rmse = np.sqrt(mean_squared_error(y_pred, y_true, multioutput='raw_values'))
    print(rmse)

preds = dense.predict(X_test)
transformed_back = preds * X_test[['total_volume_ha']].values
metrics(preds, y_test)
198/155:
from keras.optimizers import Adam
from models import models_definition

dense = models_definition.create_dense((X_train.shape[1],), y_train_transformed.shape[1],
                                       n_units=512, n_layers=3, final_activation='softmax')
opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
dense.compile(loss='mean_squared_error', optimizer=opt)
198/156: dense.fit(X_train.values, y_train_transformed, epochs=200, batch_size=128, validation_data=(X_test, y_test_transformed))
198/157:
def metrics(y_pred, y_true):
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    rmse = np.sqrt(mean_squared_error(y_pred, y_true, multioutput='raw_values'))
    print(rmse)

preds = dense.predict(X_test)
transformed_back = preds * X_test[['total_volume_ha']].values
metrics(preds, y_test)
198/158:
def metrics(y_pred, y_true):
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    rmse = np.sqrt(mean_squared_error(y_pred, y_true, multioutput='raw_values'))
    print(rmse)

preds = dense.predict(X_test)
transformed_back = preds * X_test[['total_volume_ha']].values
metrics(transformed_back, y_test)
198/159:
koskisen_folder = "/home/tman/Work/data/koskisen"
#stand_data = pd.read_csv(os.path.join(koskisen_folder, 'v_stand_level_features.csv'))
grid_data = pd.read_csv(os.path.join(koskisen_folder, 'grid', 'koskisen_grid_data.csv'))

stand_data_aggregated = grid_data.groupby('prd_id').agg('mean').reset_index()
stand_data_aggregated.head()


stand_data_aggregated.isna().mean(axis=0)
features = stand_data_aggregated.dropna()

assert features.isna().sum().sum() == 0

# Drop rows where lidar was done after harvesting and the column after filtering
features = features[features['lidared_before_harvest']].drop('lidared_before_harvest', axis=1)

target_columns = ['pine_volume_ha', 'spruce_volume_ha', 'deciduous_volume_ha']
non_feature_columns = ['prd_id']

X = features.drop(target_columns, axis=1)
y = features[target_columns]
#transformed_y = transform_targets(y)

drop_cols = ["prd_id", "plot_id", "lon", "lat", "total_volume_ha"]
X_train, X_test = data_loading.split_from_ids(X, split_name='koskisen', id_column='prd_id')
X_train, X_test = X_train.drop(drop_cols, axis=1), X_test.drop(drop_cols, axis=1)
y_train, y_test = y.loc[X_train.index, :], y.loc[X_test.index, :]
y_train_transformed, y_test_transformed = transform_targets(y_train), transform_targets(y_test)
198/160:
from keras.optimizers import Adam
from models import models_definition

dense = models_definition.create_dense((X_train.shape[1],), y_train_transformed.shape[1],
                                       n_units=512, n_layers=3, final_activation='softmax')
opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
dense.compile(loss='mean_squared_error', optimizer=opt)
198/161: dense.fit(X_train.values, y_train_transformed, epochs=200, batch_size=128, validation_data=(X_test, y_test_transformed))
198/162:
def metrics(y_pred, y_true):
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    rmse = np.sqrt(mean_squared_error(y_pred, y_true, multioutput='raw_values'))
    print(rmse)

preds = dense.predict(X_test)
transformed_back = preds * X_test[['total_volume_ha']].values
metrics(transformed_back, y_test)
198/163:
koskisen_folder = "/home/tman/Work/data/koskisen"
#stand_data = pd.read_csv(os.path.join(koskisen_folder, 'v_stand_level_features.csv'))
grid_data = pd.read_csv(os.path.join(koskisen_folder, 'grid', 'koskisen_grid_data.csv'))

stand_data_aggregated = grid_data.groupby('prd_id').agg('mean').reset_index()
stand_data_aggregated.head()


stand_data_aggregated.isna().mean(axis=0)
features = stand_data_aggregated.dropna()

assert features.isna().sum().sum() == 0

# Drop rows where lidar was done after harvesting and the column after filtering
features = features[features['lidared_before_harvest']].drop('lidared_before_harvest', axis=1)

target_columns = ['pine_volume_ha', 'spruce_volume_ha', 'deciduous_volume_ha']
non_feature_columns = ['prd_id']

X = features.drop(target_columns, axis=1)
y = features[target_columns]
#transformed_y = transform_targets(y)

drop_cols = ["prd_id", "plot_id", "lon", "lat", "total_volume_ha"]
X_train, X_test = data_loading.split_from_ids(X, split_name='koskisen', id_column='prd_id')
test_total_vols = X_test[['total_volume_ha']].values
X_train, X_test = X_train.drop(drop_cols, axis=1), X_test.drop(drop_cols, axis=1)
y_train, y_test = y.loc[X_train.index, :], y.loc[X_test.index, :]
y_train_transformed, y_test_transformed = transform_targets(y_train), transform_targets(y_test)
198/164:
def metrics(y_pred, y_true):
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    rmse = np.sqrt(mean_squared_error(y_pred, y_true, multioutput='raw_values'))
    print(rmse)

preds = dense.predict(X_test)
transformed_back = preds * test_total_vols
metrics(transformed_back, y_test)
198/165:
from keras.optimizers import Adam
from models import models_definition

dense = models_definition.create_dense((X_train.shape[1],), y_train_transformed.shape[1],
                                       n_units=512, n_layers=3, final_activation='softmax')
opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
dense.compile(loss='categorical_crossentropy', optimizer=opt)
198/166: dense.fit(X_train.values, y_train_transformed, epochs=200, batch_size=128, validation_data=(X_test, y_test_transformed))
198/167:
def metrics(y_pred, y_true):
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    rmse = np.sqrt(mean_squared_error(y_pred, y_true, multioutput='raw_values'))
    print(rmse)

preds = dense.predict(X_test)
transformed_back = preds * test_total_vols
metrics(transformed_back, y_test)
198/168:
from keras.optimizers import Adam
from models import models_definition

dense_distribution = models_definition.create_dense((X_train.shape[1],), y_train_transformed.shape[1],
                                       n_units=512, n_layers=3, final_activation='softmax')
dense_regression = models_definition.create_dense((X_train.shape[1],), 1,
                                       n_units=512, n_layers=3, final_activation='linear')
opt_distribution = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
opt_regression = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
dense_distribution.compile(loss='categorical_crossentropy', optimizer=opt)
dense_total.compile(loss='mean_squared_error', optimizer=opt)
198/169:
from keras.optimizers import Adam
from models import models_definition

dense_distribution = models_definition.create_dense((X_train.shape[1],), y_train_transformed.shape[1],
                                       n_units=512, n_layers=3, final_activation='softmax')
dense_regression = models_definition.create_dense((X_train.shape[1],), 1,
                                       n_units=512, n_layers=3, final_activation='linear')
opt_distribution = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
opt_regression = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
dense_distribution.compile(loss='categorical_crossentropy', optimizer=opt)
dense_regression.compile(loss='mean_squared_error', optimizer=opt)
198/170:
koskisen_folder = "/home/tman/Work/data/koskisen"
#stand_data = pd.read_csv(os.path.join(koskisen_folder, 'v_stand_level_features.csv'))
grid_data = pd.read_csv(os.path.join(koskisen_folder, 'grid', 'koskisen_grid_data.csv'))

stand_data_aggregated = grid_data.groupby('prd_id').agg('mean').reset_index()
stand_data_aggregated.head()


stand_data_aggregated.isna().mean(axis=0)
features = stand_data_aggregated.dropna()

assert features.isna().sum().sum() == 0

# Drop rows where lidar was done after harvesting and the column after filtering
features = features[features['lidared_before_harvest']].drop('lidared_before_harvest', axis=1)

target_columns = ['pine_volume_ha', 'spruce_volume_ha', 'deciduous_volume_ha']
non_feature_columns = ['prd_id']

X = features.drop(target_columns, axis=1)
y = features[target_columns]
#transformed_y = transform_targets(y)

drop_cols = ["prd_id", "plot_id", "lon", "lat", "total_volume_ha"]
X_train, X_test = data_loading.split_from_ids(X, split_name='koskisen', id_column='prd_id')
train_total_vols = X_train[['total_volume_ha']].values
test_total_vols = X_test[['total_volume_ha']].values
X_train, X_test = X_train.drop(drop_cols, axis=1), X_test.drop(drop_cols, axis=1)
y_train, y_test = y.loc[X_train.index, :], y.loc[X_test.index, :]
y_train_transformed, y_test_transformed = transform_targets(y_train), transform_targets(y_test)
198/171:
from keras.optimizers import Adam
from models import models_definition

dense_distribution = models_definition.create_dense((X_train.shape[1],), y_train_transformed.shape[1],
                                       n_units=512, n_layers=3, final_activation='softmax')
dense_regression = models_definition.create_dense((X_train.shape[1],), 1,
                                       n_units=512, n_layers=3, final_activation='linear')
opt_distribution = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
opt_regression = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
dense_distribution.compile(loss='categorical_crossentropy', optimizer=opt)
dense_regression.compile(loss='mean_squared_error', optimizer=opt)
198/172:
#dense_distribution.fit(X_train.values, y_train_transformed, epochs=200, batch_size=128, validation_data=(X_test, y_test_transformed))
dense_regression.fit(X_train.values, train_total_vols, epochs=200, batch_size=128, validation_data=(X_test, test_total_vols))
198/173:
from keras.optimizers import Adam
from models import models_definition

dense_distribution = models_definition.create_dense((X_train.shape[1],), y_train_transformed.shape[1],
                                       n_units=512, n_layers=3, final_activation='softmax')
dense_regression = models_definition.create_dense((X_train.shape[1],), 1,
                                       n_units=512, n_layers=3, final_activation='linear')
opt_distribution = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
opt_regression = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
dense_distribution.compile(loss='categorical_crossentropy', optimizer=opt_distribution)
dense_regression.compile(loss='mean_squared_error', optimizer=opt_regression)
198/174:
#dense_distribution.fit(X_train.values, y_train_transformed, epochs=200, batch_size=128, validation_data=(X_test, y_test_transformed))
dense_regression.fit(X_train.values, train_total_vols, epochs=200, batch_size=128, validation_data=(X_test, test_total_vols))
198/175:
from keras.optimizers import Adam
from models import models_definition

dense_distribution = models_definition.create_dense((X_train.shape[1],), y_train_transformed.shape[1],
                                       n_units=512, n_layers=3, final_activation='softmax')
dense_regression = models_definition.create_dense((X_train.shape[1],), 1,
                                       n_units=512, n_layers=3, final_activation='linear')
opt_distribution = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
opt_regression = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
dense_distribution.compile(loss='categorical_crossentropy', optimizer=opt_distribution)
dense_regression.compile(loss='mean_squared_error', optimizer=opt_regression)
198/176:
#dense_distribution.fit(X_train.values, y_train_transformed, epochs=200, batch_size=128, validation_data=(X_test, y_test_transformed))
dense_regression.fit(X_train.values, train_total_vols, epochs=200, batch_size=128, validation_data=(X_test, test_total_vols))
198/177:
from keras.optimizers import Adam
from models import models_definition

dense_distribution = models_definition.create_dense((X_train.shape[1],), y_train_transformed.shape[1],
                                       n_units=512, n_layers=3, final_activation='softmax')
dense_regression = models_definition.create_dense((X_train.shape[1],), 1,
                                       n_units=512, n_layers=3, final_activation='linear')
opt_distribution = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
opt_regression = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
dense_distribution.compile(loss='categorical_crossentropy', optimizer=opt_distribution)
dense_regression.compile(loss='mean_squared_error', optimizer=opt_regression)
198/178:
#dense_distribution.fit(X_train.values, y_train_transformed, epochs=200, batch_size=128, validation_data=(X_test, y_test_transformed))
dense_regression.fit(X_train.values, train_total_vols, epochs=200, batch_size=128, validation_data=(X_test, test_total_vols))
198/179:
def metrics(y_pred, y_true):
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    rmse = np.sqrt(mean_squared_error(y_pred, y_true, multioutput='raw_values'))
    print(rmse)

preds = dense_distribution.predict(X_test)
total_vol_preds = dense_regression(X_test)
transformed_back = preds * total_vol_preds
metrics(transformed_back, y_test)
198/180:
def metrics(y_pred, y_true):
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    rmse = np.sqrt(mean_squared_error(y_pred, y_true, multioutput='raw_values'))
    print(rmse)

preds = dense_distribution.predict(X_test)
total_vol_preds = dense_regression.predict(X_test)
transformed_back = preds * total_vol_preds
metrics(transformed_back, y_test)
198/181:
def metrics(y_pred, y_true):
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    rmse = np.sqrt(mean_squared_error(y_pred, y_true, multioutput='raw_values'))
    print(rmse)

preds = dense_distribution.predict(X_test.values)
total_vol_preds = dense_regression.predict(X_test.values)
transformed_back = preds * total_vol_preds
metrics(transformed_back, y_test)
198/182: metrics(test_total_vols, total_vol_preds)
198/183:
def metrics(y_pred, y_true):
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    rmse = np.sqrt(mean_squared_error(y_pred, y_true, multioutput='raw_values'))
    print(rmse)

preds = dense_distribution.predict(X_test.values)
total_vol_preds = dense_regression.predict(X_test.values)
transformed_back = preds * test_total_vols
metrics(transformed_back, y_test)
198/184:
dense_distribution.fit(X_train.values, y_train_transformed, epochs=200, batch_size=128, validation_data=(X_test, y_test_transformed))
dense_regression.fit(X_train.values, train_total_vols, epochs=200, batch_size=128, validation_data=(X_test, test_total_vols))
198/185:
def metrics(y_pred, y_true):
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    rmse = np.sqrt(mean_squared_error(y_pred, y_true, multioutput='raw_values'))
    print(rmse)

preds = dense_distribution.predict(X_test.values)
total_vol_preds = dense_regression.predict(X_test.values)
transformed_back = preds * test_total_vols
metrics(transformed_back, y_test)
198/186:
def metrics(y_pred, y_true):
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    rmse = np.sqrt(mean_squared_error(y_pred, y_true, multioutput='raw_values'))
    print(rmse)

preds = dense_distribution.predict(X_test.values)
total_vol_preds = dense_regression.predict(X_test.values)
transformed_back = preds * total_vol_preds
metrics(transformed_back, y_test)
201/1: import pandas as pd
201/2: codes = pd.read_csv("Silvia_codes_translated.csv")
201/3: codes = pd.read_csv("Silvia_codes_translated.csv")
201/4: codes = pd.read_csv("Silvia_codes_translated.csv")
201/5: codes = pd.read_csv("Silvia_codes_translated.csv")
201/6: codes = pd.read_csv("Silvia_codes_translated.csv")
201/7: codes = pd.read_csv("Silvia_codes_translated.csv")
201/8: codes
202/1: import pandas as pd
202/2: codes = pd.read_csv("Silvia_codes_translated.csv")
202/3: codes['NAME']
203/1:
with open("ids.txt", "r") as f:
    for line in f:
        print(line)
203/2: ids = set()
203/3:
with open("ids.txt", "r") as f:
    for line in f:
        set.add(line.trim())
203/4:
with open("ids.txt", "r") as f:
    for line in f:
        set.add(line.strip())
203/5:
with open("ids.txt", "r") as f:
    for line in f:
        ids.add(line.strip())
203/6: ids
204/1: pytorch
205/1:
from __future__ import print_function
import torch
x = torch.rand(5, 3)
print(x)
205/2: torch.cuda.is_available()
206/1: import numpy as np
206/2: gg = numpy.load('0000.npy')
206/3: gg = no.load('0000.npy')
206/4: gg = np.load('0000.npy')
206/5: gg
206/6: gg.sum()
206/7: np.unique(gg)
206/8: gg.shape
207/1: import numpy as np
207/2: np.zeros((1,1,24))
207/3: np.zeros((1,1,24)).shape
208/1: import numpy as np
208/2: from PIL import image
208/3: from PIL import Image
208/4: gg = Image.open("../data/bcs_floor6_play_only_formatted/images/val/official/0000.png")
208/5: gg.shape
208/6: np.array(gg).shape
208/7: gg.shape[:2]
208/8: npgg = np.array(gg)
208/9: npgg
208/10: npgg.shape
208/11: npgg.shape[:2]
208/12: np.zeros(npgg.shape[:2] + (,1)).shape
208/13: np.zeros(npgg.shape[:2] + (1)).shape
208/14: np.zeros(npgg.shape[:2] + (1,)).shape
209/1: import numpy as np
209/2: depth = np.load("../data/bcs_floor6_play_only_formatted/depth/0000.npy")
209/3: depth.shape
209/4: import os
210/1: import os
210/2: gg = os.path("../data/bcs_floor6_play_only_formatted/")
210/3: gg = os.path.join("../data/bcs_floor6_play_only_formatted/")
210/4: gg
210/5: gg.replace("data", "lol2)
210/6: gg.replace("data", "lol")
211/1: ls
211/2: cd ..
211/3: ls
211/4: cd data/
211/5: ls
211/6: cd bcs_floor6_play_only_formatted/
211/7: ls
211/8: poses = np.loadtxt("poses.txt")
211/9: import numpy as np
211/10: poses = np.loadtxt("poses.txt")
211/11: posts
211/12: poses
211/13: poses.shape
211/14: K = np.loadtxt("K.txt")
211/15: K
211/16: pose = poses[0]
211/17: pose
211/18: pose.reshape(4,4)
211/19: R = pose[:3,:3]
211/20: pose.shape
211/21: pose = pose.reshape(4,4)
211/22: R = pose[:3,:3]
211/23: R
211/24: t = pose[:,-1][:3]
211/25: t
211/26: t.dot(np.array([0,0,1/0.5]))
211/27: dep = np.array([0,0,1/0.5])
211/28: dep
211/29: dep.shape
211/30: dep.transpose.shape
211/31: dep.transpose().shape
211/32: dep.shape = (1,3)
211/33: t.dot(dep)
211/34: t.shape
211/35: t.shape = (3,1)
211/36: t.dot(dep)
211/37: K
211/38: H = K.dot((R + t.dot(dep))).dot(K.inv())
211/39: H = K.dot((R + t.dot(dep))).dot(np.linalg.inv(K))
211/40: H
212/1: import numpy as np
212/2: K = np.loadtxt("K.txt")
212/3: poses = np.loadtxt("poses.txt")
212/4: K
212/5: poses
212/6: poses.shape
212/7: poss = poses.shape = (poses.shape[0], 4, 4)
212/8: poss.shape
212/9: poses.shape = (poses.shape[0], 4, 4)
212/10: poses.shape
212/11: poses[0]
212/12: t_j = poses[0, -1, :3]
212/13: t_j
212/14: t_j = poses[0, :3, -1]
212/15: t_j
212/16: poses[:, :3, -1] - t_j
212/17: ti_minus_tj = poses[:, :3, -1] - t_j
212/18: ti_minus_tj.shape
212/19: np.inner(ti_minus_tj, ti_minus_tj)
212/20: np.inner(ti_minus_tj, ti_minus_tj).shape
212/21: np.inner(ti_minus_tj, ti_minus_tj.T).shape
212/22: ti_minus_tj.T.shape
212/23: ti_minus_tj.dot(ti_minus_tj.T).shape
212/24: ti_minus_tj.dot(ti_minus_tj).shape
212/25: np.linalg.norm(ti_minus_tj, ord=2)**2
212/26: np.linalg.norm(ti_minus_tj, ord=2, axis=0)**2
212/27: np.linalg.norm(ti_minus_tj, ord=2, axis=1)**2
212/28: r_j = poses[0,:3,:3]
212/29: r_j
212/30: r_is = poses[:,:3,:3]
212/31: r_is.T
212/32: r_is.shape
212/33: r_is.T.shape
212/34: np.transmute(r_is, axes=(0, 2, 1))
212/35: np.transpose(r_is, axes=(0, 2, 1))
212/36: np.transpose(r_is, axes=(0, 2, 1)).shape
212/37: np.transpose(r_is, axes=(0, 2, 1)).dot(r_j)
212/38: np.transpose(r_is, axes=(0, 2, 1)).dot(r_j).shape
212/39: np.zeros(100, 100)
213/1: import numpy as np
213/2: from utils_mvs_temporal import *
213/3: poses = np.loadtxt("../data/bcs_floor6_play_only_formatted/poses.txt")
213/4: poses .shape = (poses.shape[0], 4, 4)
213/5: poses.shape
213/6: pose_distance_measure(poses)
213/7: from utils_mvs_temporal import *
213/8: pose_distance_measure(poses)
214/1: from utils_mvs_temporal import *
214/2: from utils_mvs_temporal import *
214/3: import numpy as np
214/4: poses = np.loadtxt("../data/bcs_floor6_play_only_formatted/poses.txt")
214/5: poses .shape = (poses.shape[0], 4, 4)
214/6: pose_distance_measure(poses)
214/7: import importlib
214/8: importlib.reload(from utils_mvs_temporal import *)
215/1: %load_ext autoreload
215/2: %autoreload 2
215/3: from utils_mvs_temporal import *
215/4: import numpy as np
215/5: poses = np.loadtxt("../data/bcs_floor6_play_only_formatted/poses.txt")
215/6: poses .shape = (poses.shape[0], 4, 4)
215/7: pose_distance_measure(poses)
215/8: idx = 0
215/9:
t_j = poses[idx, :3, -1]
        ti_minus_tj_norm = np.linalg.norm(poses[:, :3, -1] - t_j, ord=2, axis=1)**2

        r_j = poses[idx, :3, :3]
        r_is = poses[:, :3, :3]
        tr_in = np.transpose(r_is, axes=(0,2,1)).dot(r_j)
215/10:
t_j = poses[idx, :3, -1]
        ti_minus_tj_norm = np.linalg.norm(poses[:, :3, -1] - t_j, ord=2, axis=1)**2

        r_j = poses[idx, :3, :3]
        r_is = poses[:, :3, :3]tr_in = np.transpose(r_is, axes=(0,2,1)).dot(r_j)
215/11:
t_j = poses[idx, :3, -1]
        ti_minus_tj_norm = np.linalg.norm(poses[:, :3, -1] - t_j, ord=2, axis=1)**2

        r_j = poses[idx, :3, :3]
        r_is = poses[:, :3, :3]tr_in = np.transpose(r_is, axes=(0,2,1)).dot(r_j)
215/12: %paste
215/13: %paste
215/14: %paste
215/15: t_j
215/16: tr_in.shape
215/17: np.trace(np.eye(3) - tr_in)
215/18: np.trace(np.eye(3) - tr_in, axis1=1, axis2=2)
215/19: np.trace(np.eye(3) - tr_in, axis1=1, axis2=2).shape
215/20: pose_distance_measure(poses)
215/21: pose_distance_measure(poses)
215/22: distances = pose_distance_measure(poses)
215/23: wat = ti_minus_tj_norm + tr_calc
215/24: tr_calc = (2./3)*np.trace(np.eye(3) - tr_in, axis1=1, axis2=2)
215/25: wat = ti_minus_tj_norm + tr_calc
215/26: wat.shape
215/27: wat
215/28: np.sum(wat < 0)
215/29: wat[wat<0] = 0
215/30: wat
215/31: distances
215/32: distances == np.nan
215/33: np.isnan(distances)
215/34: distances = pose_distance_measure(poses)
215/35: distances
215/36: matern_kernel(distances)
215/37: matern_kernel(distances).shape
215/38: 18**2
215/39: 13.82**2
215/40: matern_kernel(distances)[0]
216/1: import torch
216/2: import torchvision.models
216/3: models.
216/4: torchvision.models.mobilenet()
217/1: import torch
217/2: checkpoint = torch.load("../models/mobilenet-nnconv5dw-skipadd-pruned.pth.tar")
217/3: checkpoint = torch.load("../models/mobilenet-nnconv5dw-skipadd-pruned.pth.tar")
217/4: checkpoint
218/1: import torch
218/2: state_dict = torch.hub.load_state_dict_from_url('https://download.pytorch.org/models/mobilenet_v2-b0353104.pth')
218/3: state_dict = torch.utils.model_zoo.load_url('https://download.pytorch.org/models/mobilenet_v2-b0353104.pth')
219/1: import torch
219/2: torch.version
219/3: torch.version()
219/4: torch.version.debug()
219/5: torch.version.debug
219/6: torch.__version__
220/1: import torch
220/2: torch.cuda_is_available()
220/3: torch.cuda.is_available()
221/1: import torch
221/2: impor torchvision.models
221/3: import torchvision.models
222/1: import torchvision.models
223/1: import torchvision.models
223/2: model = torchvision.models.mobilenet_v2(pretrained=True)
223/3: model
224/1: import torch
224/2: checkpoint = torch.load("../models/mobilenet_sgd_rmsprop_69.526.tar")
224/3: checkpoint
224/4: import imagenet
224/5: mobilenet = imagenet.mobilenet.MobileNet()
224/6: import imagenet.mobilenet
224/7: mobilenet = imagenet.mobilenet.MobileNet()
224/8: state_dict = checkpoint['state_dict']
224/9: %paste
224/10: mobilenet.load_state_dict(new_state_dict)
225/1: import models
225/2: gg = models.MobileNetSkipAdd(10)
225/3: gg
226/1: import sys; print('Python %s on %s' % (sys.version, sys.platform))
226/2: i
226/3: input.shape
226/4: target.shape
226/5: model
226/6: layer = getattr(model, 'conv13')
226/7: layer
226/8: model[:5]
226/9: model.children()
226/10: model.children()[:10]
226/11: *list(model.children())[:10]
226/12: list(model.children())[:10]
226/13: list(model.children())[:14]
226/14: list(model.children())[:15]
226/15: list(model.children())[:14]
   1: %run ipython_start.py
   2: %run ./ipython_start.py
   3: %load ipython_start.py
   4:
# %load ipython_start.py
%load_ext autoreload
%autoreload 2

import numpy as np
import torch
import os
   5: torch.load("../models/mobilenet-nnconv5dw-skipadd-pruned.pth.tar")
   6: basemodel = torch.load("../models/mobilenet-nnconv5dw-skipadd-pruned.pth.tar")
   7: from models_pose import *
   8: augmented = AugmentedFastDepth("../models/mobilenet-nnconv5dw-skipadd-pruned.pth.tar")
   9: augmented
  10: base_model
  11: basemodel
  12: from models import MobileNetSkipAdd
  13: gg = MobileNetSkipAdd(10)
  14: gg
  15: basemodel
  16: augmented.load_state_dict(basemodel)
  17: augmented.load_state_dict(basemodel['state_dict'])
  18: history
  19: basemodel.model
  20: basemodel[0]
  21: basemodel.layer
  22: basemodel.layers
  23: basemodel.keys()
  24: basemodel.model
  25: basemodel['model']
  26: basemodel['model'][0]
  27: augmented['model']
  28: basemodel['model'].layers
  29: basemodel['model'].layer
  30: basemodel['model'].layer()
  31: basemodel['model'].layers()
  32: basemodel[:5]
  33: basemodel['model'][:5]
  34: augmented['model']
  35: basemodel['model']
  36: len(basemodel['model'])
  37: getattr(basemodel, 'conv{}'.format(0))
  38: getattr(basemodel['model'], 'conv{}'.format(0))
  39: getattr(basemodel['model'], 'conv{}'.format(1))
  40: import models_pose.py
  41: import models_pose
  42: augmented = AugmentedFastDepth("asd")
  43: import models_pose
  44: augmented = AugmentedFastDepth("asd")
  45: gg = MobileNetSkipAdd(10)
  46: import models_pose
  47: augmented = AugmentedFastDepth("asd")
  48: augmented = AugmentedFastDepth("asd")
  49: import models_pose
  50: augmented = AugmentedFastDepth("asd")
  51: import models_pose
  52: augmented = AugmentedFastDepth("asd")
  53: import models_pose
  54: augmented = AugmentedFastDepth("asd")
  55: augmented
  56: import models_pose
  57: augmented = AugmentedFastDepth("asd")
  58: augmented
  59: %paste
  60: next(iter(val_loader))
  61: %paste
  62: batch = next(iter(val_loader))
  63: batch
  64: batch.shape
  65: batch.shape[0]
  66: batch[0]
  67: batch[0].shape
  68: len(batch)
  69: history
  70: basemodel(batch[0])
  71: basemodel.eval()
  72: basemodel['model'](batch[0])
  73: torch.cuda.synchronize()
  74:
with torch.no_grad():
    pred = basemodel['model'](batch[0])
  75:
with torch.no_grad():
    pred = basemodel['model'](batch[0].cuda())
  76: pred
  77:
with torch.no_grad():
    pred2 = augmented(batch[0].cuda())
  78:
with torch.no_grad():
    pred2 = augmented(batch[0].cuda(), batch[2].cuda())
  79: import models_pose
  80: augmented = AugmentedFastDepth("asd")
  81:
with torch.no_grad():
    pred2 = augmented(batch[0].cuda(), batch[2].cuda())
  82: import models_pose
  83: augmented = AugmentedFastDepth("asd")
  84:
with torch.no_grad():
    pred2 = augmented(batch[0].cuda(), batch[2].cuda())
  85: pred2
  86: pred
  87: import models_pose
  88: augmented = AugmentedFastDepth("asd")
  89:
with torch.no_grad():
    pred2 = augmented(batch[0].cuda(), batch[2].cuda())
  90: import models_pose
  91: augmented = AugmentedFastDepth("asd")
  92:
with torch.no_grad():
    pred2 = augmented(batch[0].cuda(), batch[2].cuda())
  93: pred2
  94: batch[1]
  95: batch[2]
  96: batch[2].shape
  97:
# set batch size to be 1 for validation
val_loader = torch.utils.data.DataLoader(val_dataset,
    batch_size=5, shuffle=False, num_workers=4, pin_memory=True)
  98: batch = next(iter(val_loader))
  99:
with torch.no_grad():
    pred2 = augmented(batch[0].cuda(), batch[2].cuda())
 100: pred2.shape
 101: pred2[0]
 102: batch[0]
 103: batch[0].shape
 104: history
 105: history > gg.txt
 106: ls
 107: %history
 108: %history > gg.text
 109: ls
 110: %history -g -f ipythonhistory25088
 111: ls
 112: %history -g -f ipythonhistory2508.py
