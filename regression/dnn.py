import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from regression_main import do_pca
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.callbacks import EarlyStopping

def create_model():
    model = Sequential()
    model.add(Dense(1024, input_dim=503, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(1))
    
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

num_train = 53
norm_method = StandardScaler()
X_path = "/Users/zimenglyu/Documents/datasets/microbeam/PPM/combined/Cyclone_10_202303_202105_202209_lab_results_spectra.csv"
X_data = pd.read_csv(X_path)
y_path = "/Users/zimenglyu/Documents/datasets/microbeam/PPM/combined/Cyclone_10_202303_202105_202209_lab_results.csv"
y_data = pd.read_csv(y_path)
datetime = y_data['DateTime']
scaler_X = norm_method
X_scaled = scaler_X.fit_transform(X_data.iloc[:, 1:])
X = X_scaled[:num_train, :]
X_test = X_scaled[num_train:, :]

pred_result = pd.DataFrame()
pred_result['DateTime'] = datetime
cols = y_data.columns[1:]

# Prepare the K-fold cross-validation
k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)

mse_per_fold = []
models = []
early_stopping = EarlyStopping(monitor='val_loss', patience=10)
for col in cols:
    # X = df.drop(col, axis=1).values
    scaler_y = norm_method
    # print(y_data[col].to_numpy().reshape(-1, 1).shape
    y_scaled = scaler_y.fit_transform(y_data[[col]])
    y = y_scaled[:num_train, :]
    y_test = y_scaled[num_train:, :]

    for train_index, val_index in kf.split(X):
        train_X, val_X = X[train_index], X[val_index]
        train_y, val_y = y[train_index], y[val_index]

        model = create_model()
        model.fit(train_X, train_y, epochs=100, batch_size=5, verbose=0, validation_data=(val_X, val_y), callbacks=[early_stopping])
        
        models.append(model)
        
        predictions = model.predict(val_X)
        mse = mean_squared_error(val_y, predictions)
        mse_per_fold.append(mse)

    # find the model with the smallest mse
    best_model_index = np.argmin(mse_per_fold)
    best_model = models[best_model_index]

    # Use the best model to make predictions on the test set
    y_pred_test = best_model.predict(X_test)
    y_all = np.append(y, y_pred_test)

    pred_result[[col]] = scaler_y.inverse_transform(y_all.reshape(-1, 1))
    
    filename = "test_dnn.csv"

    pred_result.to_csv(filename, index=False)