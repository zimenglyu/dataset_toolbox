import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, ExpSineSquared, WhiteKernel, ConstantKernel as C
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import optuna
import pymc3 as pm
from sklearn.model_selection import KFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

class Regression:
    def __init__(self, regression_type, kwargs):
        if regression_type not in ["linear", "poly", "gaussian", "dnn", "knn", "k-nei", "dbscan", "dpmm"]:
            raise ValueError("Invalid regression type. Valid options are 'linear', 'poly', 'dnn','dpmm', 'knn', 'k-nei', 'dbscan', and 'gaussian'.")
        self.regression_type = regression_type
        self.kwargs = kwargs
    
    def do_regression(self, X, y, X_test, y_test):
        if self.regression_type == "linear":
            return self.do_linear_regresion(X, y, X_test, y_test)
        elif self.regression_type == "poly":
            return self.do_polynomial_regression(X, y, X_test, y_test)
        elif self.regression_type == "gaussian":
            return self.do_gaussian_regression(X, y, X_test, y_test)
        elif self.regression_type == "dnn":
            return self.do_dnn(X, y, X_test, y_test)
        elif self.regression_type == "knn":
            return self.do_knn(X, y, X_test, y_test)
        elif self.regression_type == "k-nei":
            return self.kmeans_regression(X, y, X_test, y_test)
        elif self.regression_type == "dpmm":
            return self.dpmm_regression(X, y, X_test, y_test)
        
    def dpmm_regression(self, X, y, X_test, y_test):
        # Sample data: X (features) and y (target)
        # Replace these with your actual data

        # DPMM Model
        with pm.Model() as model:
            # Priors for unknown model parameters
            alpha = pm.Gamma('alpha', 1., 1.)
            beta = pm.Normal('beta', 0, sd=10, shape=(2,))
            sigma = pm.HalfNormal('sigma', sd=1)

            # Latent variable for mixture component
            component = pm.DirichletProcess('component', alpha, pm.Normal.dist(0, sd=10), shape=len(X))

            # Regression model for each component
            mu = pm.Deterministic('mu', beta[0] + beta[1] * X.squeeze())

            # Likelihood (sampling distribution) of observations
            Y_obs = pm.Normal('Y_obs', mu=mu, sd=sigma, observed=y)

            # Inference
            trace = pm.sample(1000)
        # Assuming 'model' is your trained PyMC3 model and 'trace' is the trace from the model training
        # Define new data points for which you want to make predictions
        # X_new = np.array([...])  # Replace with new data points

        with model:
            # Update the model with new data
            # Assume 'X_shared' is a theano shared variable used for the input data in your model
            # If you haven't used a shared variable, you'll need to modify the model to accommodate new data
            # X_shared.set_value(X_test)

            # Sample from the posterior predictive distribution for new data
            posterior_predictive = pm.sample_posterior_predictive(trace, var_names=['Y_obs'], samples=1000)

        # Extract the predictive mean for each new data point
        predicted_means = np.mean(posterior_predictive['Y_obs'], axis=0)

        # Alternatively, you can keep the entire distribution for each point to represent the uncertainty
        predicted_distributions = posterior_predictive['Y_obs']

        # Print or return the predictions
        print("Predicted means:", predicted_means)

        return predicted_means, 1



    def kmeans_regression(self, X, y, X_test, y_test):
        print("doing k-fold regression")
        n_clusters = 20
        # Step 1: K-means Clustering
        kmeans = KMeans(n_clusters=n_clusters)
        clusters = kmeans.fit_predict(X)

        # Step 2: K-fold Cross Validation
        kf = KFold(n_splits=5)
        scores = []
        models = []
        
        for train_index, test_index in kf.split(X):
            print("traing index", train_index)
            print("test index,", test_index)
            X_train, Xtest = X[train_index], X[test_index]
            y_train, ytest = y[train_index], y[test_index]
            cluster_models = []
            predictions = np.zeros_like(ytest)

            for cluster in range(n_clusters):
                # Apply regression within each cluster
                cluster_train_index = (clusters[train_index] == cluster)
                cluster_test_index = (clusters[test_index] == cluster)
                
                reg = LinearRegression()
                reg.fit(X_train[cluster_train_index], y_train[cluster_train_index])
                predictions[cluster_test_index] = reg.predict(Xtest[cluster_test_index])
                cluster_models.append(reg)

            score = mean_squared_error(y_test, predictions, squared=False)  # RMSE
            scores.append(score)
            models.append(cluster_models)

        # Step 3: Choose the best model based on average score
        best_index = np.argmin(scores)
        best_models = models[best_index]

        # Step 4: Function to make predictions using the best models
        def predict(X_test):
            clusters_new = kmeans.predict(X_test)
            predictions_new = np.zeros(len(X_test))

            for cluster, model in enumerate(best_models):
                cluster_index = (clusters_new == cluster)
                predictions_new[cluster_index] = model.predict(X_test[cluster_index])

            return predictions_new

        return predict, np.min(scores)


        # return best_predictions, 1


    def do_knn(self, X, y, X_test, y_test):
        neighbor_values = range(1, 10)

        # Perform k-Fold Cross-Validation for each n_neighbors and record the average score
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        avg_scores = {}

        for n in neighbor_values:
            knn = KNeighborsRegressor(n_neighbors=n)
            cv_scores = cross_val_score(knn, X, y, cv=kf)
            avg_scores[n] = np.mean(cv_scores)

        # Find the best n_neighbors value and its score
        best_n = max(avg_scores, key=avg_scores.get)
        best_score = avg_scores[best_n]

        # Train the model with the best n_neighbors
        best_knn = KNeighborsRegressor(n_neighbors=best_n)
        best_knn.fit(X, y)

        # Making predictions on the test set
        y_pred_test = best_knn.predict(X_test)
        return y_pred_test, best_score

    def do_linear_regresion(self, X, y, X_test, y_test):
        model = LinearRegression()
        cv_scores = cross_val_score(model, X, y, cv=self.kwargs.num_k_fold)
        # print("Cross-Validation Scores:", np.mean(cv_scores))
        model.fit(X, y)
        test_score = model.score(X_test, y_test)
        y_pred_test = model.predict(X_test)
        # print("test score:", test_score)
        # test_mse = mean_squared_error(y_test, y_pred_test)
        return y_pred_test, test_score
    
    def do_polynomial_regression(self, X, y, X_test, y_test):
        best_model = None
        best_degree = None
        best_score = -np.inf
        
        for degree in range(1, 5):
            poly_features = PolynomialFeatures(degree=degree)
            X_poly = poly_features.fit_transform(X)

            # Create a linear regression model
            model = LinearRegression()

            # Perform cross-validation on the training set
            scores = cross_val_score(model, X_poly, y, cv=self.kwargs.num_k_fold)
            mean_score = np.mean(scores)

            # Check if this model has the best score so far
            if mean_score > best_score:
                best_score = mean_score
                best_degree = degree
                best_model = model

        # Step 4: Train the best model on the full training set
        poly_features = PolynomialFeatures(degree=best_degree)
        print("best degree:", best_degree)
        X_poly = poly_features.fit_transform(X)
        best_model.fit(X_poly, y)

        X_test_poly = poly_features.transform(X_test)
        y_pred_test = best_model.predict(X_test_poly)
        return y_pred_test, best_score
    
    def create_dnn_model(self):
        model = Sequential()
        model.add(Dense(27, input_dim=27, activation='relu'))
        # model.add(Dense(30, activation='relu'))
        model.add(Dense(15, activation='relu'))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(1))
        
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model
    # def create_dnn_model(self):
    #     model = Sequential()
    #     model.add(Dense(40, input_dim=32, activation='relu'))
    #     model.add(Dense(20,  activation='relu'))
    #     model.add(Dense(10, activation='relu'))
    #     model.add(Dense(1))
        
    #     model.compile(loss='mean_squared_error', optimizer='adam')
    #     return model
    
    def do_dnn(self, X, y, X_test, y_test):
        k = 5
        kf = KFold(n_splits=k, shuffle=True, random_state=42)

        mse_per_fold = []
        models = []
        early_stopping = EarlyStopping(monitor='val_loss', patience=10)

        for train_index, val_index in kf.split(X):
            train_X, val_X = X[train_index], X[val_index]
            train_y, val_y = y[train_index], y[val_index]

            model = self.create_dnn_model()
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
        return y_pred_test, 1
    
    def do_gaussian_regression(self, X, y, X_test, y_test):
        study = optuna.create_study(direction='minimize')
        best_model = None
        if (self.kwargs.kernal_function == "RBF"):
            print("doing RBF")
            study.optimize(lambda trial: self.objective_RBF(trial, X, y, X_test, y_test), n_trials=self.kwargs.num_trails)
            # study.optimize(self.objective_RBF(), n_trials=self.kwargs.num_trails)
            best_model = self.get_best_model_RBF(study)
        elif (self.kwargs.kernal_function == "Matern"):
            print("doing Matern")
            # study.optimize(self.objective_matern, n_trials=self.kwargs.num_trails)
            study.optimize(lambda trial: self.objective_matern(trial, X, y, X_test, y_test), n_trials=self.kwargs.num_trails)
            best_model = self.get_best_model_matern(study)
        # print("Best trial: ", study.best_trial.params)
        else:
            print("Invalid kernal function {}. Valid options are 'RBF' and 'Matern'.".format(self.kwargs.kernal_function))
        
        best_model.fit(X, y)
        y_pred_test = best_model.predict(X_test)
        return y_pred_test, best_model.score(X_test, y_test)

    def get_best_model_RBF(self, study):
        best_alpha = study.best_params['alpha']
        best_constant = study.best_params['constant']
        best_length_scale = study.best_params['length_scale']
        best_kernel = C(best_constant) * RBF(length_scale=best_length_scale)
        best_model = GaussianProcessRegressor(kernel=best_kernel, alpha=best_alpha)
        print("best alpha:", best_alpha)
        print("best constant:", best_constant)
        print("best length scale:", best_length_scale)
        return best_model

    def get_best_model_matern(self, study):
        best_alpha = study.best_params['alpha']
        best_constant = study.best_params['constant']
        best_length_scale = study.best_params['length_scale']
        best_nu = study.best_params['nu']
        best_kernel = C(best_constant) * Matern(length_scale=best_length_scale, nu=best_nu)
        best_model = GaussianProcessRegressor(kernel=best_kernel, alpha=best_alpha)
        print("best alpha:", best_alpha)
        print("best constant:", best_constant)
        print("best length scale:", best_length_scale)
        print("best nu:", best_nu)
        return best_model


    def objective_RBF(self, trial, X, y, X_test, y_test):
        alpha = trial.suggest_float('alpha', 1e-5, 1e-1, log=True)
        constant = trial.suggest_float('constant', 1e-5, 1e1, log=True)
        length_scale = trial.suggest_float('length_scale', 1e-1, 10.0, log=True)
        kernel = C(constant) * RBF(length_scale=length_scale)
        model = GaussianProcessRegressor(kernel=kernel, alpha=alpha, n_restarts_optimizer=5)

        # Perform cross-validation on the training set
        cv_scores = cross_val_score(model, X, y, cv=self.kwargs.num_k_fold)
        return -np.mean(cv_scores)  # Minimize the negative mean cross-validation score

    def objective_matern(self, trial, X, y, X_test, y_test):
        alpha = trial.suggest_float('alpha', 1e-5, 1e-1, log=True)
        constant = trial.suggest_float('constant', 1e-5, 1e1, log=True)
        length_scale = trial.suggest_float('length_scale', 1e-1, 10.0, log=True)
        nu = trial.suggest_float('nu', 0.1, 2.5)
        kernel = C(constant) * Matern(length_scale=length_scale, nu=nu)
        model = GaussianProcessRegressor(kernel=kernel, alpha=alpha)

        # Perform cross-validation on the training set
        cv_scores = cross_val_score(model, X, y, cv=self.kwargs.num_k_fold)
        return -np.mean(cv_scores)  # Minimize the negative mean cross-validation score