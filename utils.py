from sklearn.metrics import make_scorer
from sklearn.metrics import matthews_corrcoef
import time
from cloudpickle import dump, load
import os
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd
from sklearn.model_selection import  RandomizedSearchCV
# Set the configuration of sklearn
SEED = 42 # for reproducibility

def MCC(y_true, y_pred):
    MCC= matthews_corrcoef(y_true, y_pred)
    return MCC


# Define a function to preprocess the data
def PreprocessData(df):
    
    '''
    This function preprocesses the data by imputing missing values and scaling the numerical features.
    '''

    # Define the numerical and categorical features
    num_features = df._get_numeric_data().columns
    cat_features = list(set(df.columns) - set(num_features))

    # Set up the numerical and categorical transformers
    numeric_transformer = Pipeline(
    steps=[("imputation_mean",SimpleImputer(missing_values=np.nan,strategy="mean")),
          ("scaler",StandardScaler())]
    )
    categorial_transformer = Pipeline(
    steps=[("imputation_mode",SimpleImputer(missing_values=np.nan,strategy="most_frequent")),('onehot',OneHotEncoder(handle_unknown='ignore'))]
    )
    # Set up the preprocessor
    preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, num_features),
        ('cat', categorial_transformer, cat_features)
    ])

    # Fit and transform the data
    X = preprocessor.fit_transform(df)

    # Return the preprocessed data
    return X, preprocessor

# create a utility function to sort the dictionary by values aphabatically
def sort_dict(d):
    return dict(sorted(d.items(), key=lambda x: x[0]))

# Define a function to obtain the out of fold predictions
def get_oof_predictions(model, X_train, y_train, X_test, kf):

    """
    This function trains the model on the training set and returns the out of fold predictions and the test predictions
    """
    
    # Initialize the out of fold predictions
    oof_predictions = np.zeros((X_train.shape[0],))
    # Initialize the test predictions
    test_predictions = np.zeros((X_test.shape[0],))

    # Loop through the training and validation sets
    for j, (train_index, val_index) in enumerate (kf.split(X_train,y_train)):
        print(f'training fold: # {j+1}')
        # Split the training and validation sets
        X_tr, X_val = X_train[train_index], X_train[val_index]
        y_tr, y_val = y_train[train_index], y_train[val_index]
        
        # Fit the model
        model.fit(X_tr, y_tr) 
        # Make predictions
        oof_predictions[val_index] = model.predict(X_val)
        test_predictions += model.predict(X_test)

    # Return the out of fold predictions and the test predictions
    return model, oof_predictions, test_predictions/kf.get_n_splits() # Average the test predictions


def model_evaluation(models, X_train, y_train, X_val, y_val, kf,params=None,mode='tuning'):
    """
    This function evaluates the models using the training and validation datasets, and saves them in a directory with the name of the model.
    The function returns the MCC scores in a dataframe and saves the results in a csv file.
    If activated using the 'tuning' mode, the function tunes the hyperparameters of the models using RandomizedSearchCV.
    If activated using the 'training' mode, the function trains the models and returns the out of fold predictions.
    """
    # Initialize the lists to store the results
    model_list = []
    MCC_train_list = []
    MCC_val_list = []
    time_list = []
    y_train_pred_list = []
    model_params = []
    # Initialize a dataframe of out of fold predictions for the models
    oof_predictions_df = pd.DataFrame()
    # Initialize a dataframe of val predictions for the models
    val_predictions_df = pd.DataFrame()

    for i in range(len(list(models))):
        # Load the model if it is saved in the dictionary
        if os.path.exists((f"/kaggle/working/{list(models.keys())[i]}"+"_"+mode+".pkl" if 'KAGGLE_KERNEL_RUN_TYPE' in os.environ else f"Output\\Models\\{list(models.keys())[i]}"+"_"+mode+".pkl")):
            
            print ('=' * 100)
            print ('Loading model:', list(models.keys())[i])
            with open((f"/kaggle/working/{list(models.keys())[i]}"+"_"+mode+".pkl" if 'KAGGLE_KERNEL_RUN_TYPE' in os.environ else f"Output\\Models\\{list(models.keys())[i]}"+"_"+mode+".pkl"), 'rb') as file:
                model = load(file)
            print('Model-loading success:', list(models.keys())[i], 'Best Parameters:', model.get_params())


            # print('Model-refitting:', list(models.keys())[i], 'Best Parameters:', model.get_params())
            # model.fit(X_train, y_train)

            # Make predictions
            print ('Predicting')
            y_train_pred = model.predict(X_train)
            y_val_pred = model.predict(X_val)

            # Evaluate Train and val dataset
            MCC_train = MCC(y_train, y_train_pred)
            MCC_val = MCC(y_val, y_val_pred)

            print('Model prediction success:', list(models.keys())[i], 'MCC_train:', MCC_train, ' , MCC_val:', MCC_val)

            # Append the results to the lists
            model_list.append(list(models.keys())[i])
            MCC_train_list.append(MCC_train)
            MCC_val_list.append(MCC_val)
            y_train_pred_list.append(y_train_pred)
            model_params.append(model.get_params())

            # read the oof predictions and val predictions from the csv file if the mode is 'training'
            if mode == 'training':
                print ('Reading oof predictions and val predictions')
                oof_predictions = pd.read_csv(f"/kaggle/working/oof_predictions.csv" if 'KAGGLE_KERNEL_RUN_TYPE' in os.environ else f"Output\\Results\\oof_predictions.csv")
                val_predictions = pd.read_csv(f"/kaggle/working/val_predictions.csv" if 'KAGGLE_KERNEL_RUN_TYPE' in os.environ else f"Output\\Results\\val_predictions.csv")
                oof_predictions_df[list(models.keys())[i]] = oof_predictions[list(models.keys())[i]]
                val_predictions_df[list(models.keys())[i]] = val_predictions[list(models.keys())[i]]
                print ('Model:', list(models.keys())[i], 'OOF predictions and val predictions read successfully')

            print('=' * 100)
            print('\n')

        else:
            try:
                model = list(models.values())[i]
                print ('=' * 100)
                print ('Running model:', list(models.keys())[i])
                
                if mode == 'tuning':
                    para = params[list(models.keys())[i]]
                    # Tune the hyperparameters of the models using RandomizedSearchCV
                    if 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:
                        RS = RandomizedSearchCV(model, para, n_iter=5, scoring=make_scorer(MCC, greater_is_better=True), refit=True, n_jobs=-1, random_state=SEED, verbose=1)
                    else:
                        RS = RandomizedSearchCV(model, para, n_iter=5, scoring=make_scorer(MCC, greater_is_better=True), refit=True, n_jobs=6, random_state=SEED, verbose=1)
                    RS.fit(X_train, y_train)
                    # Set the best parameters found by RandomizedSearchCV into the model
                
                    best_model = RS.best_estimator_
                    model_params.append(RS.best_params_)
                    print('Model-tuning success:', list(models.keys())[i], 'Best Parameters:', RS.best_params_)


                elif mode == 'training':
                    # Get the out of fold predictions and the val predictions
                    best_model,oof_predictions, val_predictions = get_oof_predictions(model, X_train, y_train, X_val, kf)
                    # Add the out of fold predictions to the dataframe
                    oof_predictions_df[list(models.keys())[i]] = oof_predictions
                    # Add the val predictions to the dataframe
                    val_predictions_df[list(models.keys())[i]] = val_predictions
                    print('Model-training success:', list(models.keys())[i])
                    # Save the predictions in a csv file
                    oof_predictions_df.to_csv(f"/kaggle/working/oof_predictions.csv" if 'KAGGLE_KERNEL_RUN_TYPE' in os.environ else f"Output\\Results\\oof_predictions.csv", index=False)
                    val_predictions_df.to_csv(f"/kaggle/working/val_predictions.csv" if 'KAGGLE_KERNEL_RUN_TYPE' in os.environ else f"Output\\Results\\val_predictions.csv", index=False)
                    print('Model:', list(models.keys())[i], 'oof predictions and val predictions saved successfully')

                # Save the model
                with open((f"/kaggle/working/{list(models.keys())[i]}"+"_"+mode+".pkl" if 'KAGGLE_KERNEL_RUN_TYPE' in os.environ else f"Output\\Models\\{list(models.keys())[i]}"+"_"+mode+".pkl"), 'wb') as file:
                    dump(best_model, file)
                print('Model saved:', list(models.keys())[i])
                
                # Save the parameters in a csv file
                model_params_df = pd.DataFrame(model_params)
                model_params_df.to_csv(f"/kaggle/working/model_params.csv" if 'KAGGLE_KERNEL_RUN_TYPE' in os.environ else f"Output\\Models\\model_params.csv", index=False)


                # Make predictions
                print ('Predicting')
                y_train_pred = best_model.predict(X_train)
                y_val_pred = best_model.predict(X_val)

                # Evaluate Train and val dataset
                MCC_train = MCC(y_train, y_train_pred)
                MCC_val = MCC(y_val, y_val_pred)

                # Append the results to the lists
                model_list.append(list(models.keys())[i])
                MCC_train_list.append(MCC_train)
                MCC_val_list.append(MCC_val)
                y_train_pred_list.append(y_train_pred)

                print('Model-prediction success:', list(models.keys())[i], 'MCC_train:', MCC_train, ' , MCC_val:', MCC_val)
                print('=' * 100)
                print('\n')

            # Raise exception if the model fails
            except Exception as e:
                print(list(models.keys())[i])
                model_list.append(list(models.keys())[i])
                MCC_train_list.append(np.nan)
                MCC_val_list.append(np.nan)
                y_train_pred_list.append(np.nan)
                model_params.append(np.nan)
                print('Model failed:', e)
                print('=' * 100)
                print('\n')
                continue

    # save the results in a dataframe
    results = pd.DataFrame({'Model': model_list, 'MCC_train': MCC_train_list, 'MCC_val': MCC_val_list})
    
    # save the results in a csv file
    results.to_csv(f"/kaggle/working/results_"+mode+'.csv' if 'KAGGLE_KERNEL_RUN_TYPE' in os.environ else f'Output/Results\\results_'+mode+'.csv', index=False, date_format='%Y%m%d',mode='a')   
    if mode == 'tuning':
        return model_list, MCC_train_list, MCC_val_list, y_train_pred_list, model_params
    elif mode == 'training':
        return model_list, MCC_train_list, MCC_val_list, y_train_pred_list, oof_predictions_df, val_predictions_df