import numpy as np
from sklearn.metrics import matthews_corrcoef

# Define a class to evaluate the model
class model_eval():
    '''
    The class is used to evaluate the model:
    1. Fit the model
    2. Predict the value
    3. Get out of fold prediction
    4. Evaluate the model
    '''

    def __init__(self, model, X_train, y_train, X_test, y_test, cv):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.cv = cv
        self.oof_train = np.zeros((X_train.shape[0],))
        self.oof_test = np.zeros((X_test.shape[0],))
        self.oof_test_skf = np.empty((cv, X_test.shape[0]))
    
    def fit_predict(self):
        for i, (train_index, test_index) in enumerate(self.cv.split(self.X_train)):
            X_tr = self.X_train[train_index]
            y_tr = self.y_train[train_index]
            X_te = self.X_train[test_index]
            self.model.fit(X_tr, y_tr)
            self.oof_train[test_index] = self.model.predict(X_te)
            self.oof_test_skf[i, :] = self.model.predict(self.X_test)
        self.oof_test[:] = self.oof_test_skf.mean(axis=0)
        return self.oof_train, self.oof_test
    
    def evaluate(self):
        train_mcc = matthews_corrcoef(self.y_train, self.oof_train)
        test_mcc = matthews_corrcoef(self.y_test, self.oof_test)
        print('Train MCC: %.3f, Test MCC: %.3f' % (train_mcc, test_mcc))
        return train_mcc, test_mcc
    