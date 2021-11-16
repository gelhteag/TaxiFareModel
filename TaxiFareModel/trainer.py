from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from TaxiFareModel.encoders import DistanceTransformer, TimeFeaturesEncoder
from TaxiFareModel.utils import compute_rmse
from sklearn.model_selection import train_test_split
from TaxiFareModel.data import *

class Trainer():

    def __init__(self,df, pipe=None):
        self.df = df
        self.pipe = pipe


    # implement set_pipeline() function
    def set_pipeline(self):
        '''returns a pipelined model'''
        dist_pipe = Pipeline([('dist_trans', DistanceTransformer()),
                            ('stdscaler', StandardScaler())])
        time_pipe = Pipeline([('time_enc', TimeFeaturesEncoder('pickup_datetime')),
                            ('ohe', OneHotEncoder(handle_unknown='ignore'))])
        preproc_pipe = ColumnTransformer([('distance', dist_pipe, [
            "pickup_latitude", "pickup_longitude", 'dropoff_latitude',
            'dropoff_longitude'
        ]), ('time', time_pipe, ['pickup_datetime'])],
                                        remainder="drop")
        self.pipe = Pipeline([('preproc', preproc_pipe),
                        ('linear_model', LinearRegression())])
        return self.pipe

    def hold_out_data(self, test_size):
        y = self.df["fare_amount"]
        X = self.df.drop("fare_amount", axis=1)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size)
        return self.X_train, self.X_test, self.y_train, self.y_test

    # implement run() function to train the models
    def run(self, test_size=0.15):
        '''returns a trained pipelined model'''
        self.X_train, self.X_test, self.y_train, self.y_test = self.hold_out_data(test_size)
        self.pipe.fit(self.X_train, self.y_train)
        return self.pipe


    # implement evaluate() function
    def evaluate(self, test_size=0.15):
        '''returns the value of the RMSE'''
        self.X_train, self.X_test, self.y_train, self.y_test = self.hold_out_data(test_size)
        y_pred = self.pipe.predict(self.X_test)
        rmse = compute_rmse(y_pred, self.y_test)
        print(rmse)
        return rmse


if __name__ == "__main__":
    data = get_data("/home/tobiours/code/gelhteag/TaxiFareModel/raw_data/train_1k.csv")
    data = clean_data(data)
    trainer = Trainer(data)
    trainer.set_pipeline()
    print(trainer.run())
    print(trainer.evaluate())
    # set X and y
    # hold out
    # train
    # evaluate
