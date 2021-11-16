"""
params: nrows, path
get_data fetch the data from the path and return a datafram df with nrows
"""

import pandas as pd


def get_data(path, nrows=None):
    '''returns a DataFrame with nrows from s3 bucket'''

    df = pd.read_csv(path, nrows=nrows)
    return df

def clean_data(df, test=False):
    '''returns a DataFrame without outliers and missing values'''
    df = df.dropna(how='any')
    df = df[(df.dropoff_latitude != 0) | (df.dropoff_longitude != 0)]
    df = df[(df.pickup_latitude != 0) | (df.pickup_longitude != 0)]
    if "fare_amount" in list(df):
        df = df[df.fare_amount.between(0, 4000)]
    df = df[df.passenger_count < 8]
    df = df[df.passenger_count >= 0]
    df = df[df["pickup_latitude"].between(left=40, right=42)]
    df = df[df["pickup_longitude"].between(left=-74.3, right=-72.9)]
    df = df[df["dropoff_latitude"].between(left=40, right=42)]
    df = df[df["dropoff_longitude"].between(left=-74, right=-72.9)]
    return df
