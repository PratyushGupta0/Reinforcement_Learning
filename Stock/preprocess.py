import pandas as pd


def preprocess():
    stock_data = pd.read_csv('data/stock_data.csv')
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])
    stock_data = stock_data.drop(["Volume"], axis=1)
    test_df = stock_data[1600:]
    train_df = stock_data[:1600]
    return train_df, test_df
