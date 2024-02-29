import pandas as pd

filename = '/Users/zimenglyu/Downloads/SnP_500.csv'

pd.read_csv(filename).iloc[::-1].to_csv('/Users/zimenglyu/Downloads/SnP_500_flip.csv', index=False)