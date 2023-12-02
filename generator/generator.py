import pandas as pd
import numpy as np

def linear_regression_generator(id):
    np.random.seed(0)
    x = np.random.rand(100, 1)  
    y = 2 * x + 1 + np.random.randn(100, 1) * 0.5 

    df = pd.DataFrame({'x': x.flatten(), 'y': y.flatten()})

    csv_file = f'generator/generated/linear_regression_data{id}.csv'
    df.to_csv(csv_file, index=False)

    return csv_file
#linear_regression_generator(5)
