from pandas_ml_utils import Model, pd
import sys


if __name__ == '__main__':
    model_file = sys.argv[1]
    print(f"load model {model_file}")
    model = Model.load(model_file)
    print(model)

    df = pd.DataFrame({
        "variance": [0.40614],
        "skewness": [1.34920],
        "kurtosis": [-1.4501],
        "entropy": [- 0.55949]
    })

    print(df.model.predict(model))

