import matplotlib.pyplot as plt
import streamlit as st

from pandas_ml_quant import pd, np
from pandas_ml_utils import Model


# @st.cache()
def load_models():
    return (
        Model.load("./models/VixPeakBinaryClassifier.model"),
        Model.load("./models/MultiVixModel.model")
    )


def plot_binary_forecasts(pdf):
    nr_of_timesteps = len(pdf)
    nr_of_charts = nr_of_timesteps + 1
    nr_of_predictions = len(pdf["prediction"].columns)
    x = np.arange(nr_of_predictions + nr_of_timesteps - 1)

    fig, ax = plt.subplots(nr_of_charts, 1, sharex=True, figsize=(15, 20))
    ax[-1].plot(x, df["^VIX", "Close"].shift(-9)[-len(x):].values, label="VIX")
    ax[-1].plot(x, df["^VIX", "Close"].ta.ema(10).shift(-9)[-len(x):].values, label="EMA")

    for i, loc in enumerate(pdf.index):
        data = pdf.loc[loc]["prediction"].values - 0.5
        ax[i].bar(np.arange(i, i + len(data)), data, color=np.where(data > 0, "g", "r"))
        ax[i].hlines(0, 0, len(x), linestyle='--', label=f'{pdf["target"].loc[loc].values[0]:.2f}', alpha=0.2)
        ax[i].set_ylim(-0.55, 0.55)

    fig.legend()
    return fig


def plot_peak_forecast(df):
    fig, ax = plt.subplots(1, 1, figsize=(15, 5))
    ax = df["prediction"].plot.bar(ax = ax)
    ax.set_ylim(0, 1)
    return fig


# Start of APP
st.title("VIX Prediction Model")


# load models
model_load_state = st.text('Loading models ...')
peak_model, binary_classifier_models = load_models()
model_load_state.text('Loading models ... done!')


# load data
data_load_state = st.text('Loading data ...')
df = pd.fetch_yahoo('^VIX', '^VVIX', 'SVXY', multi_index=True)
vix_df = df._[[("^VIX", "Close"), lambda df: df[("^VIX", "Close")].ta.ema(7)]][-90:]
svxy_df = df._[[("SVXY", "Close"), lambda df: df[("SVXY", "Close")].ta.ema(7)]][-90:]
data_load_state.text('Loading data ... done!')

fig, ax = plt.subplots(2, 1, figsize=(15, 15))
vix_df.plot(ax=ax[0])
svxy_df.plot(ax=ax[1])

st.write(vix_df.join(svxy_df).tail(10))
st.write(fig)


# Execute models
prediction_execution_state = st.text("Running models ...")
df_peak = df.model.predict(peak_model, tail=10)
df_classes = df.model.predict(binary_classifier_models, tail=12)
prediction_execution_state.text("Running models ... done!")

st.text("Probability of a VIX spike within the next 7 trading days")
st.write(plot_peak_forecast(df_peak))
st.write(df_peak[["target", "prediction"]][-1:])

st.text("Probability of a VIX to be above of below the next 2 to 9 trading days")
st.write(plot_binary_forecasts(df_classes))
st.write(df_classes[["target", "prediction"]][-1:])


# Show backtest of most recent data to verify model quality
st.write(" ... TODO ...")