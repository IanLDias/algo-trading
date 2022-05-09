#%%
import requests
import pandas as pd

#%% 
assets_repsonse = requests.get('https://community-api.coinmetrics.io/v4/catalog/assets?pretty=true').json()
response_df = pd.DataFrame(assets_repsonse['data'])
df_exchange_markets = response_df.drop('metrics', axis=1)

# Give the number of exchanges. Make a ranking of most common to least common
# Give the number of markets. Find more information for each
df_exchange_markets
# %%
available_metrics_response = requests.get('https://community-api.coinmetrics.io/v4/catalog/metrics?pretty=true').json()
available_metrics_df = pd.DataFrame(available_metrics_response['data'])
# %%
asset_metrics_response = requests.get('https://community-api.coinmetrics.io/v4/timeseries/exchange-metrics?exchanges=binance&metrics=volume_reported_spot_usd_1d&frequency=1d&pretty=true').json()


# %%
