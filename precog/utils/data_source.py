"""Unified data source that can use either CoinMetrics or Binance"""

from datetime import datetime
from typing import Optional, Union

import pandas as pd

from precog.utils.binance_data import BinanceData
from precog.utils.cm_data import CMData


class DataSource:
    """
    Unified data source that automatically chooses between CoinMetrics and Binance
    """

    def __init__(self, source: str = "binance", api_key: str = ""):
        """
        Initialize data source

        Args:
            source: 'binance' or 'coinmetrics'
            api_key: CoinMetrics API key (if using coinmetrics)
        """
        self.source = source.lower()

        if self.source == "binance":
            self.client = BinanceData()
            print("📊 Using Binance API (free, unlimited historical data)")
        elif self.source == "coinmetrics":
            self.client = CMData(api_key=api_key)
            print("📊 Using CoinMetrics API")
        else:
            # Default to Binance
            self.client = BinanceData()
            print("📊 Using Binance API (default)")

    def get_CM_ReferenceRate(
        self,
        assets: Union[list, str],
        start: Optional[Union[datetime, str]] = None,
        end: Optional[Union[datetime, str]] = None,
        frequency: str = "1m",
        **kwargs,
    ) -> pd.DataFrame:
        """
        Fetch reference rate data (unified interface)

        Returns:
            DataFrame with columns: ['asset', 'time', 'ReferenceRateUSD']
        """
        return self.client.get_CM_ReferenceRate(
            assets=assets, start=start, end=end, frequency=frequency, **kwargs
        )

    def get_pair_candles(
        self,
        pairs: Union[list, str],
        start: Optional[Union[datetime, str]] = None,
        end: Optional[Union[datetime, str]] = None,
        frequency: str = "1h",
        **kwargs,
    ) -> pd.DataFrame:
        """
        Fetch OHLC candle data (unified interface)

        Returns:
            DataFrame with columns: ['pair', 'time', 'price_open', 'price_close', 'price_high', 'price_low']
        """
        return self.client.get_pair_candles(pairs=pairs, start=start, end=end, frequency=frequency, **kwargs)


# For backward compatibility, create an alias
def get_data_source(use_binance: bool = True, api_key: str = "") -> Union[BinanceData, CMData, DataSource]:
    """
    Get data source based on preference

    Args:
        use_binance: If True, use Binance (free, unlimited). If False, use CoinMetrics.
        api_key: CoinMetrics API key (only needed if use_binance=False)

    Returns:
        Data source object
    """
    if use_binance:
        return DataSource(source="binance")
    else:
        return DataSource(source="coinmetrics", api_key=api_key)

