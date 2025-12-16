"""Binance API data fetcher for cryptocurrency prices"""

import time
from datetime import datetime, timedelta
from typing import Optional, Union

import pandas as pd
import requests


class BinanceData:
    """
    Fetch historical cryptocurrency data from Binance API
    Free, no API key required, supports extensive historical data
    """

    def __init__(self):
        self.base_url = "https://api.binance.com/api/v3"
        self._cache = pd.DataFrame()
        self._last_update = None

    # Asset name mapping: precog format -> Binance symbol
    ASSET_MAPPING = {
        "btc": "BTCUSDT",
        "eth": "ETHUSDT",
        "tao_bittensor": "TAOUSDT",  # If TAO is listed on Binance
    }

    # Interval mapping: precog format -> Binance format
    INTERVAL_MAPPING = {
        "1s": "1s",  # Binance doesn't support 1s
        "1m": "1m",
        "5m": "5m",
        "1h": "1h",
        "1d": "1d",
    }

    def get_symbol(self, asset: str) -> str:
        """Convert asset name to Binance symbol"""
        asset_lower = asset.lower()
        return self.ASSET_MAPPING.get(asset_lower, f"{asset_lower.upper()}USDT")

    def get_klines(
        self,
        symbol: str,
        interval: str = "1m",
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: int = 1000,
    ) -> list:
        """
        Fetch klines (candlestick) data from Binance

        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            interval: Kline interval (1m, 5m, 1h, 1d)
            start_time: Start time in milliseconds
            end_time: End time in milliseconds
            limit: Number of klines (max 1000)

        Returns:
            List of klines data
        """
        url = f"{self.base_url}/klines"
        params = {"symbol": symbol, "interval": interval, "limit": limit}

        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data from Binance: {e}")
            return []

    def klines_to_dataframe(self, klines: list, asset: str) -> pd.DataFrame:
        """
        Convert Binance klines to DataFrame

        Binance kline format:
        [
            Open time,
            Open,
            High,
            Low,
            Close,
            Volume,
            Close time,
            Quote asset volume,
            Number of trades,
            Taker buy base asset volume,
            Taker buy quote asset volume,
            Ignore
        ]
        """
        if not klines:
            return pd.DataFrame()

        df = pd.DataFrame(
            klines,
            columns=[
                "open_time",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "close_time",
                "quote_volume",
                "trades",
                "taker_buy_base",
                "taker_buy_quote",
                "ignore",
            ],
        )

        # Convert to proper types
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)

        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Add asset column
        df["asset"] = asset.lower()

        # Rename for compatibility with CMData format
        df["time"] = df["open_time"]
        df["ReferenceRateUSD"] = df["close"]  # Use close price as reference

        return df

    def get_CM_ReferenceRate(
        self,
        assets: Union[list, str],
        start: Optional[Union[datetime, str]] = None,
        end: Optional[Union[datetime, str]] = None,
        frequency: str = "1m",
        **kwargs,
    ) -> pd.DataFrame:
        """
        Fetch reference rate data (compatible with CMData interface)

        Args:
            assets: Asset or list of assets ('btc', 'eth', etc.)
            start: Start datetime
            end: End datetime
            frequency: Data frequency (1m, 5m, 1h, 1d)

        Returns:
            DataFrame with columns: ['asset', 'time', 'ReferenceRateUSD']
        """
        # Handle single asset
        if isinstance(assets, str):
            assets = [assets]

        # Convert frequency
        binance_interval = self.INTERVAL_MAPPING.get(frequency, "1m")

        # Parse times
        if isinstance(start, str):
            start = pd.to_datetime(start)
        if isinstance(end, str):
            end = pd.to_datetime(end)

        if start is None:
            start = datetime.now() - timedelta(days=30)  # Default 30 days
        if end is None:
            end = datetime.now()

        # Convert to milliseconds
        start_ms = int(start.timestamp() * 1000)
        end_ms = int(end.timestamp() * 1000)

        all_data = []

        for asset in assets:
            symbol = self.get_symbol(asset)
            print(f"Fetching {asset} ({symbol}) data from Binance...")

            # Fetch data in chunks (max 1000 per request)
            current_start = start_ms
            chunk_count = 0

            while current_start < end_ms:
                klines = self.get_klines(
                    symbol=symbol, interval=binance_interval, start_time=current_start, end_time=end_ms, limit=1000
                )

                if not klines:
                    break

                df_chunk = self.klines_to_dataframe(klines, asset)
                if not df_chunk.empty:
                    all_data.append(df_chunk)

                # Move to next chunk
                last_time = klines[-1][6]  # Close time
                current_start = last_time + 1
                chunk_count += 1

                # Rate limiting
                time.sleep(0.1)

                # Progress
                if chunk_count % 10 == 0:
                    fetched_time = datetime.fromtimestamp(current_start / 1000)
                    print(f"  Fetched {len(klines) * chunk_count} candles up to {fetched_time}")

                # Safety limit
                if chunk_count > 1000:
                    print(f"  Warning: Reached safety limit of 1000 chunks")
                    break

        if not all_data:
            return pd.DataFrame()

        # Combine all data
        df = pd.concat(all_data, ignore_index=True)

        # Sort and remove duplicates
        df = df.sort_values("time").reset_index(drop=True)
        df = df.drop_duplicates(subset=["asset", "time"], keep="last")

        # Select only needed columns
        result = df[["asset", "time", "ReferenceRateUSD"]].copy()

        print(f"Total fetched: {len(result)} data points")

        return result

    def get_pair_candles(
        self,
        pairs: Union[list, str],
        start: Optional[Union[datetime, str]] = None,
        end: Optional[Union[datetime, str]] = None,
        frequency: str = "1h",
        **kwargs,
    ) -> pd.DataFrame:
        """
        Fetch OHLC candle data

        Returns:
            DataFrame with columns: ['pair', 'time', 'price_open', 'price_close', 'price_high', 'price_low']
        """
        if isinstance(pairs, str):
            pairs = [pairs]

        binance_interval = self.INTERVAL_MAPPING.get(frequency, "1h")

        if isinstance(start, str):
            start = pd.to_datetime(start)
        if isinstance(end, str):
            end = pd.to_datetime(end)

        if start is None:
            start = datetime.now() - timedelta(days=30)
        if end is None:
            end = datetime.now()

        start_ms = int(start.timestamp() * 1000)
        end_ms = int(end.timestamp() * 1000)

        all_data = []

        for pair in pairs:
            # Convert pair format: 'btc-usd' -> 'BTCUSDT'
            symbol = pair.upper().replace("-", "")
            if not symbol.endswith("USDT") and not symbol.endswith("USD"):
                symbol = symbol + "USDT"

            print(f"Fetching {pair} ({symbol}) candles...")

            current_start = start_ms
            while current_start < end_ms:
                klines = self.get_klines(
                    symbol=symbol, interval=binance_interval, start_time=current_start, end_time=end_ms, limit=1000
                )

                if not klines:
                    break

                df_chunk = self.klines_to_dataframe(klines, pair)
                if not df_chunk.empty:
                    all_data.append(df_chunk)

                last_time = klines[-1][6]
                current_start = last_time + 1
                time.sleep(0.1)

        if not all_data:
            return pd.DataFrame()

        df = pd.concat(all_data, ignore_index=True)
        df = df.sort_values("time").reset_index(drop=True)

        # Rename columns for compatibility
        result = df.rename(
            columns={
                "asset": "pair",
                "open": "price_open",
                "close": "price_close",
                "high": "price_high",
                "low": "price_low",
            }
        )

        return result[["pair", "time", "price_open", "price_close", "price_high", "price_low"]]


# Example usage and testing
if __name__ == "__main__":
    binance = BinanceData()

    # Test 1: Fetch BTC data for last 30 days
    print("\n" + "=" * 60)
    print("Test 1: Fetch BTC last 30 days")
    print("=" * 60)

    end_time = datetime.now()
    start_time = end_time - timedelta(days=30)

    data = binance.get_CM_ReferenceRate(assets=["btc"], start=start_time, end=end_time, frequency="1m")

    print(f"\nFetched {len(data)} data points")
    print(f"Date range: {data['time'].min()} to {data['time'].max()}")
    print(f"\nFirst few rows:")
    print(data.head())
    print(f"\nLast few rows:")
    print(data.tail())

    # Test 2: Fetch multiple assets
    print("\n" + "=" * 60)
    print("Test 2: Fetch multiple assets (last 7 days)")
    print("=" * 60)

    start_time = end_time - timedelta(days=7)
    data_multi = binance.get_CM_ReferenceRate(assets=["btc", "eth"], start=start_time, end=end_time, frequency="1m")

    print(f"\nFetched {len(data_multi)} data points")
    for asset in ["btc", "eth"]:
        asset_data = data_multi[data_multi["asset"] == asset]
        print(f"{asset}: {len(asset_data)} points")

