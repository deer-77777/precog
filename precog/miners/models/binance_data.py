"""
Binance Data Fetcher for LSTM+XGBoost Hybrid Model

Fetches 1-minute OHLCV klines from Binance for BTC, ETH, and TAO.
"""

import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import bittensor as bt
import numpy as np
import pandas as pd
from binance.client import Client
from binance.exceptions import BinanceAPIException


class BinanceDataFetcher:
    """Fetches and caches OHLCV data from Binance."""

    # Symbol mappings for our assets
    SYMBOL_MAP = {
        "btc": "BTCUSDT",
        "eth": "ETHUSDT",
        "tao": "TAOUSDT",  # TAO is listed on Binance
    }

    # Binance kline columns
    KLINE_COLUMNS = [
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
    ]

    def __init__(self, api_key: str = "", api_secret: str = ""):
        """
        Initialize Binance client.

        Args:
            api_key: Binance API key (optional for public endpoints)
            api_secret: Binance API secret (optional for public endpoints)
        """
        self.client = Client(api_key, api_secret)
        self._cache: Dict[str, pd.DataFrame] = {}
        self._cache_timestamps: Dict[str, datetime] = {}
        self._cache_max_age = timedelta(minutes=1)  # Cache expires after 1 min

    def get_symbol(self, asset: str) -> str:
        """Convert asset name to Binance symbol."""
        asset_lower = asset.lower()
        if asset_lower in self.SYMBOL_MAP:
            return self.SYMBOL_MAP[asset_lower]
        # Try direct symbol (e.g., if already BTCUSDT)
        return asset.upper()

    def fetch_klines(
        self,
        symbol: str,
        interval: str = "1m",
        limit: int = 1000,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Fetch klines from Binance.

        Args:
            symbol: Trading pair symbol (e.g., BTCUSDT)
            interval: Kline interval (1m, 5m, 1h, etc.)
            limit: Number of klines to fetch (max 1000 per request)
            start_time: Start time for historical data
            end_time: End time for historical data

        Returns:
            DataFrame with OHLCV data
        """
        try:
            kwargs = {"symbol": symbol, "interval": interval, "limit": limit}

            if start_time:
                kwargs["startTime"] = int(start_time.timestamp() * 1000)
            if end_time:
                kwargs["endTime"] = int(end_time.timestamp() * 1000)

            klines = self.client.get_klines(**kwargs)

            if not klines:
                bt.logging.warning(f"No klines returned for {symbol}")
                return pd.DataFrame()

            df = pd.DataFrame(klines, columns=self.KLINE_COLUMNS)

            # Convert types
            df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
            df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")

            for col in ["open", "high", "low", "close", "volume", "quote_volume"]:
                df[col] = df[col].astype(float)

            df["trades"] = df["trades"].astype(int)

            # Set index
            df.set_index("open_time", inplace=True)
            df.index.name = "timestamp"

            return df

        except BinanceAPIException as e:
            bt.logging.error(f"Binance API error for {symbol}: {e}")
            return pd.DataFrame()
        except Exception as e:
            bt.logging.error(f"Error fetching klines for {symbol}: {e}")
            return pd.DataFrame()

    def fetch_historical_klines(
        self,
        symbol: str,
        interval: str = "1m",
        days: int = 30,
        end_time: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Fetch historical klines for multiple days.

        Binance limits to 1000 klines per request, so we batch.
        For 1m interval over 30 days = 43,200 klines = ~44 requests.

        Args:
            symbol: Trading pair symbol
            interval: Kline interval
            days: Number of days of historical data
            end_time: End time (defaults to now)

        Returns:
            DataFrame with OHLCV data
        """
        if end_time is None:
            end_time = datetime.utcnow()

        start_time = end_time - timedelta(days=days)

        all_klines = []
        current_start = start_time

        bt.logging.info(f"Fetching {days} days of {interval} klines for {symbol}...")

        request_count = 0
        max_requests = 100  # Safety limit

        while current_start < end_time and request_count < max_requests:
            df = self.fetch_klines(
                symbol=symbol,
                interval=interval,
                limit=1000,
                start_time=current_start,
                end_time=end_time,
            )

            if df.empty:
                break

            all_klines.append(df)

            # Move start time forward
            current_start = df.index[-1].to_pydatetime() + timedelta(minutes=1)
            request_count += 1

            # Rate limiting - be nice to Binance
            if request_count % 10 == 0:
                time.sleep(0.5)

        if not all_klines:
            bt.logging.warning(f"No historical data fetched for {symbol}")
            return pd.DataFrame()

        result = pd.concat(all_klines)
        result = result[~result.index.duplicated(keep="last")]
        result.sort_index(inplace=True)

        bt.logging.info(f"Fetched {len(result)} klines for {symbol} ({request_count} requests)")

        return result

    def get_training_data(
        self,
        assets: List[str],
        days: int = 30,
        end_time: Optional[datetime] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Get training data for multiple assets.

        Args:
            assets: List of assets (btc, eth, tao)
            days: Number of days of historical data
            end_time: End time for data

        Returns:
            Dict mapping asset name to DataFrame
        """
        data = {}

        for asset in assets:
            symbol = self.get_symbol(asset)
            df = self.fetch_historical_klines(
                symbol=symbol,
                interval="1m",
                days=days,
                end_time=end_time,
            )
            if not df.empty:
                data[asset.lower()] = df
            else:
                bt.logging.warning(f"No training data for {asset}")

        return data

    def get_latest_klines(
        self,
        asset: str,
        lookback: int = 120,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Get the latest klines for inference.

        Args:
            asset: Asset name (btc, eth, tao)
            lookback: Number of klines to fetch
            use_cache: Whether to use cached data

        Returns:
            DataFrame with latest OHLCV data
        """
        asset_lower = asset.lower()
        symbol = self.get_symbol(asset_lower)

        # Check cache
        if use_cache and asset_lower in self._cache:
            cache_time = self._cache_timestamps.get(asset_lower)
            if cache_time and datetime.utcnow() - cache_time < self._cache_max_age:
                cached_df = self._cache[asset_lower]
                if len(cached_df) >= lookback:
                    return cached_df.tail(lookback)

        # Fetch fresh data
        df = self.fetch_klines(symbol=symbol, interval="1m", limit=lookback)

        if not df.empty:
            self._cache[asset_lower] = df
            self._cache_timestamps[asset_lower] = datetime.utcnow()

        return df

    def get_current_price(self, asset: str) -> Optional[float]:
        """Get the current price for an asset."""
        try:
            symbol = self.get_symbol(asset)
            ticker = self.client.get_symbol_ticker(symbol=symbol)
            return float(ticker["price"])
        except Exception as e:
            bt.logging.error(f"Error getting current price for {asset}: {e}")
            return None

    def clear_cache(self):
        """Clear the data cache."""
        self._cache.clear()
        self._cache_timestamps.clear()


# Singleton instance for global use
_binance_fetcher: Optional[BinanceDataFetcher] = None


def get_binance_fetcher(api_key: str = "", api_secret: str = "") -> BinanceDataFetcher:
    """Get or create the global Binance fetcher instance."""
    global _binance_fetcher
    if _binance_fetcher is None:
        _binance_fetcher = BinanceDataFetcher(api_key, api_secret)
    return _binance_fetcher

