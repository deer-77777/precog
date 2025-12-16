"""
Binance API Data Fetcher for LSTM Price Prediction

Fetches historical kline/candlestick data from Binance API.
"""

import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union

import bittensor as bt
import pandas as pd
import requests


class BinanceData:
    """
    Binance API client for fetching historical price data.
    
    Binance Kline API returns:
    [
        [
            1499040000000,      # Open time (timestamp ms)
            "0.01634000",       # Open price
            "0.80000000",       # High price
            "0.01575800",       # Low price
            "0.01577100",       # Close price
            "148976.11427815",  # Volume
            1499644799999,      # Close time (timestamp ms)
            "2434.19055334",    # Quote asset volume
            308,                # Number of trades
            "1756.87402397",    # Taker buy base asset volume
            "28.46694368",      # Taker buy quote asset volume
            "17928899.62484339" # Ignore
        ],
        ...
    ]
    """
    
    BASE_URL = "https://api.binance.com"
    KLINES_ENDPOINT = "/api/v3/klines"
    
    # Binance interval mappings
    VALID_INTERVALS = [
        "1s", "1m", "3m", "5m", "15m", "30m",
        "1h", "2h", "4h", "6h", "8h", "12h",
        "1d", "3d", "1w", "1M"
    ]
    
    # Maximum limit per request
    MAX_LIMIT = 1000
    
    def __init__(self, api_key: str = "", api_secret: str = "") -> None:
        """
        Initialize Binance data fetcher.
        
        Args:
            api_key: Binance API key (optional for public endpoints)
            api_secret: Binance API secret (optional for public endpoints)
        """
        self._api_key = api_key
        self._api_secret = api_secret
        self._session = requests.Session()
        
        # Cache for each asset
        self._cache: Dict[str, pd.DataFrame] = {}
        self._last_update: Dict[str, datetime] = {}
    
    @property
    def api_key(self) -> str:
        return self._api_key
    
    def _get_headers(self) -> dict:
        """Get request headers"""
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["X-MBX-APIKEY"] = self._api_key
        return headers
    
    def _timestamp_to_ms(self, dt: Union[datetime, str, int]) -> int:
        """Convert datetime to milliseconds timestamp"""
        if isinstance(dt, int):
            return dt
        if isinstance(dt, str):
            dt = pd.to_datetime(dt)
        return int(dt.timestamp() * 1000)
    
    def _ms_to_datetime(self, ms: int) -> datetime:
        """Convert milliseconds timestamp to datetime"""
        return datetime.fromtimestamp(ms / 1000)
    
    def get_klines(
        self,
        symbol: str,
        interval: str = "1m",
        start_time: Optional[Union[datetime, str, int]] = None,
        end_time: Optional[Union[datetime, str, int]] = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """
        Fetch kline/candlestick data from Binance.
        
        Args:
            symbol: Trading pair symbol (e.g., "BTCUSDT")
            interval: Kline interval (e.g., "1m", "5m", "1h", "1d")
            start_time: Start time for data
            end_time: End time for data
            limit: Number of records per request (max 1000)
        
        Returns:
            DataFrame with columns:
                ['open_time', 'open', 'high', 'low', 'close', 'volume',
                 'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                 'taker_buy_quote']
        """
        if interval not in self.VALID_INTERVALS:
            raise ValueError(f"Invalid interval: {interval}. Valid: {self.VALID_INTERVALS}")
        
        params = {
            "symbol": symbol.upper(),
            "interval": interval,
            "limit": min(limit, self.MAX_LIMIT),
        }
        
        if start_time:
            params["startTime"] = self._timestamp_to_ms(start_time)
        if end_time:
            params["endTime"] = self._timestamp_to_ms(end_time)
        
        try:
            response = self._session.get(
                f"{self.BASE_URL}{self.KLINES_ENDPOINT}",
                params=params,
                headers=self._get_headers(),
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()
            
            if not data:
                return pd.DataFrame()
            
            df = pd.DataFrame(data, columns=[
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "quote_volume", "trades", "taker_buy_base",
                "taker_buy_quote", "ignore"
            ])
            
            # Drop the ignore column
            df = df.drop(columns=["ignore"])
            
            # Convert types
            df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
            df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
            
            numeric_cols = ["open", "high", "low", "close", "volume",
                          "quote_volume", "taker_buy_base", "taker_buy_quote"]
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            
            df["trades"] = df["trades"].astype(int)
            
            return df
            
        except requests.exceptions.RequestException as e:
            bt.logging.error(f"Binance API request failed: {e}")
            return pd.DataFrame()
        except Exception as e:
            bt.logging.error(f"Error fetching Binance data: {e}")
            return pd.DataFrame()
    
    def get_historical_klines(
        self,
        symbol: str,
        interval: str = "1m",
        days: int = 60,
        end_time: Optional[Union[datetime, str]] = None,
    ) -> pd.DataFrame:
        """
        Fetch historical klines for a specified number of days.
        
        Handles pagination automatically since Binance limits to 1000 records per request.
        
        Args:
            symbol: Trading pair symbol (e.g., "BTCUSDT")
            interval: Kline interval
            days: Number of days of historical data to fetch (default: 60)
            end_time: End time (defaults to now)
        
        Returns:
            DataFrame with historical kline data
        """
        if end_time is None:
            end_time = datetime.utcnow()
        elif isinstance(end_time, str):
            end_time = pd.to_datetime(end_time)
        
        start_time = end_time - timedelta(days=days)
        
        bt.logging.info(f"Fetching {days} days of {symbol} data from {start_time} to {end_time}")
        
        all_data = []
        current_start = start_time
        
        # Calculate interval in milliseconds for pagination
        interval_ms = self._interval_to_ms(interval)
        
        request_count = 0
        max_requests = 1000  # Safety limit
        
        while current_start < end_time and request_count < max_requests:
            df = self.get_klines(
                symbol=symbol,
                interval=interval,
                start_time=current_start,
                end_time=end_time,
                limit=self.MAX_LIMIT,
            )
            
            if df.empty:
                break
            
            all_data.append(df)
            
            # Move start time forward
            last_time = df["open_time"].max()
            current_start = last_time + timedelta(milliseconds=interval_ms)
            
            request_count += 1
            
            # Rate limiting: Binance allows 1200 requests per minute
            if request_count % 10 == 0:
                time.sleep(0.1)
            
            bt.logging.trace(f"Fetched {len(df)} records, total requests: {request_count}")
        
        if not all_data:
            bt.logging.warning(f"No data fetched for {symbol}")
            return pd.DataFrame()
        
        result = pd.concat(all_data, ignore_index=True)
        result = result.drop_duplicates(subset=["open_time"]).sort_values("open_time").reset_index(drop=True)
        
        bt.logging.info(f"Fetched {len(result)} total records for {symbol}")
        
        return result
    
    def _interval_to_ms(self, interval: str) -> int:
        """Convert interval string to milliseconds"""
        unit = interval[-1]
        value = int(interval[:-1])
        
        multipliers = {
            "s": 1000,
            "m": 60 * 1000,
            "h": 60 * 60 * 1000,
            "d": 24 * 60 * 60 * 1000,
            "w": 7 * 24 * 60 * 60 * 1000,
            "M": 30 * 24 * 60 * 60 * 1000,
        }
        
        return value * multipliers.get(unit, 60 * 1000)
    
    def get_multiple_assets(
        self,
        symbols: List[str],
        interval: str = "1m",
        days: int = 60,
        end_time: Optional[Union[datetime, str]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical data for multiple assets.
        
        Args:
            symbols: List of trading pair symbols
            interval: Kline interval
            days: Number of days of data (default: 60)
            end_time: End time for data
        
        Returns:
            Dictionary mapping symbol to DataFrame
        """
        result = {}
        
        for symbol in symbols:
            bt.logging.info(f"Fetching data for {symbol}...")
            df = self.get_historical_klines(
                symbol=symbol,
                interval=interval,
                days=days,
                end_time=end_time,
            )
            result[symbol] = df
            
            # Small delay between assets
            time.sleep(0.5)
        
        return result
    
    def get_latest_price(self, symbol: str) -> Optional[float]:
        """
        Get the latest price for a symbol.
        
        Args:
            symbol: Trading pair symbol
        
        Returns:
            Latest price or None if failed
        """
        try:
            response = self._session.get(
                f"{self.BASE_URL}/api/v3/ticker/price",
                params={"symbol": symbol.upper()},
                headers=self._get_headers(),
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()
            return float(data["price"])
        except Exception as e:
            bt.logging.error(f"Error fetching latest price for {symbol}: {e}")
            return None
    
    def get_recent_klines(
        self,
        symbol: str,
        interval: str = "1m",
        limit: int = 100,
    ) -> pd.DataFrame:
        """
        Get the most recent klines (for incremental updates).
        
        Args:
            symbol: Trading pair symbol
            interval: Kline interval
            limit: Number of recent records to fetch
        
        Returns:
            DataFrame with recent kline data
        """
        return self.get_klines(symbol=symbol, interval=interval, limit=limit)
    
    def update_cache(
        self,
        symbol: str,
        interval: str = "1m",
        days: int = 60,
    ) -> pd.DataFrame:
        """
        Update the cache for a symbol with fresh data.
        
        Args:
            symbol: Trading pair symbol
            interval: Kline interval
            days: Number of days of data (default: 60)
        
        Returns:
            Updated DataFrame
        """
        cache_key = f"{symbol}_{interval}"
        
        if cache_key in self._cache and not self._cache[cache_key].empty:
            # Get data since last cached point
            last_time = self._cache[cache_key]["open_time"].max()
            new_data = self.get_klines(
                symbol=symbol,
                interval=interval,
                start_time=last_time,
            )
            
            if not new_data.empty:
                self._cache[cache_key] = pd.concat(
                    [self._cache[cache_key], new_data],
                    ignore_index=True,
                ).drop_duplicates(subset=["open_time"]).sort_values("open_time").reset_index(drop=True)
                
                # Trim to keep only last N days
                cutoff = datetime.utcnow() - timedelta(days=days)
                self._cache[cache_key] = self._cache[cache_key][
                    self._cache[cache_key]["open_time"] >= cutoff
                ].reset_index(drop=True)
        else:
            # Fetch full history
            self._cache[cache_key] = self.get_historical_klines(
                symbol=symbol,
                interval=interval,
                days=days,
            )
        
        self._last_update[cache_key] = datetime.utcnow()
        return self._cache[cache_key]
    
    def get_cache_stats(self) -> Dict[str, dict]:
        """Get statistics about cached data"""
        stats = {}
        for key, df in self._cache.items():
            if not df.empty:
                stats[key] = {
                    "rows": len(df),
                    "start_time": df["open_time"].min().isoformat(),
                    "end_time": df["open_time"].max().isoformat(),
                    "last_update": self._last_update.get(key, "N/A"),
                }
        return stats
    
    def clear_cache(self):
        """Clear all cached data"""
        self._cache = {}
        self._last_update = {}


# Utility function to map precog asset names to Binance symbols
def precog_to_binance_symbol(asset: str) -> str:
    """
    Convert precog asset name to Binance trading pair.
    
    Args:
        asset: Precog asset name (e.g., "btc", "eth", "tao_bittensor")
    
    Returns:
        Binance trading pair symbol (e.g., "BTCUSDT")
    """
    mapping = {
        "btc": "BTCUSDT",
        "eth": "ETHUSDT",
        "tao_bittensor": "TAOUSDT",
        "tao": "TAOUSDT",
    }
    return mapping.get(asset.lower(), f"{asset.upper()}USDT")


def binance_to_precog_asset(symbol: str) -> str:
    """
    Convert Binance symbol to precog asset name.
    
    Args:
        symbol: Binance trading pair (e.g., "BTCUSDT")
    
    Returns:
        Precog asset name (e.g., "btc")
    """
    mapping = {
        "BTCUSDT": "btc",
        "ETHUSDT": "eth",
        "TAOUSDT": "tao_bittensor",
    }
    return mapping.get(symbol.upper(), symbol.replace("USDT", "").lower())

