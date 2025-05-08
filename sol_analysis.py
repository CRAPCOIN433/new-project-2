#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Основной модуль анализа SOLUSDT на 4-часовом таймфрейме.
"""

import os
import time
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from binance.client import Client
from binance.exceptions import BinanceAPIException
import ta
from ta.trend import MACD, SMAIndicator, EMAIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands
from ta.volume import VolumeWeightedAveragePrice
from dotenv import load_dotenv
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.patterns import detect_patterns
from utils.support_resistance import find_support_resistance_levels
from utils.market_regime import detect_market_regime
from utils.report_generator import generate_report

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("sol_analysis.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Загрузка переменных окружения
load_dotenv()

class SOLAnalyzer:
    """Класс для анализа SOLUSDT на 4-часовом таймфрейме."""
    
    def __init__(self):
        """Инициализация анализатора."""
        self.api_key = os.getenv('BINANCE_API_KEY')
        self.api_secret = os.getenv('BINANCE_API_SECRET')
        self.symbol = 'SOLUSDT'
        self.timeframe = '4h'
        self.client = None
        self.data = None
        self.connect_to_binance()
    
    def connect_to_binance(self):
        """Установка соединения с Binance API."""
        try:
            # Если ключи не указаны, используем публичный API с ограничениями
            if not self.api_key or not self.api_secret:
                logger.warning("API ключи не найдены. Используется публичный API с ограничениями.")
                self.client = Client()
            else:
                self.client = Client(self.api_key, self.api_secret)
            
            # Проверка соединения
            server_time = self.client.get_server_time()
            logger.info(f"Соединение с Binance установлено. Время сервера: "
                      f"{datetime.fromtimestamp(server_time['serverTime']/1000)}")
        except BinanceAPIException as e:
            logger.error(f"Ошибка при подключении к Binance API: {e}")
            raise
    
    def fetch_historical_data(self, limit=500):
        """
        Получение исторических данных для SOLUSDT.
        
        Args:
            limit (int): Количество свечей для загрузки
            
        Returns:
            pd.DataFrame: Данные в формате DataFrame
        """
        try:
            logger.info(f"Загрузка исторических данных {self.symbol} на таймфрейме {self.timeframe}...")
            
            # Преобразование таймфрейма в формат Binance
            interval_map = {'1m': Client.KLINE_INTERVAL_1MINUTE,
                           '5m': Client.KLINE_INTERVAL_5MINUTE,
                           '15m': Client.KLINE_INTERVAL_15MINUTE,
                           '30m': Client.KLINE_INTERVAL_30MINUTE,
                           '1h': Client.KLINE_INTERVAL_1HOUR,
                           '4h': Client.KLINE_INTERVAL_4HOUR,
                           '1d': Client.KLINE_INTERVAL_1DAY}
            
            interval = interval_map.get(self.timeframe)
            if not interval:
                raise ValueError(f"Неподдерживаемый таймфрейм: {self.timeframe}")
            
            # Получение исторических данных
            klines = self.client.get_historical_klines(
                symbol=self.symbol,
                interval=interval,
                limit=limit
            )
            
            # Преобразование данных в DataFrame
            self.data = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Преобразование типов данных
            self.data['timestamp'] = pd.to_datetime(self.data['timestamp'], unit='ms')
            numeric_columns = ['open', 'high', 'low', 'close', 'volume', 
                               'quote_asset_volume', 'taker_buy_base_asset_volume', 
                               'taker_buy_quote_asset_volume']
            self.data[numeric_columns] = self.data[numeric_columns].astype(float)
            
            # Установка timestamp в качестве индекса
            self.data.set_index('timestamp', inplace=True)
            
            logger.info(f"Загружено {len(self.data)} свечей с {self.data.index[0]} по {self.data.index[-1]}")
            return self.data
            
        except BinanceAPIException as e:
            logger.error(f"Ошибка при получении исторических данных: {e}")
            raise
    
    def calculate_indicators(self):
        """
        Расчет технических индикаторов для анализа.
        
        Returns:
            pd.DataFrame: Данные с добавленными индикаторами
        """
        if self.data is None:
            logger.error("Данные не загружены. Сначала вызовите fetch_historical_data()")
            return None
        
        logger.info("Расчет технических индикаторов...")
        
        # Скользящие средние
        self.data['sma20'] = SMAIndicator(close=self.data['close'], window=20).sma_indicator()
        self.data['sma50'] = SMAIndicator(close=self.data['close'], window=50).sma_indicator()
        self.data['sma100'] = SMAIndicator(close=self.data['close'], window=100).sma_indicator()
        self.data['sma200'] = SMAIndicator(close=self.data['close'], window=200).sma_indicator()
        
        self.data['ema20'] = EMAIndicator(close=self.data['close'], window=20).ema_indicator()
        self.data['ema50'] = EMAIndicator(close=self.data['close'], window=50).ema_indicator()
        self.data['ema100'] = EMAIndicator(close=self.data['close'], window=100).ema_indicator()
        
        # RSI
        rsi = RSIIndicator(close=self.data['close'], window=14)
        self.data['rsi'] = rsi.rsi()
        
        # MACD
        macd = MACD(close=self.data['close'], window_slow=26, window_fast=12, window_sign=9)
        self.data['macd'] = macd.macd()
        self.data['macd_signal'] = macd.macd_signal()
        self.data['macd_diff'] = macd.macd_diff()
        
        # Bollinger Bands
        bollinger = BollingerBands(close=self.data['close'], window=20, window_dev=2)
        self.data['bb_high'] = bollinger.bollinger_hband()
        self.data['bb_mid'] = bollinger.bollinger_mavg()
        self.data['bb_low'] = bollinger.bollinger_lband()
        self.data['bb_width'] = bollinger.bollinger_wband()
        
        # Стохастический осциллятор
        stoch = StochasticOscillator(
            high=self.data['high'],
            low=self.data['low'],
            close=self.data['close'],
            window=14,
            smooth_window=3
        )
        self.data['stoch_k'] = stoch.stoch()
        self.data['stoch_d'] = stoch.stoch_signal()
        
        # Объем
        self.data['volume_sma20'] = SMAIndicator(close=self.data['volume'], window=20).sma_indicator()
        
        # Дополнительные показатели
        self.data['daily_return'] = self.data['close'].pct_change() * 100
        self.data['volatility'] = self.data['daily_return'].rolling(window=20).std()
        
        # Определение тренда
        self.data['trend'] = np.where(self.data['sma20'] > self.data['sma50'], 1, 
                                     np.where(self.data['sma20'] < self.data['sma50'], -1, 0))
        
        # Определение перекупленности/перепроданности
        self.data['overbought'] = np.where(self.data['rsi'] > 70, 1, 0)
        self.data['oversold'] = np.where(self.data['rsi'] < 30, 1, 0)
        
        logger.info("Технические индикаторы рассчитаны успешно")
        return self.data
    
    def detect_patterns(self):
        """
        Обнаружение свечных паттернов в данных.
        
        Returns:
            pd.DataFrame: Данные с добавленными паттернами
        """
        if self.data is None:
            logger.error("Данные не загружены. Сначала вызовите fetch_historical_data()")
            return None
        
        logger.info("Обнаружение свечных паттернов...")
        self.data = detect_patterns(self.data)
        return self.data
    
    def find_support_resistance(self):
        """
        Определение уровней поддержки и сопротивления.
        
        Returns:
            tuple: (уровни поддержки, уровни сопротивления)
        """
        if self.data is None:
            logger.error("Данные не загружены. Сначала вызовите fetch_historical_data()")
            return None, None
        
        logger.info("Определение уровней поддержки и сопротивления...")
        
        support_levels, resistance_levels = find_support_resistance_levels(self.data)
        
        self.support_levels = support_levels
        self.resistance_levels = resistance_levels
        
        return support_levels, resistance_levels
    
    def analyze_market_regime(self):
        """
        Анализ текущего режима рынка (тренд, боковик, волатильность).
        
        Returns:
            dict: Результаты анализа режима рынка
        """
        if self.data is None:
            logger.error("Данные не загружены. Сначала вызовите fetch_historical_data()")
            return None
        
        logger.info("Анализ режима рынка...")
        
        regime_info = detect_market_regime(self.data)
        self.market_regime = regime_info
        
        return regime_info
    
    def plot_analysis(self, save_path=None):
        """
        Визуализация результатов анализа.
        
        Args:
            save_path (str, optional): Путь для сохранения графика
            
        Returns:
            plotly.graph_objects.Figure: Объект графика
        """