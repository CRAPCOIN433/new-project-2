# SOLUSDT 4h Технический Анализ

Этот проект предназначен для технического анализа торговой пары SOLUSDT на 4-часовом таймфрейме. Инструмент получает данные с биржи Binance, проводит комплексный технический анализ и визуализирует результаты.

## Возможности

- Загрузка исторических данных SOLUSDT с таймфреймом 4h
- Расчет основных технических индикаторов (MA, EMA, RSI, MACD, Bollinger Bands)
- Определение ключевых уровней поддержки и сопротивления
- Обнаружение паттернов свечей
- Визуализация графиков с индикаторами
- Генерация отчета о текущем состоянии рынка
- Оповещения о важных сигналах

## Установка

1. Клонируйте репозиторий:
git clone https://github.com/yourusername/solusdt-analysis.git
cd solusdt-analysis

2. Создайте виртуальное окружение и активируйте его:
python -m venv venv
source venv/bin/activate  # для Linux/Mac
venv\Scripts\activate     # для Windows

3. Установите зависимости:
pip install -r requirements.txt

4. Создайте файл `.env` в корневой директории проекта и добавьте ваши API ключи:
BINANCE_API_KEY=your_api_key
BINANCE_API_SECRET=your_api_secret

## Использование

Запустите основной скрипт для анализа:
python sol_analysis.py

Для запуска с пользовательским интерфейсом:
python app.py

## Примеры

Смотрите примеры использования в директории `examples/`.

## Лицензия

MIT
