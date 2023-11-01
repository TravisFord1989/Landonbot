import pandas as pd
import websocket
import json
import schedule
import time
import talib
import logging
import threading
from pybit.unified_trading import HTTP
from queue import Queue

# Setting up logging
logging.basicConfig(filename='app.log', filemode='w', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)


class DataProcessor:
    def __init__(self, data_queue):
        self.data_df = pd.DataFrame()
        self.processed_data_list = []  # List to store processed data before converting to DataFrame
        self.data_queue = data_queue

    def process_message(self, processed_data):
        # Convert the single trade data to a DataFrame
        trade_df = pd.DataFrame([processed_data])

        # Concatenate the new data to the existing DataFrame
        self.data_df = pd.concat([self.data_df, trade_df], ignore_index=True)

        # Calculate technical indicators if enough data is available
        self.calculate_technical_indicators()

        # Process RSI
        if 'RSI' in self.data_df.columns:
            last_rsi = self.data_df['RSI'].iloc[-1]
            if last_rsi > 70:
                logging.info(f"Overbought RSI: {last_rsi}")
            elif last_rsi < 30:
                logging.info(f"Oversold RSI: {last_rsi}")

        # Process MACD
        if 'MACD' in self.data_df.columns and 'MACDSIGNAL' in self.data_df.columns:
            last_macd = self.data_df['MACD'].iloc[-1]
            last_macd_signal = self.data_df['MACDSIGNAL'].iloc[-1]
            if last_macd > last_macd_signal:
                logging.info(f"MACD crossover: {last_macd} (MACD) > {last_macd_signal} (Signal)")
            elif last_macd < last_macd_signal:
                logging.info(f"MACD crossunder: {last_macd} (MACD) < {last_macd_signal} (Signal)")

        # Process Bollinger Bands
        if 'Bollinger Upper' in self.data_df.columns and 'Bollinger Lower' in self.data_df.columns:
            last_close = self.data_df['price'].iloc[-1]
            last_upper_band = self.data_df['Bollinger Upper'].iloc[-1]
            last_lower_band = self.data_df['Bollinger Lower'].iloc[-1]
            if last_close > last_upper_band:
                logging.info(f"Price above upper Bollinger Band: {last_close} > {last_upper_band}")
            elif last_close < last_lower_band:
                logging.info(f"Price below lower Bollinger Band: {last_close} < {last_lower_band}")

        # Process Stochastic
        if 'Stochastic K' in self.data_df.columns and 'Stochastic D' in self.data_df.columns:
            last_stoch_k = self.data_df['Stochastic K'].iloc[-1]
            last_stoch_d = self.data_df['Stochastic D'].iloc[-1]
            if last_stoch_k > 80:
                logging.info(f"Stochastic Overbought: %K = {last_stoch_k}")
            elif last_stoch_k < 20:
                logging.info(f"Stochastic Oversold: %K = {last_stoch_k}")
            if last_stoch_k > last_stoch_d:
                logging.info(f"Stochastic Crossover: %K = {last_stoch_k} > %D = {last_stoch_d}")
            elif last_stoch_k < last_stoch_d:
                logging.info(f"Stochastic Crossunder: %K = {last_stoch_k} < %D = {last_stoch_d}")

        # Optionally, you can also trim the DataFrame to keep only recent data in memory
        if len(self.data_df) > 2000:
            self.data_df = self.data_df.iloc[-2000:]

    def fetch_bybit_data(self):
        try:
            session = HTTP(testnet=True)
            data = session.get_kline(category="inverse", symbol="BTCUSDT", interval="1", limit=2000)
            logging.info("Fetched historical data")

            df = pd.DataFrame(data)
            self.data_df = pd.concat([self.data_df, df], ignore_index=True)
            self.calculate_technical_indicators()
        except Exception as e:
            logging.error(f"Error fetching Bybit data: {e}")

    def calculate_technical_indicators(self):
        try:
            window_size = 50
            if len(self.data_df) >= window_size:
                close = self.data_df['price'].astype(float).values
                high = self.data_df['price'].astype(float).values
                low = self.data_df['price'].astype(float).values

                self.data_df['RSI'] = talib.RSI(close, timeperiod=14)
                self.data_df['MACD'], self.data_df['MACDSIGNAL'], _ = talib.MACD(close, fastperiod=12, slowperiod=26,
                                                                                 signalperiod=9)
                self.data_df['Bollinger Upper'], self.data_df['Bollinger Middle'], self.data_df[
                    'Bollinger Lower'] = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
                slowk, slowd = talib.STOCH(high, low, close, fastk_period=5, slowk_period=3, slowk_matype=0,
                                           slowd_period=3, slowd_matype=0)
                self.data_df['Stochastic K'] = slowk
                self.data_df['Stochastic D'] = slowd
        except Exception as e:
            logging.error(f"Error calculating technical indicators: {e}")

    def on_open(self, ws):
        try:
            logging.info("Websocket opened")
            data = {"op": "subscribe", "args": ["publicTrade.BTCUSDT"]}
            ws.send(json.dumps(data))
        except Exception as e:
            logging.error(f"Error opening WebSocket: {e}")

    def on_message(self, ws, message):
        try:
            data = json.loads(message)
            logging.info("Received a message")

            if 'topic' in data and data['topic'] == 'publicTrade.BTCUSDT':
                for trade_data in data['data']:
                    timestamp = trade_data['T']
                    processed_data = {
                        'timestamp': pd.to_datetime(timestamp, unit='ms'),
                        'symbol': trade_data['s'],
                        'side': 1 if trade_data['S'] == "Buy" else 0,
                        'volume': trade_data['v'],
                        'price': float(trade_data['p']),
                        'tick_direction': trade_data['L'],
                        'trade_id': trade_data['i'],
                        'is_buy_trade': trade_data['BT']
                    }
                    self.processed_data_list.append(processed_data)

                self.process_message(processed_data)  # Call the process_message method of the current instance

        except json.JSONDecodeError as e:
            logging.error(f"Error decoding JSON: {e}")
        except KeyError as e:
            logging.error(f"Missing expected key in data: {e}")
        except Exception as e:
            logging.error(f"Error processing WebSocket message: {e}")

    def on_error(self, ws, error):
        logging.error(f"WebSocket error: {error}")

    def on_close(self, ws, close_status_code, close_msg):
        logging.info(f"WebSocket Closed: {close_status_code}, {close_msg}")

    def save_data(self):
        try:
            logging.info("Saving data before exit...")
            current_date = pd.to_datetime("now").strftime('%Y_%m_%d_%H_%M_%S')
            file_path = f'E:\\Documents\\bybit_data_{current_date}.csv'
            self.data_df.to_csv(file_path, index=False)
            logging.info(f"Saved data to {file_path}")
        except Exception as e:
            logging.error(f"Error saving data: {e}")

    def run(self):
        self.fetch_bybit_data()
        schedule.every(30).minutes.do(self.fetch_bybit_data)

        endpoint = "wss://stream.bybit.com/v5/public/linear"
        ws = websocket.WebSocketApp(endpoint, on_open=self.on_open, on_message=self.on_message, on_error=self.on_error, on_close=self.on_close)

        ws_thread = threading.Thread(target=ws.run_forever)
        ws_thread.start()

        try:
            while True:
                schedule.run_pending()
                time.sleep(1)
        except KeyboardInterrupt:
            ws.close()
            ws_thread.join()

if __name__ == "__main__":
    data_queue = Queue()
    processor = DataProcessor(data_queue)
    processor.run()



