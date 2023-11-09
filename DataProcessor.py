import pandas as pd
import websocket
import json
import schedule
import time
import talib
import logging
import threading
import numpy as np
from pybit.unified_trading import HTTP
from queue import Queue

# Setting up logging
logging.basicConfig(filename='app.log', filemode='w', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

expected_columns = {
    'timestamp': 'datetime64[ns]',
    'symbol': 'object',
    'side': 'float64',
    'volume': 'float64',
    'price': 'float64',
    'tick_direction': 'object',
    'trade_id': 'object',
    'is_buy_trade': 'object'
}

expected_columns = {
    'timestamp': 'datetime64[ns]',
    'symbol': 'object',
    'side': 'int64',
    'volume': 'float64',
    'price': 'float64',
    'tick_direction': 'int64',
    'trade_id': 'int64',
    'is_buy_trade': 'bool',
}


class DataProcessor:
    def __init__(self, data_queue, window_size=50):
        self.data_queue = data_queue
        self.window_size = window_size
        self.data_df = pd.DataFrame(columns=expected_columns.keys()).astype(expected_columns)
        self.processed_data_list = []
        self.technical_indicators_calculated = False
        logging.getLogger().setLevel(logging.INFO)

    def process_message(self, message):
        try:
            if not isinstance(message, (pd.DataFrame, pd.Series)):
                logging.error(f"Expected message to be a DataFrame or Series, but got {type(message)}")
                if isinstance(message, dict):
                    message = pd.DataFrame([message])
                    for col in expected_columns:
                        if col not in message.columns:
                            message[col] = np.nan

            # At this point, 'message' must be a DataFrame, append to list
            self.processed_data_list.append(message)

            # When enough data is collected
            if len(self.processed_data_list) >= self.window_size:
                self.data_df = pd.concat(self.processed_data_list).astype(expected_columns)
                self.processed_data_list = []
                self.calculate_technical_indicators()

                # Handling non-finite values and ensuring correct data types
                self.data_df['side'] = pd.to_numeric(self.data_df['side'], errors='coerce').fillna(0).astype(int)
                self.data_df['volume'] = pd.to_numeric(self.data_df['volume'], errors='coerce')
                self.data_df['tick_direction'] = pd.to_numeric(self.data_df['tick_direction'], errors='coerce').fillna(
                    0).astype(int)
                self.data_df['trade_id'] = pd.to_numeric(self.data_df['trade_id'], errors='coerce').fillna(0).astype(
                    int)
                self.data_df['is_buy_trade'] = self.data_df['is_buy_trade'].astype(bool)

            self.data_queue.put(self.data_df.tail(1))
            logging.info("Data enqueued. Queue size: %d", self.data_queue.qsize())
            logging.debug("Enqueued data: %s", message)

            if self.technical_indicators_calculated:
                if self.data_df['RSI'].iloc[-1] > 70:
                    logging.info(f"Overbought RSI: {self.data_df['RSI'].iloc[-1]}")
                elif self.data_df['RSI'].iloc[-1] < 30:
                    logging.info(f"Oversold RSI: {self.data_df['RSI'].iloc[-1]}")
                if self.data_df['MACD'].iloc[-1] > self.data_df['MACDSIGNAL'].iloc[-1]:
                    logging.info(
                        f"MACD crossover: {self.data_df['MACD'].iloc[-1]} (MACD) > {self.data_df['MACDSIGNAL'].iloc[-1]} (Signal)")
                elif self.data_df['MACD'].iloc[-1] < self.data_df['MACDSIGNAL'].iloc[-1]:
                    logging.info(
                        f"MACD crossunder: {self.data_df['MACD'].iloc[-1]} (MACD) < {self.data_df['MACDSIGNAL'].iloc[-1]} (Signal)")
                if self.data_df['price'].iloc[-1] > self.data_df['Bollinger Upper'].iloc[-1]:
                    logging.info(
                        f"Price above upper Bollinger Band: {self.data_df['price'].iloc[-1]} > {self.data_df['Bollinger Upper'].iloc[-1]}")
                elif self.data_df['price'].iloc[-1] < self.data_df['Bollinger Lower'].iloc[-1]:
                    logging.info(
                        f"Price below lower Bollinger Band: {self.data_df['price'].iloc[-1]} < {self.data_df['Bollinger Lower'].iloc[-1]}")
                if self.data_df['Stochastic K'].iloc[-1] > 80:
                    logging.info(f"Stochastic Overbought: %K = {self.data_df['Stochastic K'].iloc[-1]}")
                elif self.data_df['Stochastic K'].iloc[-1] < 20:
                    logging.info(f"Stochastic Oversold: %K = {self.data_df['Stochastic K'].iloc[-1]}")
                if self.data_df['Stochastic K'].iloc[-1] > self.data_df['Stochastic D'].iloc[-1]:
                    logging.info(
                        f"Stochastic Crossover: %K = {self.data_df['Stochastic K'].iloc[-1]} > %D = {self.data_df['Stochastic D'].iloc[-1]}")
                elif self.data_df['Stochastic K'].iloc[-1] < self.data_df['Stochastic D'].iloc[-1]:
                    logging.info(
                        f"Stochastic Crossunder: %K = {self.data_df['Stochastic K'].iloc[-1]} < %D = {self.data_df['Stochastic D'].iloc[-1]}")


        except AttributeError as e:
            logging.error(f"AttributeError in process_message: {e}")
        except Exception as e:
            logging.error(f"Unexpected error in process_message: {e}")
        

    def fetch_bybit_data(self):
        try:
            session = HTTP(testnet=True)
            data = session.get_kline(category="inverse", symbol="BTCUSDT", interval="1", limit=2000)
            logging.info("Fetched historical data")

            self.processed_data_list.append(pd.DataFrame(data))
            self.calculate_technical_indicators()
        except Exception as e:
            logging.error(f"Error fetching Bybit data: {e}")

    def calculate_technical_indicators(self):
        try:
            window_size = 50  # Define window_size here
            if len(self.data_df) >= self.window_size:  # Use self.window_size instead of window_size
                close = self.data_df['price'].astype(float).values[-self.window_size:]  # Use self.window_size
                high = self.data_df['price'].astype(float).values[-self.window_size:]  # Use self.window_size
                low = self.data_df['price'].astype(float).values[-self.window_size:]  # Use self.window_size

                rsi = talib.RSI(close, timeperiod=14)
                macd, macd_signal, _ = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
                bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
                slowk, slowd = talib.STOCH(high, low, close, fastk_period=5, slowk_period=3, slowk_matype=0,
                                           slowd_period=3, slowd_matype=0)

                self.data_df['RSI'].iloc[-window_size:] = rsi
                self.data_df['MACD'].iloc[-window_size:] = macd
                self.data_df['MACDSIGNAL'].iloc[-window_size:] = macd_signal
                self.data_df['Bollinger Upper'].iloc[-window_size:] = bb_upper
                self.data_df['Bollinger Middle'].iloc[-window_size:] = bb_middle
                self.data_df['Bollinger Lower'].iloc[-window_size:] = bb_lower
                self.data_df['Stochastic K'].iloc[-window_size:] = slowk
                self.data_df['Stochastic D'].iloc[-window_size:] = slowd

                self.technical_indicators_calculated = True
        except Exception as e:
            logging.error(f"Error calculating technical indicators: {e}")

    def on_open(self, ws):
        try:
            logging.info("Websocket opened")
            data = {"op": "subscribe", "args": ["publicTrade.BTCUSDT"]}
            ws.send(json.dumps(data))
        except Exception as e:
            logging.error(f"Error opening WebSocket: {e}")

    def check_dataframe_format(self):
        if not isinstance(self.data_df, pd.DataFrame):
            logging.error(f"self.data_df is not a DataFrame: {type(self.data_df)}")
            return

        missing_values = self.data_df[expected_columns.keys()].isnull().sum()
        for col, missing_count in missing_values.items():
            if missing_count > 0:
                logging.warning(f"Missing values detected in column {col}: {missing_count} rows")

        for col, expected_type in expected_columns.items():
            if col in self.data_df.columns:
                col_type = self.data_df[col].dtype
                if not isinstance(col_type, np.dtype):  # Check if col_type is not a numpy dtype object
                    logging.error(f"The dtype of column {col} is not valid: {type(self.data_df[col])}")
                    continue  # Skip this iteration
                if str(col_type) != expected_type:
                    logging.warning(
                        f"Unexpected data type for column {col}. Expected {expected_type} but got {col_type}")

    def on_message(self, ws, message):
        try:
            data = json.loads(message)
            logging.info("Received a message")

            self.process_data_message(data)

            self.check_dataframe_format()

        except json.JSONDecodeError as e:
            logging.error(f"Error decoding JSON: {e}")
        except KeyError as e:
            logging.error(f"Missing expected key in data: {e}")
        except Exception as e:
            logging.error(f"Error processing WebSocket message: {e}")

    def process_data_message(self, data):
        try:
            if 'topic' in data and data['topic'] == 'publicTrade.BTCUSDT':
                for trade_data in data['data']:
                    # Log the type of trade_data to ensure it's what we expect (should be a dict)
                    logging.debug(f"Type of trade_data: {type(trade_data)}")

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
                    # Log the type of processed_data to ensure it's a dict
                    logging.debug(f"Type of processed_data: {type(processed_data)}")
                    self.process_message(processed_data)
        except AttributeError as e:
            logging.error(f"AttributeError in process_data_message: {e}")
        except Exception as e:
            logging.error(f"Unexpected error in process_data_message: {e}")

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
        if len(self.data_df) < 1000:
            self.fetch_bybit_data()
        schedule.every(30).minutes.do(self.fetch_bybit_data)

        endpoint = "wss://stream.bybit.com/v5/public/linear"
        ws = websocket.WebSocketApp(endpoint, on_open=self.on_open, on_message=self.on_message, on_error=self.on_error,
                                    on_close=self.on_close)

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



