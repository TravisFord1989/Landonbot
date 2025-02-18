from tensorflow.keras.layers import Dense, Reshape, LSTM
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras import mixed_precision
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras import layers
from tensorflow.keras.metrics import Mean
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.optimizers.schedules import ExponentialDecay
import pandas as pd
import numpy as np
import queue
import logging
import time
from sklearn.preprocessing import MinMaxScaler
import traceback

from tensorflow.keras.mixed_precision import set_global_policy



set_global_policy('mixed_float16')


class LandonBot:
    def __init__(self, data_queue):
        self.data_queue = data_queue
        self.model = self.build_model()
        self.scaler = MinMaxScaler()
        self.data = pd.DataFrame()
        self.column_names = [
            "timestamp", "symbol", "side", "volume", "price",
            "tick_direction", "trade_id", "is_buy_trade", "High", "Low",
            "close", "RSI", "MACD", "Bollinger Upper", "Bollinger Lower",
            "Stochastic Oscillator"
        ]
        logging.info("LandonBot initialized")
        self.check_gpu_support()

    def check_gpu_support(self):
        if tf.config.experimental.list_physical_devices('GPU'):
            policy = mixed_precision.Policy('mixed_float16')
            mixed_precision.set_global_policy(policy)

    def build_model(self, num_neurons=64, initial_learning_rate=0.001, decay_steps=10000, decay_rate=0.9,
                    num_features=16, seq_length=60):
        """
        Builds a model for regression using the given parameters.

        Args:
            num_neurons (int): Number of neurons for the Dense and LSTM layers.
            initial_learning_rate (float): Initial learning rate for the optimizer.
            decay_steps (int): Number of steps for the learning rate decay.
            decay_rate (float): Rate of decay for the learning rate.
            num_features (int): Number of features for the input shape.
            seq_length (int): Length of the sequence for the input shape.

        Returns:
            tf.keras.Model: The built model.
        """

        # Build the model
        model = keras.Sequential([
            Dense(num_neurons, activation='relu', input_shape=(seq_length, num_features)),
            # Specify the input shape here
            Reshape((seq_length, num_neurons)),
            LSTM(num_neurons, return_sequences=True),
            LSTM(num_neurons),
            Dense(1, activation='linear', dtype='float32')
        ])

        # Use AdamW optimizer with learning rate scheduler
        learning_rate_scheduler = ExponentialDecay(
            initial_learning_rate=initial_learning_rate,
            decay_steps=decay_steps,
            decay_rate=decay_rate
        )
        optimizer = AdamW(learning_rate=learning_rate_scheduler, weight_decay=0.01)

        # Use mean squared error as the loss function
        model.compile(optimizer=optimizer, loss=MeanSquaredError())

        return model

    def preprocess_data(self, df):
        # Handle missing values
        df = df.ffill().bfill()

        # Convert non-numeric columns to appropriate data types
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['symbol'] = df['symbol'].astype('category')

        # Normalization
        df[['volume', 'price', 'tick_direction', 'trade_id', 'is_buy_trade', 'High', 'Low', 'close', 'RSI', 'MACD', 'Bollinger Upper', 'Bollinger Lower', 'Stochastic Oscillator']] = self.scaler.transform(df[['volume', 'price', 'tick_direction', 'trade_id', 'is_buy_trade', 'High', 'Low', 'close', 'RSI', 'MACD', 'Bollinger Upper', 'Bollinger Lower', 'Stochastic Oscillator']])
        logging.info(f"Processing data: {df}")
        return df

    def create_sequences(self, data, seq_length):
        sequences = []
        for i in range(0, len(data) - seq_length, seq_length):
            seq = data[i:i + seq_length]
            label = data[i + seq_length, -1]
            sequences.append((seq, np.array([label])))
        return np.array(sequences, dtype=object)

    def train_model(self, data, batch_size):
        if len(data) < 61:
            logging.warning("Not enough data to create sequences for training. Data length: %s", len(data))
            return
        sequences = self.create_sequences(data, 60)
        X, y = np.array([seq[0] for seq in sequences]), np.array([seq[1] for seq in sequences])
        self.model.fit(X, y, epochs=10, batch_size=batch_size, verbose=0)

    def predict(self, data):
        if len(data) < 61:
            logging.warning("Not enough data to create sequences for prediction. Data length: %s", len(data))
            return
        sequences = self.create_sequences(data, 60)
        X = np.array([seq[0] for seq in sequences])
        predictions = self.model.predict(X)
        logging.info("Predictions: %s", predictions)

    def run(self, batch_size=512):
        logging.info("Starting LandonBot")
        tf.profiler.experimental.start("E:\\Documents")
        while True:
            queue_size = self.data_queue.qsize()
            logging.info("Data queue size: %d", queue_size)

            is_data_queue_empty = self.data_queue.empty()
            try:
                if not is_data_queue_empty:
                    new_data = self.get_new_data()

                    if new_data.empty:
                        logging.warning("Received empty data")
                        continue

                    self.process_data(new_data)

                    processed_data = self.preprocess_data(self.data)
                    self.train_model(processed_data, batch_size)
                    self.predict(processed_data)

            except ValueError as ve:
                logging.error("ValueError occurred: %s", traceback.format_exc())
                # Handle ValueError appropriately
            except KeyError as ke:
                logging.error("KeyError occurred: %s", traceback.format_exc())
                # Handle KeyError appropriately
            except Exception as e:
                logging.error("Error occurred: %s", traceback.format_exc())
                # Handle other exceptions appropriately
            time.sleep(1)
        tf.profiler.experimental.stop()

    def get_new_data(self):
        new_data = pd.DataFrame(self.data_queue.get()).transpose()

        if 'timestamp' not in new_data.columns:
            if 'time' in new_data.columns:
                new_data = new_data.rename(columns={"time": "timestamp"})
            else:
                logging.warning("Data does not contain 'timestamp' or 'time' column. Skipping...")
                return pd.DataFrame()

        return new_data

    def process_data(self, new_data):
        self.data = self.data.append(new_data)
        self.data = self.data.drop_duplicates(subset='timestamp').sort_values(by='timestamp').reset_index(drop=True)
        logging.info("Data after processing: %s", self.data.head())

    def check_dataframe_format(self):
        expected_columns = {
            'timestamp': 'datetime64[ns]',
            'symbol': 'category',
            'side': 'int8',
            'volume': 'float32',
            'price': 'float32',
            'tick_direction': 'int8',
            'trade_id': 'int64',
            'is_buy_trade': 'bool',
            'High': 'float32',
            'Low': 'float32',
            'close': 'float32',
            'RSI': 'float32',
            'MACD': 'float32',
            'Bollinger Upper': 'float32',
            'Bollinger Lower': 'float32',
            'Stochastic Oscillator': 'float32'
        }

        for col in expected_columns.keys():
            if col not in self.data.columns:
                logging.warning(f"Missing column {col} in data")

        for col in self.data.columns:
            if col not in expected_columns.keys():
                logging.warning(f"Unexpected column {col} in data")

        for col in expected_columns.keys():
            if col not in self.data.columns:
                continue
            if str(self.data[col].dtype) != expected_columns[col]:
                logging.warning(
                    f"Unexpected data type for column {col}. Expected {expected_columns[col]} but got {self.data[col].dtype}")

        missing_count = self.data.isnull().sum().sum()
        if missing_count > 0:
            logging.warning(f"Missing values detected: {missing_count} rows")

        logging.info("Dataframe format checked")
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    data_queue = queue.Queue()
    bot = LandonBot(data_queue)
    bot.run(batch_size=512)  # Pass desired batch size her
