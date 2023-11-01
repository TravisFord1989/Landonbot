import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import queue
import logging
import time
from tensorflow.keras import mixed_precision
from sklearn.preprocessing import MinMaxScaler


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

    def build_model(self):
        # Enable mixed precision training
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)

        # Build the model
        model = keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=(60, 14)),
            layers.LSTM(64, return_sequences=True),  # Set to False if you only need the last output
            layers.LSTM(64),  # This layer will only return the last output
            layers.Dense(1, activation='sigmoid', dtype='float32')
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    def preprocess_data(self, df):
        # Handle missing values
        df = df.ffill().bfill()

        # Drop non-numeric columns and convert to float32 for memory efficiency
        df = df.drop(columns=["timestamp", "symbol"]).astype('float32')

        # Normalization
        df = self.scaler.fit_transform(df)
        return df

    def create_sequences(self, data, seq_length):
        sequences = []
        for i in range(len(data) - seq_length):
            seq = data[i:i + seq_length]
            label = data[i + seq_length, -1]
            sequences.append((seq, np.array([label])))
        return np.array(sequences, dtype=object)

    def train(self, data, batch_size=512):  # Add batch_size parameter
        if len(data) < 61:  # Ensure there is enough data
            logging.warning("Not enough data to create sequences for training. Data length: %s", len(data))
            return
        sequences = self.create_sequences(data, 60)
        X, y = np.array([seq[0] for seq in sequences]), np.array([seq[1] for seq in sequences])
        self.model.fit(X, y, epochs=10, batch_size=batch_size, verbose=0)

    def predict(self, data):
        if len(data) < 61:  # Ensure there is enough data
            logging.warning("Not enough data to create sequences for prediction. Data length: %s", len(data))
            return
        sequences = self.create_sequences(data, 60)
        X = np.array([seq[0] for seq in sequences])
        predictions = self.model.predict(X)
        logging.info("Predictions: %s", predictions)

    def run(self, batch_size=512):  # Add batch_size parameter
        logging.info("Starting LandonBot")
        tf.profiler.experimental.start("E:\\Documents")
        while True:
            try:
                queue_size = self.data_queue.qsize()
                logging.info("Data queue size: %d", queue_size)

                if not self.data_queue.empty():
                    new_data = pd.DataFrame(self.data_queue.get()).transpose()

                    if new_data.empty:
                        logging.warning("Received empty data")
                        continue

                    logging.info("Received new data: %s", new_data.head())

                    self.data = pd.concat([self.data, new_data])
                    self.data = self.data.drop_duplicates(subset='timestamp').sort_values(by='timestamp').reset_index(
                        drop=True)

                    logging.info("Data after processing: %s", self.data.head())

                    processed_data = self.preprocess_data(self.data)
                    self.train(processed_data, batch_size)  # Pass batch_size
                    self.predict(processed_data)

            except Exception as e:
                logging.error("Error occurred: %s", str(e))
            time.sleep(1)
        tf.profiler.experimental.stop()


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    data_queue = queue.Queue()
    bot = LandonBot(data_queue)
    bot.run(batch_size=512)  # Pass desired batch size here







