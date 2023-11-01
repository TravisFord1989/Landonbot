import logging
import threading
import websocket
import json
import queue
import pandas as pd
from LandonBot import LandonBot
from DataProcessor import DataProcessor

logging.basicConfig(filename='E:\\Documents\\Main_logging.txt', filemode='a', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger("").addHandler(console)

data_queue = queue.Queue()
bot = LandonBot(data_queue)
bot_thread = threading.Thread(target=bot.run, kwargs={'batch_size': 512}, daemon=True)
bot_thread.start()

data_processor = DataProcessor(data_queue)

def on_open(ws):
    logging.info("Websocket opened")
    data = {"op": "subscribe", "args": ["publicTrade.BTCUSDT"]}
    ws.send(json.dumps(data))

def on_message(ws, message):
    logging.info("Received a message")
    data = json.loads(message)

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
            data_processor.processed_data_list.append(processed_data)  # Add to processed_data_list in DataProcessor

        if processed_data_list:
            data_processor.on_message()  # Call on_message method of DataProcessor
            new_data_df = pd.DataFrame(processed_data_list)
            data_df = pd.concat([data_df, new_data_df], ignore_index=True)
            calculate_technical_indicators(data_df)  # Call the function
            data_queue.put(new_data_df.iloc[-1])  # Place the new data into the queue
            processed_data_list.clear()  # Clear the list after processing

def on_error(ws, error):
    logging.error(f"Error occurred: {error}")

def on_close(ws, close_status_code, close_msg):
    logging.info(f"Websocket Closed: {close_status_code}, {close_msg}")

# Initialize WebSocket
endpoint = "wss://stream.bybit.com/v5/public/linear"
ws = websocket.WebSocketApp(endpoint, on_open=on_open, on_message=on_message, on_error=on_error, on_close=on_close)
ws_thread = threading.Thread(target=ws.run_forever, daemon=True)
ws_thread.start()

if __name__ == "__main__":
    bot_thread.join()
    ws_thread.join()