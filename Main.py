import logging
import threading
import queue
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
data_processor_thread = threading.Thread(target=data_processor.run, daemon=True)
data_processor_thread.start()

if __name__ == "__main__":
    bot_thread.join()
    data_processor_thread.join()
