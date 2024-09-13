from datetime import datetime
import os

from .utils import Logger

current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
os.makedirs(f"./logs/{current_time}", exist_ok=True)
logger = Logger(f"./logs/{current_time}")

checkpoints_dir = f"./checkpoints/{current_time}"
os.makedirs(checkpoints_dir, exist_ok=True)

