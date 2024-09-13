import logging
import os
import yaml
from torch import nn

def read_yaml_file(file_path):
    """
    Reads a YAML file and returns the content as a dictionary.

    Parameters:
    file_path (str): The path to the YAML file to read.

    Returns:
    dict: The content of the YAML file as a dictionary.
    """
    with open(file_path, 'r') as file:
        try:
            content = yaml.safe_load(file)  # Load the YAML file content
            return content
        except yaml.YAMLError as e:
            print(f"Error reading YAML file: {e}")
            return None


class Logger:
    def __init__(self, save_dir):
        self.logger = logging.getLogger(__name__)
    def __init__(self, save_dir):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # Create handlers
        console_handler = logging.StreamHandler()
        
        # Create the directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(save_dir, "logfile.log"))

        # Create formatters and add it to handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        # Add handlers to the logger
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(file_handler)

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def debug(self, message):
        self.logger.debug(message)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



def get_one_hot(inp_seg, num_labels):
    B, C, H, W, D = inp_seg.shape
    inp_onehot = nn.functional.one_hot(inp_seg.long(), num_classes=num_labels)
    inp_onehot = inp_onehot.squeeze(dim=1)
    inp_onehot = inp_onehot.permute(0, 4, 1, 2, 3).contiguous()
    return inp_onehot

