import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.logger import setup_logger



test_logger = setup_logger(name = 'test_logger')
test_logger.info("this is an info")
test_logger.warning("this is a warning")
test_logger.debug("this will not show at infor level")
