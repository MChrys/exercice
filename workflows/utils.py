import subprocess
from opentelemetry import trace
def print_directory_tree(startpath):
    """Print the directory tree"""
    try:
        result = subprocess.run(['tree', startpath], capture_output=True, text=True)
        logger.info(f"Arborescence  {startpath}:\n{result.stdout}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error tree: {e}")

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class logx():
    def __init__(self,force_record=False):
        self.force_record = force_record
        if (span:=trace.get_current_span()).is_recording() or force_record: 
            self.span = span
            self.record = True
        else:
            self.record = False

    def __call__(self, message, level="info"):
        if self.record:
            self.span.add_event(message)
        else:
            getattr(logger, level)(message)

    def __getattr__(self, name):
        if self.record:
            if self.force_record:
                return getattr(logger, name)
            else:
                return self.span.add_event
        else:
            return getattr(logger, name)