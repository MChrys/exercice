import sys
import json
import importlib
import os
import logging
from workflows.pipeline import serialize, unserialize
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def import_function(import_path):
    module_name, function_name = import_path.rsplit('.', 1)
    module = importlib.import_module(module_name)
    return getattr(module, function_name)

def run_function(func, input_data):
    return func(input_data)

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print(sys.argv)
        print("Usage: python generic_runner.py <input_file> <function_import_path> <deserialize_function_import_path>")
        sys.exit(1)

    input_name = sys.argv[1]
    value_path = sys.argv[2]
    function_import_path = sys.argv[3]
    deserialize_function_import_path = sys.argv[4]

    logger.info(f"input_file_path: {value_path}")
    logger.info(f"function_import_path: {function_import_path}")
    logger.info(f"deserialize_function_import_path: {deserialize_function_import_path}")

    deserialize_func = import_function(deserialize_function_import_path)
    logger.info(f"deserialize_func: {deserialize_func}")
    logger.info(f"input_path: {os.path.join(value_path, input_name)}")
    input_data = deserialize_func(os.path.join(value_path, input_name))
    logger.info(f"input_data: {input_data}")
    #input_data = deserialize_func(serialized_input)

    main_func = import_function(function_import_path)
    logger.info(f"main_func: {main_func}")
    result = run_function(main_func, input_data)
    logger.info(f"result: {len(result)}")

    serialized_result = serialize(result,"output",path=value_path)
    logger.info(f"{serialized_result}")
