import sys
import json
import importlib

def import_function(import_path):
    module_name, function_name = import_path.rsplit('.', 1)
    module = importlib.import_module(module_name)
    return getattr(module, function_name)

def run_function(func, input_data):
    return func(input_data)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python generic_runner.py <input_file> <function_import_path> <deserialize_function_import_path>")
        sys.exit(1)

    input_file = sys.argv[1]
    function_import_path = sys.argv[2]
    deserialize_function_import_path = sys.argv[3]

    # Importer la fonction de désérialisation
    deserialize_func = import_function(deserialize_function_import_path)

    # Lire et désérialiser l'entrée
    with open(input_file, 'r') as f:
        serialized_input = f.read()
    input_data = deserialize_func(serialized_input)

    # Importer et exécuter la fonction principale
    main_func = import_function(function_import_path)
    result = run_function(main_func, input_data)

    # Écrire le résultat
    output_path = '/output/result.json'
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=4)