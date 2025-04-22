import sys
import yaml

if len(sys.argv) < 2:
    print("Error: Please provide config file path")
    sys.exit(1)

config_file = sys.argv[1]
try:
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    print(config['network']['type'])
    print(config['network']['arch'])
    print(config['data']['dataset'])
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)