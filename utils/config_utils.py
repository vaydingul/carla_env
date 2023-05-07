import yaml


def parse_yml(yml_path):
	with open(yml_path, 'r') as f:
		return yaml.load(f, Loader=yaml.FullLoader)

def copy_config_to_path(config_dict, path):

	with open(path, 'w') as f:
		yaml.dump(config_dict, f)
	
	

if __name__ == '__main__':
	yml_path = 'utils/config.yml'
	config = parse_yml(yml_path)
	print(config)