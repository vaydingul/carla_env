import yaml


def parse_yml(yml_path):
	with open(yml_path, 'r') as f:
		return yaml.load(f, Loader=yaml.FullLoader)



if __name__ == '__main__':
	yml_path = 'utils/config.yml'
	config = parse_yml(yml_path)
	print(config)