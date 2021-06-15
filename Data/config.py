from configparser import ConfigParser
import config
import pathlib

parser = ConfigParser()
current_path = pathlib.Path(__file__).parent.absolute()

def config(filename=str(current_path)+'/database.ini', section='postgresql'):
    parser = ConfigParser()
    parser.read(filename)

    db = {}

    if parser.has_section(section):
        params = parser.items(section)
        for param in params:
            db[param[0]] = param[1]
    
    else:
        raise Exception(f"Section {section} not found in {filename}")
    
    return db

if __name__ == '__main__':
    print(config())

#---Cryptocurrency---
#nomics
API_KEY_NOMICS = config.API_KEY_NOMICS
BASE_URL_NOMICS = 'https://api.nomics.com/v1'

#cryptocompare
API_KEY_COMP = config.API_KEY_COMP
BASE_URL_COMP = 'https://min-api.cryptocompare.com/'