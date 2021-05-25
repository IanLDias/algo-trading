from configparser import ConfigParser
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
API_KEY_NOMICS = 'c3f2b1307b497fba9c7b97ae18e6e63d'
BASE_URL_NOMICS = 'https://api.nomics.com/v1'

#cryptocompare
API_KEY_COMP = '17666b6cf3df20d3d971212b0646e2a15447b36bf7b7a4974283778402417772'
BASE_URL_COMP = 'https://min-api.cryptocompare.com/'