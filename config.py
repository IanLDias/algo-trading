from configparser import ConfigParser
import config_vars
import pathlib

parser = ConfigParser()
current_path = pathlib.Path(__file__).parent.absolute()

def config(filename=str(current_path)+'/Data/database.ini', section='postgresql'):
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
