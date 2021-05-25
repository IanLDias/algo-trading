import psycopg2
from config import config

params = config()

conn = psycopg2.connect(**params)
cur = conn.cursor()
