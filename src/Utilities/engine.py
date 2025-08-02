import sqlalchemy as sa
from dotenv import load_dotenv
import os

load_dotenv()

host = os.getenv('HOST')
port = int(os.getenv('DB_PORT'))
db = os.getenv('DB_NAME')
user = os.getenv('DB_USER')
password = os.getenv('DB_PASSWORD')
engine = sa.create_engine(f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}")