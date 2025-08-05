"""
Initializes the database connection for the orion context broker.
"""
import os
from ngsildclient import Client
from dotenv import load_dotenv
load_dotenv()

host = os.getenv('HOST')

# Connect to Context Broker
client = Client(hostname=host,port=1026)