import uuid
import requests
import json
import time
from datetime import datetime, timezone, timedelta
from ngsildclient import Client, Entity, iso8601
from Generic.Global.Borg import Borg
from System.App.Entity.Entity import Entity
from System.util.date import parse_date


class OrionManager(Borg):

    def __init__(self,
                 client_id: str,
                 client_secret: str) -> None:
        """
        Class builder, all the contextual configurations are charged from the base class (borg pattern) in a shared state
        this class is in charge of creating entities in the Orion-LD context broker

        Args:
            client_id: str
            client_secret: str
        Returns:
            [None]: None
        """
        # Setting contextual generic objects
        super().__init__()
        self.ctx['__obj']['__log'].setLog('Initializing OrionManager')
        self.client = Client(hostname="localhost",port=1026)
        # Load configuration
        self.ctx['__obj']['__log'].setLog('OrionManager initialized')



    def create_videoRecorded_entity(self, video: dict) -> dict:
        """
        Create a videoRecorded entity.

        Args:
            video: dict
        Returns:
            dict: Entity data
        """
        self.ctx['__obj']['__log'].setLog('Creating videoRecorded entity')

        e = Entity("videoRecorded", f"{video['camera']}:{video['id']}")
        e.prop("path", video["path"])
        e.prop("inferred", False)
        e.prop("dateObserved", video["dateObserved"])
        self.client.upsert(e)
        self.ctx['__obj']['__log'].setLog('videoRecorded entity created')


