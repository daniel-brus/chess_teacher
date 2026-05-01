import os

from dotenv import load_dotenv


class Configuration:
    def __init__(self):
        load_dotenv("config.env")


class Secrets:
    def __init__(self):
        load_dotenv(".env")
        self.DATABASE_URL = os.getenv("DATABASE_URL")


secrets = Secrets()
configuration = Configuration()
