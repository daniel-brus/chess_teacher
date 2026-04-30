import os

from dotenv import load_dotenv


class Configuration:
    def __init__(self):
        load_dotenv("config.env")


class Secrets:
    def __init__(self):
        load_dotenv(".env")
        self.DOCKERHUB_USERNAME = os.getenv("DOCKERHUB_USERNAME")
        self.DOCKERHUB_TOKEN = os.getenv("DOCKERHUB_TOKEN")


secrets = Secrets()
configuration = Configuration()
