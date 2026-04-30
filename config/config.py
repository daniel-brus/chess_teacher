from dotenv import load_dotenv


class Configuration:
    def __init__(self):
        load_dotenv("config.env")


class Secrets:
    def __init__(self):
        load_dotenv(".env")


secrets = Secrets()
configuration = Configuration()
