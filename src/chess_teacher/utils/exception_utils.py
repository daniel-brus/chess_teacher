class ConfigError(Exception):
    pass


class AuthError(Exception):
    pass


class DatabaseError(Exception):
    pass


class PipelineError(Exception):
    pass


class MetadataError(Exception):
    pass


class AdapterError(Exception):
    pass


class FileWriteError(Exception):
    pass
