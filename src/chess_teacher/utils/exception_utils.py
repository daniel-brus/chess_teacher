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


class FileError(Exception):
    pass


class FileWriteError(FileError):
    pass


class FileReadError(FileError):
    pass
