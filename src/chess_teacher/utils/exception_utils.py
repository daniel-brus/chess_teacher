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


class AdapterClientError(AdapterError):
    """Non-retryable client-side adapter failure (e.g. HTTP 4xx except 429)."""

    pass


class FileError(Exception):
    pass


class FileWriteError(FileError):
    pass


class FileReadError(FileError):
    pass


class DataError(Exception):
    pass


class TransformationError(PipelineError):
    pass


class PipelineLockError(PipelineError):
    pass
