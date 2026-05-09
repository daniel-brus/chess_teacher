import hashlib


def generate_hash(input: str | list[str]) -> str:
    """Generate a sha256hash for the given input string."""
    if isinstance(input, list):
        input_string = ",".join(input)
    else:
        input_string = input

    return hashlib.sha256(input_string.encode()).hexdigest()
