from builtins import Exception


class NoMatchingFilesException(Exception):
    def __init__(self, extension, path):
        self.extension = extension
        self.path = path

    def __str__(self):
        return f'No Matching {self.extension} in path: {self.path}'
