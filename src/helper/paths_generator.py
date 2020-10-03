import os


def generate_absolute_paths(paths):
    abs_paths = [os.path.abspath(path) if not os.path.isabs(path) else path for path in paths]
    return [[os.path.join(abs_path, pic) for pic in os.listdir(abs_path)]
            if os.path.isdir(abs_path) else abs_path for abs_path in abs_paths]


def parent_path(path):
    return os.path.abspath(os.path.join(path, os.pardir))
