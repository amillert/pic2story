"""
Module providing some helper functions
"""
import os


# TODO: file needs general refactor
def extract(path):
    """
    Function for extracting path or paths if directory

    :param path: str
    :return: str / list[str]  # TODO: fix
    """
    if os.path.isdir(path):
        res = []
        contents = os.listdir(path)
        for cont in contents:
            res.append(extract(os.path.join(path, cont)))
        return res
    return path


def generate_abs_for_dir(dir_path, abs_paths):
    """
    Function for generating absolute paths in the directory

    :param dir_path: str
    :param abs_paths: list[str]
    :return:
    """
    for path in os.listdir(dir_path):
        if "-" not in path:
            joined = os.path.join(dir_path, path)
            if os.path.isdir(joined):
                abs_paths.extend(extract(joined))
            else:
                abs_paths.append(extract(joined))
    return abs_paths


def generate_absolute_paths(paths):
    """
    Function for generating paths

    :param paths: list[str]
    :return: list[list[str]] ?
    """
    abs_paths = []
    for path in paths:
        abs_path = os.path.abspath(path)
        if os.path.isdir(abs_path):
            abs_paths = generate_abs_for_dir(abs_path, abs_paths)
        else:
            abs_paths.append(abs_path)

    return abs_paths


def parent_path(path):
    """
    Function for easier access to the parent directory

    :param path: str
    :return: str
    """
    return os.path.abspath(os.path.join(path, os.pardir))
