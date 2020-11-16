import os


# TODO: file needs general refactor
def extract(path):
    if os.path.isdir(path):
        res = []
        contents = os.listdir(path)
        for cont in contents:
            res.append(extract(os.path.join(path, cont)))
        return res
    return path


def generate_abs_for_dir(dir_path, abs_paths):
    for path in os.listdir(dir_path):
        if "-" not in path:
            joined = os.path.join(dir_path, path)
            if os.path.isdir(joined):
                abs_paths.extend(extract(joined))
            else:
                abs_paths.append(extract(joined))
    return abs_paths


def generate_absolute_paths(paths):
    abs_paths = []
    for path in paths:
        abs_path = os.path.abspath(path)
        if os.path.isdir(abs_path):
            abs_paths = generate_abs_for_dir(abs_path, abs_paths)
        else:
            abs_paths.append(abs_path)

    return abs_paths


def parent_path(path):
    return os.path.abspath(os.path.join(path, os.pardir))
