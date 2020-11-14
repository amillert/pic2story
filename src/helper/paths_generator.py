import os


def generate_absolute_paths(paths):
    def extract(path):
        if os.path.isdir(path):
            res = []
            contents = os.listdir(path)
            for c in contents:
                res.append(extract(os.path.join(path, c)))
            return res
        else:
            return path

    abs_paths = []
    for path in paths:
        p = os.path.abspath(path)
        if os.path.isdir(p):
            for x in os.listdir(p):
                if "-" not in x:
                    xd = os.path.join(p, x)
                    if os.path.isdir(xd):
                        abs_paths.extend(extract(xd))
                    else:
                        abs_paths.append(extract(xd))
        else: abs_paths.append(p)

    return abs_paths


def parent_path(path):
    return os.path.abspath(os.path.join(path, os.pardir))
