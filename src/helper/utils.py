import os
import pickle


def save_obj(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(path):
    if os.path.isdir(path):
        for c in os.listdir(path):
            return load_obj(os.path.abspath(os.path.join(path, c)))
    else:
        with open(path, "rb") as f:
            return pickle.load(f)
