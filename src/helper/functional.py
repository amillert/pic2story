def flatten(xs):
    return (xi for x in xs for xi in x)


def unique(xs):
    return list(set(xs))
