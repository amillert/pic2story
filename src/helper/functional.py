class Functional:
    @staticmethod
    def flatten(xs):
        return (xi for x in xs for xi in x)

    @staticmethod
    def unique(xs):
        return list(set(xs))
