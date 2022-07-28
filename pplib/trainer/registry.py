_trainer_entrypoints = {}


def register_trainer(fn):
    _trainer_entrypoints[fn.__name__] = fn
    return fn


def trainer_entrypoints(trainer_name):
    return _trainer_entrypoints[trainer_name]


def is_trainer(trainer_name):
    return trainer_name in _trainer_entrypoints
