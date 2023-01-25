from .registry import _trainer_entrypoints, is_trainer, trainer_entrypoints


def show_available_trainers():
    """Displays available trainers"""
    print(list(trainer_entrypoints.keys()))


def build_trainer(trainer_name, **kwargs):
    if not is_trainer(trainer_name):
        raise ValueError(f'Unkown trainer: {trainer_name} not '
                         f'in {list(_trainer_entrypoints.keys())}')

    return trainer_entrypoints(trainer_name)(**kwargs)
