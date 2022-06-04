from typing import Dict


def convert_arch2dict(arch: str) -> Dict:
    """Convert the arch encoding to subnet config.

    Args:
        arch (str): arch config with 14 chars.

    Returns:
        Dict: subnet config.
    """
    assert len(arch) == 14

    specific_subnet = {}
    for c, id in zip(arch, list(range(14))):
        specific_subnet[id] = c
    return specific_subnet
