"""Import helpers for dexjoco's openpi fork without modifying it.

`openpi.training.dexjoco_configs` reads `config.yaml` from the *current working directory* at import time, so
`openpi.training.config` can only be imported from `dexjoco/openpi`. We temporarily chdir there for the import.
"""

import contextlib
import os

from lifedex.paths import OPENPI_ROOT


@contextlib.contextmanager
def _cwd(path):
    prev = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def import_openpi_config():
    with _cwd(OPENPI_ROOT):
        import openpi.training.config as _config  # noqa: PLC0415
    return _config
