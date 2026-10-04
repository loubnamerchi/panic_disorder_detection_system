

from __future__ import annotations

import logging
import warnings



def get_logger(name: str) -> logging.Logger:
    if logging.getLogger().handlers:
        return logging.getLogger(name)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    # silence optuna logs only if optuna is installed (training env)
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        pass

    warnings.filterwarnings("ignore", category=UserWarning)

    return logging.getLogger(name)
