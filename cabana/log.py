"""Lightweight logging facade.

Replaces the previous hand-rolled rotating-log singleton with a thin wrapper
around stdlib :mod:`logging`. The public surface (``Log.logger``,
``Log.init_log_path(path)``, ``Log.log_parameters(param_path)``) is
preserved so existing call sites do not need to change.
"""

import os
import time
import logging
import logging.handlers

import yaml
import colorlog


logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)


_LOG_COLORS = {
    'DEBUG': 'green',
    'INFO': 'white',
    'WARNING': 'yellow',
    'ERROR': 'red',
    'CRITICAL': 'red',
}
_FORMATTER = colorlog.ColoredFormatter(
    '%(log_color)s[%(levelname)s]- %(message)s', log_colors=_LOG_COLORS)


def _build_logger():
    """Return the root cabana logger with a console handler attached.

    Always usable from import time so callers can log without first calling
    :func:`init_log_path`.
    """
    logger = logging.getLogger()
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(_FORMATTER)
    logger.addHandler(sh)
    return logger


class _LogFacade:
    """Module-level singleton providing the historical ``Log`` API."""

    def __init__(self):
        self.log_path = None
        self.log_name = None
        self.logger = _build_logger()

    def init_log_path(self, log_path=None):
        """Add a rotating file handler beside the existing stream handler."""
        if log_path:
            self.log_path = log_path
        else:
            self.log_path = os.path.join(
                os.path.dirname(os.path.realpath(__file__)), 'logs')
        os.makedirs(self.log_path, exist_ok=True)

        self.log_name = os.path.join(
            self.log_path, '%s.log' % time.strftime('%Y-%m-%d-%H-%M-%S'))

        fh = logging.handlers.RotatingFileHandler(
            filename=self.log_name, mode='a',
            maxBytes=5 * 1024 * 1024, backupCount=5, encoding='utf-8')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(_FORMATTER)
        self.logger.addHandler(fh)

    def log_parameters(self, param_path):
        if not os.path.exists(param_path):
            self.logger.error("{} not exists.".format(param_path))
            return
        header = "********{}********".format(os.path.basename(param_path))
        self.logger.info(header)
        with open(param_path, 'r') as pf:
            data = yaml.safe_load(pf)
            for line in yaml.dump(data, default_flow_style=False).split("\n"):
                self.logger.info(line)
        footer = '*' * ((len(header) - 3) // 2) + "End" + '*' * ((len(header) - 3) // 2)
        self.logger.info(footer)


Log = _LogFacade()
