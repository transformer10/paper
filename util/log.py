import colorlog
import logging


def get_logger(name='log'):
    logger = colorlog.getLogger(name)
    handler = colorlog.StreamHandler()
    handler.setFormatter(colorlog.ColoredFormatter(
        # '%(log_color)s%(name)s: %(message)s'
        '%(log_color)s%(message)s', log_colors={
            'DEBUG': 'blue',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white'
        }))
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    return logger
