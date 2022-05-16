import logging

import colorlog

handler = colorlog.StreamHandler()
handler.setFormatter(colorlog.ColoredFormatter(
    '%(log_color)s:%(name)s:%(message)s'))
logger = colorlog.getLogger('example')
logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.info("test")
