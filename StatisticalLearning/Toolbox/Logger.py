import sys
import logging

# Default logging format parameters
_LOG_FORMAT = '%(asctime)s %(levelname)-8s: %(message)s'
_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
_LEVEL = {'info': logging.INFO, 'warning': logging.warning, 'debug': logging.debug, 'error': logging.error}


class Logger:

    """
    A convenient logger tool to provide standard logging interface
    """

    @ staticmethod
    def get_logger(level: str = 'info',
                   log_format: str = _LOG_FORMAT,
                   date_format: str = _DATE_FORMAT) -> logging.Logger:
        """
        Output logging info onto python console

        :param level: str, logging output level {'debug', 'info', 'warning', 'error'}
        :param log_format: str, logging output format
        :param date_format: str, date output format
        """

        logger = logging.getLogger(__name__)
        logger.setLevel(_LEVEL[level])
        logging.basicConfig(format=log_format, stream=sys.stdout, level=_LEVEL[level], datefmt=date_format)
        return logger

    @staticmethod
    def set_file_logger(log_file: str,
                         level: str = 'info',
                         log_format: str = _LOG_FORMAT,
                         date_format: str = _DATE_FORMAT):

        """
        Output logging info into a local file

        :param log_file: str, path of a .txt logging file
        :param level: str, logging output level {'debug', 'info', 'warning', 'error'}
        :param log_format: str, logging output format
        :param date_format: str, date output format
        """

        file_handler = logging.FileHandler(filename=log_file)
        formatter = logging.Formatter(fmt=log_format, datefmt=date_format)
        file_handler.setFormatter(formatter)
        logging.getLogger("").addHandler(file_handler)
