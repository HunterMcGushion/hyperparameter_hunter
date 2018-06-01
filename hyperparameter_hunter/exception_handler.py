##################################################
# Import Miscellaneous Assets
##################################################
import logging
import sys

logger = logging.getLogger(__name__)

# noinspection PyProtectedMember
stream_handler = logging._StderrHandler()
# stream_handler = logging.StreamHandler(stream=sys.stdout)
logger.addHandler(stream_handler)


def handle_exception(exception_type, exception_value, exception_traceback):
    if issubclass(exception_type, KeyboardInterrupt):
        logging.error('KEYBOARD INTERRUPT!')
        sys.__excepthook__(exception_type, exception_value, exception_traceback)
        raise SystemExit

    logging.critical(
        'Uncaught exception!   {}: {}'.format(exception_type.__name__, exception_value),
        exc_info=(exception_type, exception_value, exception_traceback)
    )


def hook_exception_handler():
    sys.excepthook = handle_exception


# sys.excepthook = handle_exception


class EnvironmentInactiveError(Exception):
    def __init__(self, message=None, extra=''):
        if not message:
            message = 'You must activate a valid instance of :class:`environment.Environment`'
        super(EnvironmentInactiveError, self).__init__(message + extra)


class EnvironmentInvalidError(Exception):
    def __init__(self, message=None, extra=''):
        if not message:
            message = 'The currently active Environment is invalid. Please review proper Environment instantiation'
        super(EnvironmentInvalidError, self).__init__(message + extra)


class RepeatedExperimentError(Exception):
    def __init__(self, message=None, extra=''):
        if not message:
            message = 'An Experiment with identical hyperparameters has already been conducted and has saved results'
        super(RepeatedExperimentError, self).__init__(message + extra)


def execute():
    pass


if __name__ == '__main__':
    execute()
