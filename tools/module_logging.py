import logging


def addLoggingLevel(levelName: str, levelNum: int, methodName: str | None = None):
    """
    Comprehensively adds a new logging level to the `logging` module and the
    currently configured logging class.

    Args:
        levelName (str): The name of the new level attribute of the `logging` module
        levelNum (int): Determines the logging level of the attribute
        methodName (str | None, optional): The name of the convenience method for logging on the new level.
                        Defaults to the `levelName` in lower case.

    Raises:
        AttributeError: Raised if `levelName` is already an attribute of the `logging` module. or if the method name is already present.

    Examples:
    >>> addLoggingLevel('TRACE', logging.DEBUG - 5)
    >>> logging.getLogger(__name__).setLevel("TRACE")
    >>> logging.getLogger(__name__).trace('that worked')
    >>> logging.trace('so did this')
    >>> logging.TRACE
    5
    """
    if not methodName:
        methodName = levelName.lower()

    if hasattr(logging, levelName):
        raise AttributeError("{} already defined in logging module".format(levelName))
    if hasattr(logging, methodName):
        raise AttributeError("{} already defined in logging module".format(methodName))
    if hasattr(logging.getLoggerClass(), methodName):
        raise AttributeError("{} already defined in logger class".format(methodName))

    # This method was inspired by the answers to Stack Overflow post
    # http://stackoverflow.com/q/2183233/2988730, especially
    # http://stackoverflow.com/a/13638084/2988730
    def logForLevel(self, message, *args, **kwargs):
        if self.isEnabledFor(levelNum):
            self._log(levelNum, message, args, **kwargs)

    def logToRoot(message, *args, **kwargs):
        logging.log(levelNum, message, *args, **kwargs)

    logging.addLevelName(levelNum, levelName)
    setattr(logging, levelName, levelNum)
    setattr(logging.getLoggerClass(), methodName, logForLevel)
    setattr(logging, methodName, logToRoot)


def get_logger(level=logging.DEBUG) -> logging.Logger:
    """get a logger for the module with the given level

    Args:
        level (optional): Set the logging level of the logger. Defaults to logging.DEBUG.

    Returns:
        logging.Logger: The logger for the module.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(level)

    channel = logging.StreamHandler()
    channel.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s: \t%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    channel.setFormatter(formatter)
    logger.addHandler(channel)
    return logger
