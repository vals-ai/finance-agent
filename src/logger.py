import logging
import os

GREEN = "\x1b[32;20m"
GREY = "\x1b[38;20m"
YELLOW = "\x1b[33;20m"
RED = "\x1b[31;20m"
BOLD_RED = "\x1b[31;1m"
BOLD = "\x1b[1m"
RESET = "\x1b[0m"

BASE = "%(asctime)s"
LEVEL = "%(levelname)s"
NAME = "-- %(name)s:"
MSG = "%(message)s"

# Format for console output (without date)
CONSOLE_FORMAT = " ".join((LEVEL, NAME, MSG))

# Format for file output (with date)
FILE_FORMAT = " ".join((BASE, LEVEL, NAME, MSG))

VERBOSE = os.environ.get("EDGAR_AGENT_VERBOSE", "0") == "1"
MAX_MESSAGE_LENGTH = 20000

# Global run context for organized logging
_current_run_dir: str | None = None
_question_file_handlers: dict[str, list[logging.FileHandler]] = {}


def color(color):
    colored_str = "".join((color, LEVEL, RESET))
    bold_str = "".join((BOLD, NAME, RESET))
    return " ".join((colored_str, bold_str, MSG))


class TruncatingFormatter(logging.Formatter):
    def format(self, record):
        # truncate message
        if len(record.msg) > MAX_MESSAGE_LENGTH and not VERBOSE:
            record.msg = record.msg[:MAX_MESSAGE_LENGTH] + "... [truncated]"
            half = MAX_MESSAGE_LENGTH // 2
            record.msg = f"{record.msg[:half]}...[truncated]...{record.msg[-half:]}"

        return super().format(record)


class ColorFormatter(TruncatingFormatter):
    FORMATS = {
        logging.DEBUG: color(GREY),
        logging.INFO: color(GREEN),
        logging.WARNING: color(YELLOW),
        logging.ERROR: color(RED),
        logging.CRITICAL: color(BOLD_RED),
    }

    def format(self, record):
        super().format(record)

        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def get_logger(name: str) -> logging.Logger:
    """Get a configured logger instance"""

    logger = logging.getLogger(name)
    logger.propagate = False

    if not logger.hasHandlers():
        logger.setLevel(logging.DEBUG)

        # Console Handler (with colors, no date)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_formatter = ColorFormatter()
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        # File handlers are added per-question via setup_question_logging()

    return logger


def setup_question_logging(question_dir: str, loggers: list[str]) -> None:
    """Adds a file handler for the question"""

    global _question_file_handlers
    os.makedirs(question_dir, exist_ok=True)

    handlers = []
    for logger_name in loggers:
        logger = logging.getLogger(logger_name)

        log_file = os.path.join(question_dir, f"{logger_name}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = TruncatingFormatter(
            "%(asctime)s %(levelname)s -- %(name)s: %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        handlers.append(file_handler)

    _question_file_handlers[question_dir] = handlers


def teardown_question_logging(question_dir: str, loggers: list[str] | None = None) -> None:
    """Removes the file handler for the question"""

    global _question_file_handlers
    if question_dir not in _question_file_handlers:
        return

    handlers = _question_file_handlers[question_dir]

    # If logger names provided, use those; otherwise check common names
    logger_names = loggers if loggers else ["agent", "tools", "__main__"]

    for handler in handlers:
        handler.close()
        # find and remove question logger
        for logger_name in logger_names:
            logger = logging.getLogger(logger_name)
            if handler in logger.handlers:
                logger.removeHandler(handler)

    del _question_file_handlers[question_dir]
