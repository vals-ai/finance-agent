import logging
import os
from datetime import datetime

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

# Maximum length for log messages
MAX_MESSAGE_LENGTH = 1000

LOGS_DIR = "logs"
os.makedirs(LOGS_DIR, exist_ok=True)


def color(color):
    colored_str = "".join((color, LEVEL, RESET))
    bold_str = "".join((BOLD, NAME, RESET))
    return " ".join((colored_str, bold_str, MSG))


class ColorFormatter(logging.Formatter):
    FORMATS = {
        logging.DEBUG: color(GREY),
        logging.INFO: color(GREEN),
        logging.WARNING: color(YELLOW),
        logging.ERROR: color(RED),
        logging.CRITICAL: color(BOLD_RED),
    }

    def format(self, record):
        # Truncate message if it exceeds MAX_MESSAGE_LENGTH
        if len(record.msg) > MAX_MESSAGE_LENGTH:
            record.msg = record.msg[:MAX_MESSAGE_LENGTH] + "... [truncated]"

        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


class TruncatingFormatter(logging.Formatter):
    """Formatter that truncates long messages."""

    def format(self, record):
        # Truncate message if it exceeds MAX_MESSAGE_LENGTH
        if len(record.msg) > MAX_MESSAGE_LENGTH:
            record.msg = record.msg[:MAX_MESSAGE_LENGTH] + "... [truncated]"

        return super().format(record)


def get_logger(name: str) -> logging.Logger:
    """Get a configured logger instance."""
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

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(LOGS_DIR, f"{name}_{timestamp}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = TruncatingFormatter(
            "%(asctime)s %(levelname)s -- %(name)s: %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger
