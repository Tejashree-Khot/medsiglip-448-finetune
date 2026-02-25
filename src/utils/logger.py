"""Logging configuration for the project."""

import logging
import sys

LOG_FORMAT = "%(levelname)s:     %(message)s"
LOGGER = logging.getLogger("medsiglip")
LOGGER.setLevel(logging.INFO)


def configure_logging():
    """Configure logging with consistent format across modules."""
    logging.basicConfig(
        level=logging.INFO,
        format=LOG_FORMAT,
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,  # ensures consistent config across modules
    )
