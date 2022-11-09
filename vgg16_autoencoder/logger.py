import logging
import os
import datetime


LOGGER = logging.Logger("style-transfer")

log_fmt = logging.Formatter(
    fmt="[%(asctime)s] %(levelname)s :: %(message)s",
    datefmt="%Y-%m-%d|%H:%M:%S",
)

def setup_logger(quiet, debug, verbose, save_log):
    global LOGGER

    if not quiet:
        # Stream logger
        str_hdl = logging.StreamHandler()
        str_hdl.setFormatter(log_fmt)
        if debug:
            str_hdl.setLevel(logging.DEBUG)
        elif verbose:
            str_hdl.setLevel(logging.INFO)
        else:
            str_hdl.setLevel(logging.WARNING)
        LOGGER.addHandler(str_hdl)

    if save_log:
        # File logger
        now = datetime.now()
        file_name = f"{now.year}{now.month:02}{now.day:02}_{now.hour:02}{now.minute:02}{now.second:02}"
        file_path = os.path.join("logs", f"{file_name}.txt")
        file_hdl = logging.FileHandler(file_path, encoding="utf-8")
        file_hdl.setFormatter(log_fmt)
        file_hdl.setLevel(logging.DEBUG)
        LOGGER.addHandler(file_hdl)
        LOGGER.debug(f"Log file: {os.path.abspath(file_path)}")