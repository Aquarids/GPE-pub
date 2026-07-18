import logging
import traceback
from pathlib import Path

ENV_LOCAL = "local"
ENV_DEV = "dev"
ENV_PROD = "prod"


class Logger:
    def __init__(self, name, env, enabled=True, level=None, log_file=None):
        self.env = env
        self.enabled = enabled
        self.log_file = log_file

        if not enabled:
            self.logger = logging.getLogger(name)
            self.logger.addHandler(logging.NullHandler())
            self.logger.setLevel(logging.CRITICAL + 1)
        else:
            self.logger = self._init_logging(env, name, log_file)
            if level:
                self.set_level(getattr(logging, level.upper()))

    def _init_logging(self, env, name, log_file):
        if env in [ENV_LOCAL, ENV_DEV]:
            level = logging.DEBUG
            handlers = [logging.StreamHandler()]
        elif env == ENV_PROD:
            level = logging.INFO
            handlers = [
                logging.FileHandler("lpivot.log", encoding="utf-8"),
            ]

        formatter = logging.Formatter(
            fmt="%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        if log_file:
            handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

        root_logger = logging.getLogger()
        root_logger.setLevel(level)
        if not root_logger.handlers:
            for handler in handlers:
                handler.setFormatter(formatter)
                root_logger.addHandler(handler)
        elif log_file and not _has_file_handler(root_logger, log_file):
            handler = logging.FileHandler(log_file, encoding="utf-8")
            handler.setFormatter(formatter)
            root_logger.addHandler(handler)

        if env in [ENV_LOCAL, ENV_DEV]:
            logging.getLogger("httpx").setLevel(logging.WARNING)
            logging.getLogger("openai").setLevel(logging.WARNING)
            logging.getLogger("httpcore").setLevel(logging.WARNING)
            logging.getLogger('reqwest').setLevel(logging.WARNING)
            logging.getLogger('hyper').setLevel(logging.WARNING)
            logging.getLogger('h2').setLevel(logging.WARNING)
            logging.getLogger('rustls').setLevel(logging.WARNING)
            logging.getLogger('tokio').setLevel(logging.WARNING)
        elif env == ENV_PROD:
            logging.getLogger("httpx").setLevel(logging.INFO)
            logging.getLogger("openai").setLevel(logging.INFO)
            logging.getLogger("httpcore").setLevel(logging.INFO)
            logging.getLogger('hyper').setLevel(logging.INFO)
            logging.getLogger('h2').setLevel(logging.INFO)
            logging.getLogger('rustls').setLevel(logging.INFO)
            logging.getLogger('tokio').setLevel(logging.INFO)

        return logging.getLogger(name)

    def set_level(self, level):
        if self.enabled:
            self.logger.setLevel(level)

    def log_exception(self, e: Exception):
        if not self.enabled:
            return
        error_trace = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        self.logger.error(error_trace)

    def error(self, msg, *args, **kwargs):
        if not self.enabled:
            return
        self.logger.error(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        if not self.enabled:
            return
        self.logger.info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        if not self.enabled or not self.is_debug():
            return
        self.logger.warning(msg, *args, **kwargs)

    def debug(self, msg, *args, **kwargs):
        if not self.enabled or not self.is_debug():
            return
        self.logger.debug(msg, *args, **kwargs)

    def is_debug(self):
        return self.env in [ENV_LOCAL, ENV_DEV]


def _has_file_handler(logger, log_file):
    target = str(Path(log_file).resolve())
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler) and handler.baseFilename == target:
            return True
    return False
