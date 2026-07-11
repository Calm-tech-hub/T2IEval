import os
from typing import Optional

import click


def _is_main_process() -> bool:
    """Return True when running on the main process.

    Accelerate sets LOCAL_RANK for distributed launches. Treat missing or "0"
    as main; anything else as non-main.
    """

    rank = os.environ.get("LOCAL_RANK")
    return rank is None or str(rank) == "0"


class CliLogger:
    def __init__(self, quiet: bool = False, verbose: bool = False, stream: Optional[object] = None):
        self.quiet = quiet
        self.verbose = verbose
        self.stream = stream

    def _should_emit(self) -> bool:
        if self.quiet:
            return False
        if not _is_main_process():
            return False
        return True

    def info(self, msg: str):
        if self._should_emit():
            click.secho(msg, file=self.stream)

    def warn(self, msg: str):
        if self._should_emit():
            click.secho(msg, fg="yellow", file=self.stream)

    def error(self, msg: str):
        if self._should_emit():
            click.secho(msg, fg="red", file=self.stream)

    def section(self, msg: str):
        if self._should_emit():
            click.secho(msg, bold=True, file=self.stream)
