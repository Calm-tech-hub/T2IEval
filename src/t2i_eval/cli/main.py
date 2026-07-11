"""Custom CLI parser for the unified `t2i-eval` command."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# Ensure model/evaluator registries are populated.
import t2i_eval.eval  # noqa: F401  # pylint: disable=unused-import
import t2i_eval.model  # noqa: F401  # pylint: disable=unused-import

from .config_loader import load_run_config
from .logger import CliLogger
from .runner import Runner
from .utils import parse_kwargs

USAGE = """\
Usage: t2i-eval [run] [OPTIONS]

Options:
  -m, --model TEXT            Model registry key (e.g., diffusers)
  -a, --model-args TEXT       Model init args as key=value[,key=value]
  -g, --gen TEXT              Global generation args as key=value[,key=value]
  -e, --eval TEXT             Evaluation name (repeatable, defines order)
  -E, --eval-args TEXT        Eval args; name:key=value or key=value (uses last -e)
  -G, --eval-gen TEXT         Eval generation overrides; same syntax as -E
  -f, --file PATH             YAML/JSON run configuration
  -o, --output-dir PATH       Directory to place results
      --fail-fast             Abort on first evaluator failure
      --quiet                 Silence CLI log output
      --verbose               Emit verbose logs
      --no-summary            Skip per-evaluator stdout summaries
      --save-intermediate     (Reserved) dump intermediate artifacts
  -h, --help                  Show this message and exit
"""


class CLIError(Exception):
    """Raised when user input is invalid."""


class CLIHelp(Exception):
    """Raised to signal that help text should be printed."""


@dataclass
class ParsedArgs:
    model: Optional[str] = None
    model_args: Dict[str, str] = field(default_factory=dict)
    generation_args: Dict[str, str] = field(default_factory=dict)
    eval_order: List[str] = field(default_factory=list)
    eval_args: Dict[str, Dict[str, str]] = field(default_factory=dict)
    eval_gen: Dict[str, Dict[str, str]] = field(default_factory=dict)
    file_path: Optional[str] = None
    output_dir: Optional[str] = None
    fail_fast: bool = False
    quiet: bool = False
    verbose: bool = False
    save_intermediate: bool = False
    show_summary: bool = True

    def to_loader_kwargs(self) -> Dict:
        return {
            "model": self.model,
            "model_args": self.model_args,
            "gen": self.generation_args,
            "eval": self.eval_order,
            "eval_args": self.eval_args,
            "eval_gen": self.eval_gen,
            "file": self.file_path,
            "output_dir": self.output_dir,
            "fail_fast": self.fail_fast,
            "save_intermediate": self.save_intermediate,
        }


def _require_value(argv: List[str], index: int, option: str) -> str:
    if index >= len(argv):
        raise CLIError(f"Option {option} requires a value.")
    return argv[index]


def _apply_scoped(target: Dict[str, Dict[str, str]], additions: Dict[str, Dict[str, str]]) -> None:
    for scope, kv in additions.items():
        target.setdefault(scope, {}).update(kv or {})


def _parse_scoped(value: str, last_eval: Optional[str], option: str) -> Dict[str, Dict[str, str]]:
    scoped: Dict[str, Dict[str, str]] = {}
    parts = value.split(",") if value else []
    for part in parts:
        chunk = part.strip()
        if not chunk:
            continue
        if ":" in chunk:
            scope, rest = chunk.split(":", 1)
            scope = scope.strip()
        else:
            if not last_eval:
                raise CLIError(
                    f"{option} specified without a preceding -e/--eval to attach it to."
                )
            scope = last_eval
            rest = chunk
        if "=" not in rest:
            raise CLIError(f"{option} expects key=value entries (got '{chunk}').")
        key, val = rest.split("=", 1)
        key = key.strip()
        val = val.strip()
        if not scope or not key:
            raise CLIError(f"{option} entries must have non-empty scope and key.")
        scoped.setdefault(scope, {})[key] = val
    return scoped


def parse_cli_args(argv: List[str]) -> ParsedArgs:
    if not argv:
        raise CLIHelp()

    parsed = ParsedArgs()
    i = 0
    last_eval: Optional[str] = None

    while i < len(argv):
        token = argv[i]

        if token in {"-h", "--help"}:
            raise CLIHelp()
        if token.startswith("--") and token == "--":
            i += 1
            break
        elif token in {"-m", "--model"}:
            value = _require_value(argv, i + 1, token)
            parsed.model = value
            i += 2
        elif token in {"-a", "--model-args"}:
            raw = _require_value(argv, i + 1, token)
            parsed.model_args.update(parse_kwargs(raw))
            i += 2
        elif token in {"-g", "--gen"}:
            raw = _require_value(argv, i + 1, token)
            parsed.generation_args.update(parse_kwargs(raw))
            i += 2
        elif token in {"-e", "--eval"}:
            value = _require_value(argv, i + 1, token)
            parsed.eval_order.append(value)
            last_eval = value
            i += 2
        elif token in {"-E", "--eval-args"}:
            raw = _require_value(argv, i + 1, token)
            _apply_scoped(parsed.eval_args, _parse_scoped(raw, last_eval, token))
            i += 2
        elif token in {"-G", "--eval-gen"}:
            raw = _require_value(argv, i + 1, token)
            _apply_scoped(parsed.eval_gen, _parse_scoped(raw, last_eval, token))
            i += 2
        elif token in {"-f", "--file"}:
            parsed.file_path = _require_value(argv, i + 1, token)
            i += 2
        elif token in {"-o", "--output-dir"}:
            parsed.output_dir = _require_value(argv, i + 1, token)
            i += 2
        elif token == "--fail-fast":
            parsed.fail_fast = True
            i += 1
        elif token == "--quiet":
            parsed.quiet = True
            i += 1
        elif token == "--verbose":
            parsed.verbose = True
            i += 1
        elif token == "--save-intermediate":
            parsed.save_intermediate = True
            i += 1
        elif token == "--no-summary":
            parsed.show_summary = False
            i += 1
        else:
            raise CLIError(f"Unknown option '{token}'. Use --help for usage.")

    return parsed


def run(argv: Optional[List[str]] = None) -> None:
    args = sys.argv[1:] if argv is None else argv
    if args and args[0] == "run":
        args = args[1:]

    try:
        parsed = parse_cli_args(args)
    except CLIHelp:
        print(USAGE)
        return
    except CLIError as exc:
        print(f"Error: {exc}\n\n{USAGE}")
        raise SystemExit(1)

    cli_kwargs = parsed.to_loader_kwargs()
    logger = CliLogger(quiet=parsed.quiet, verbose=parsed.verbose)

    try:
        run_cfg = load_run_config(cli_kwargs)
        runner = Runner(run_cfg, logger=logger, show_summary=parsed.show_summary)
        runner.run()
    except Exception as exc:  # noqa: BLE001
        logger.error(str(exc))
        raise SystemExit(1)


def main() -> None:
    run()


if __name__ == "__main__":
    main()
