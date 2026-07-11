from __future__ import annotations

import os
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path

DEFAULT_DIST_ENV_REMOVE_KEYS = (
    "RANK",
    "WORLD_SIZE",
    "LOCAL_RANK",
    "LOCAL_WORLD_SIZE",
    "GROUP_RANK",
    "ROLE_RANK",
    "ROLE_WORLD_SIZE",
    "MASTER_ADDR",
    "MASTER_PORT",
    "NODE_RANK",
    "PMI_RANK",
    "PMI_SIZE",
    "PMIX_RANK",
    "OMPI_COMM_WORLD_RANK",
    "OMPI_COMM_WORLD_SIZE",
    "OMPI_COMM_WORLD_LOCAL_RANK",
    "SLURM_PROCID",
    "SLURM_LOCALID",
    "SLURM_NPROCS",
    "SLURM_NTASKS",
)

DEFAULT_DIST_ENV_REMOVE_PREFIXES = (
    "ACCELERATE_",
    "TORCHELASTIC_",
)


class IsolatedRunner:
    """
    Reusable isolated command executor backed by a uv-managed project environment.

    The runner separates three concerns:
    1) project environment preparation
    2) command execution directory
    3) process environment shaping

    This makes it suitable for evaluation vendors that share one dependency set
    but need to run multiple scripts from different subdirectories.
    """

    _prepared_envs: set[tuple[str, str, str | None]] = set()

    def __init__(
        self,
        workdir: str | Path,
        script: str | Path | None = None,
        command: Sequence[str] | None = None,
        args: Sequence[str] | None = None,
        version: str | None = "3.13",
        run_cwd: str | Path | None = None,
        venv_path: str | Path | None = None,
        env: Mapping[str, str] | None = None,
        env_remove_keys: Sequence[str] | None = None,
        env_remove_prefixes: Sequence[str] | None = None,
        sync_once: bool = True,
    ) -> None:
        if script is None and command is None:
            raise ValueError("IsolatedRunner requires either `script` or `command`.")
        if script is not None and command is not None:
            raise ValueError(
                "IsolatedRunner accepts only one of `script` or `command`."
            )

        self.workdir = Path(workdir).resolve()
        self.run_cwd = self._resolve_path(
            run_cwd, base=self.workdir, default=self.workdir
        )
        self.venv_path = self._resolve_path(
            venv_path,
            base=self.workdir,
            default=self.workdir / ".venv",
        )
        self.script = (
            None if script is None else self._resolve_path(script, base=self.workdir)
        )
        self.command = None if command is None else [str(arg) for arg in command]
        self.args = [str(arg) for arg in (args or [])]
        self.version = version
        self.env = {key: str(value) for key, value in (env or {}).items()}
        self.env_remove_keys = {
            str(key)
            for key in (
                DEFAULT_DIST_ENV_REMOVE_KEYS
                if env_remove_keys is None
                else env_remove_keys
            )
        }
        self.env_remove_prefixes = tuple(
            str(prefix)
            for prefix in (
                DEFAULT_DIST_ENV_REMOVE_PREFIXES
                if env_remove_prefixes is None
                else env_remove_prefixes
            )
        )
        self.sync_once = sync_once
        self._venv_lock_fd: int | None = None

    @staticmethod
    def _resolve_path(
        path: str | Path | None,
        *,
        base: Path,
        default: Path | None = None,
    ) -> Path:
        if path is None:
            if default is None:
                raise ValueError("A default path is required when `path` is None.")
            return default.resolve()

        raw_path = Path(path)
        return (
            raw_path.resolve()
            if raw_path.is_absolute()
            else (base / raw_path).resolve()
        )

    @staticmethod
    def _run_subprocess(cmd: list[str], cwd: Path, env: dict[str, str]) -> None:
        process = subprocess.run(
            cmd,
            cwd=str(cwd),
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        if process.returncode != 0:
            raise RuntimeError(
                f"Command failed: {' '.join(cmd)}\n"
                f"cwd: {cwd}\n"
                f"stdout:\n{process.stdout}\n"
                f"stderr:\n{process.stderr}"
            )

    def _lock_venv(self) -> None:
        if self._venv_lock_fd is not None:
            return
        lock_file = self.workdir / ".uv_venv.lock"
        lock_file.parent.mkdir(parents=True, exist_ok=True)
        lock_fd = os.open(str(lock_file), os.O_CREAT | os.O_RDWR, 0o644)
        try:
            import fcntl

            fcntl.flock(lock_fd, fcntl.LOCK_EX)
        except ImportError:
            pass
        self._venv_lock_fd = lock_fd

    def _unlock_venv(self) -> None:
        if self._venv_lock_fd is None:
            return
        try:
            try:
                import fcntl

                fcntl.flock(self._venv_lock_fd, fcntl.LOCK_UN)
            except ImportError:
                pass
        finally:
            os.close(self._venv_lock_fd)
            self._venv_lock_fd = None

    def _create_venv(self) -> None:
        if self.venv_path.exists():
            return

        self.venv_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = ["uv", "venv", str(self.venv_path)]
        if self.version:
            cmd.extend(["--python", self.version])
        self._run_subprocess(
            cmd,
            cwd=self.workdir,
            env=self._build_subprocess_env(),
        )

    def _sync_env(self) -> None:
        self._run_subprocess(
            ["uv", "sync", "--project", str(self.workdir), "--no-dev"],
            cwd=self.workdir,
            env=self._build_subprocess_env(),
        )

    def _env_cache_key(self) -> tuple[str, str, str | None]:
        return (str(self.workdir), str(self.venv_path), self.version)

    def _build_subprocess_env(
        self,
        extra_env: Mapping[str, str] | None = None,
    ) -> dict[str, str]:
        env = os.environ.copy()
        env.update(self.env)
        if extra_env:
            env.update({key: str(value) for key, value in extra_env.items()})

        for key in self.env_remove_keys:
            env.pop(key, None)

        if self.env_remove_prefixes:
            for key in list(env.keys()):
                if any(key.startswith(prefix) for prefix in self.env_remove_prefixes):
                    env.pop(key, None)

        env["UV_PROJECT_ENVIRONMENT"] = str(self.venv_path)
        return env

    def _prepare_env(self) -> None:
        self._lock_venv()
        try:
            env_cache_key = self._env_cache_key()
            if (
                self.sync_once
                and env_cache_key in self._prepared_envs
                and self.venv_path.exists()
            ):
                return
            self._create_venv()
            self._sync_env()
            self._prepared_envs.add(env_cache_key)
        except Exception:
            self._unlock_venv()
            raise

    def _build_uv_run_command(
        self,
        args: Sequence[str] | None = None,
        command: Sequence[str] | None = None,
    ) -> list[str]:
        payload = [str(arg) for arg in (command or self.command or [])]
        if self.script is not None:
            payload = [
                "python",
                str(self.script),
                *self.args,
                *[str(arg) for arg in (args or [])],
            ]
        elif args:
            payload.extend(str(arg) for arg in args)

        return [
            "uv",
            "run",
            "--no-sync",
            "--project",
            str(self.workdir),
            *payload,
        ]

    def run(
        self,
        args: Sequence[str] | None = None,
        *,
        command: Sequence[str] | None = None,
        cwd: str | Path | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        self._prepare_env()
        self._unlock_venv()
        run_cwd = self._resolve_path(cwd, base=self.workdir, default=self.run_cwd)
        cmd = self._build_uv_run_command(args=args, command=command)
        self._run_subprocess(cmd, cwd=run_cwd, env=self._build_subprocess_env(env))

    def __call__(
        self,
        args: Sequence[str] | None = None,
        *,
        command: Sequence[str] | None = None,
        cwd: str | Path | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        self.run(args=args, command=command, cwd=cwd, env=env)
