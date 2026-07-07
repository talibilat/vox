"""The `earshot` command: daemon lifecycle and (later) everything else."""

from __future__ import annotations

import argparse
import sys

from earshot import __version__, daemon
from earshot import config as config_module
from earshot.config import ConfigError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="earshot",
        description="Voice-to-voice control for terminal coding agents.",
    )
    parser.add_argument("--version", action="version", version=f"earshot {__version__}")
    parser.add_argument(
        "--config",
        metavar="PATH",
        help=f"config file (default: {config_module.DEFAULT_CONFIG_PATH})",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    start = sub.add_parser("start", help="start the daemon")
    start.add_argument(
        "--foreground",
        action="store_true",
        help="run in this terminal instead of detaching (for development)",
    )
    sub.add_parser("stop", help="stop the daemon")
    sub.add_parser("status", help="show whether the daemon is running")
    sub.add_parser(
        "interrupt",
        help="stop the agent's speech now (push-to-interrupt escape hatch); "
        "bind this command to a system hotkey for one-keystroke barge-in",
    )
    sub.add_parser("run", help=argparse.SUPPRESS)  # internal: detached child entrypoint
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = config_module.load(args.config)
    except ConfigError as e:
        print(f"earshot: config error: {e}", file=sys.stderr)
        return 2

    try:
        if args.command == "start":
            if args.foreground:
                daemon.run(config)
                return 0
            pid = daemon.start(config, args.config)
            print(f"earshot daemon started (pid {pid})")
        elif args.command == "stop":
            pid = daemon.stop(config)
            print(f"earshot daemon stopped (pid {pid})")
        elif args.command == "status":
            pid = daemon.read_pid(config)
            if pid is None:
                print("earshot daemon is not running")
                return 1
            print(f"earshot daemon is running (pid {pid})")
        elif args.command == "interrupt":
            pid = daemon.interrupt(config)
            print(f"interrupt sent to earshot daemon (pid {pid})")
        elif args.command == "run":
            daemon.run(config)
    except RuntimeError as e:
        print(f"earshot: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
