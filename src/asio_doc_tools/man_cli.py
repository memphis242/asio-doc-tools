"""`asio-docs man ...` - placeholder until the man page subcommands land."""

import argparse


def register(commands: "argparse._SubParsersAction[argparse.ArgumentParser]") -> None:
    del commands
