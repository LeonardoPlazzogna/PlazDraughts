"""A `--help` that answers, for the tools driven by environment variables.

THE PROBLEM IT SOLVES. Several tools in this project take no command-line
arguments: they are configured through environment variables. Having nothing
that parses the command line, they ignored `--help` and started computing.
Asking the conductor for help started a hundred-cycle run.

It is the kind of defect never seen on a machine where every tool is already
known, and that hits anyone else at the first line -- right while they are
trying to understand what the program does. Found by trying `--help` on every
tool of the project, one by one.

NOT TO BE USED in tools that already have argparse: there it would intercept
--help and print this description INSTEAD of the list of options, which is
exactly what whoever types --help is looking for. It really happened, on
benchmark.py: the new options no longer appeared.

Usage, as the first line of main():

    from env_help import help_if_requested
    help_if_requested(__doc__, __file__)

The list of variables is not written by hand: it is read from the source
itself, so it cannot go stale. Documentation that updates itself is the only
kind that still tells the truth six months later.
"""
from __future__ import annotations

import ast
import os
import re
import sys

_NAME = re.compile(r"[A-Z_][A-Z0-9_]*")


def variables_read(path: str) -> list[tuple[str, str]]:
    """The environment variables the file reads, with their default value.

    The source is parsed, not matched with a regular expression. A default that
    is itself a call -- str(os.cpu_count() or 8) -- closes a parenthesis before
    its real end, and the regular expression that used to do this job cut those
    defaults off halfway: `--help` printed `str(os.cpu_count(`.
    """
    try:
        with open(path, encoding="utf-8") as f:
            tree = ast.parse(f.read())
    except (OSError, SyntaxError, ValueError):
        return []
    calls = sorted((node for node in ast.walk(tree)
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "get" and ast.unparse(node.func.value) == "os.environ"
                    and node.args and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str) and _NAME.fullmatch(node.args[0].value)),
                   key=lambda node: (node.lineno, node.col_offset))
    seen, out = set(), []
    for call in calls:
        name = call.args[0].value
        if name in seen:
            continue
        seen.add(name)
        if len(call.args) < 2:
            default = ""
        elif isinstance(call.args[1], ast.Constant):
            default = str(call.args[1].value)
        else:
            default = ast.unparse(call.args[1])
        out.append((name, default))
    return sorted(out)


def _shown_path(path: str) -> str:
    """The script path as typed from the project root (tools/x.py, not x.py)."""
    try:
        rel = os.path.relpath(os.path.abspath(path),
                              os.path.dirname(os.path.abspath(__file__)))
    except ValueError:                      # different drive on Windows
        return os.path.basename(path)
    return rel.replace(os.sep, "/")


def help_if_requested(docstring: str | None, path: str) -> None:
    """If -h or --help is on the command line, prints the help and exits.

    It does not use argparse on purpose: these tools take NO arguments, and
    pretending they do would be worse than saying so."""
    if not any(a in ("-h", "--help", "/?") for a in sys.argv[1:]):
        return

    name = _shown_path(path)
    print((docstring or "").strip() or f"{name}: no description.")

    variables = variables_read(path)
    if variables:
        print("\nConfigured through ENVIRONMENT VARIABLES (not arguments):\n")
        width = max(len(n) for n, _ in variables)
        for n, default in variables:
            print(f"  {n:<{width}}  {'= ' + default if default else '(no default)'}")
        first = variables[0][0]
        print("\nExample:")
        if os.name == "nt":
            print(f"  $env:{first}='...'; python {name}")
        else:
            print(f"  {first}=... python {name}")

    sys.exit(0)
