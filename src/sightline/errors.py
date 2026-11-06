"""Errors meant to be read by an AI agent: what is wrong, where, and how to fix it."""

from __future__ import annotations

import difflib


class SightlineError(Exception):
    """A user-facing error. `where` points into the source file, `hint` says how to fix it."""

    def __init__(self, message: str, where: str | None = None, hint: str | None = None):
        self.message = message
        self.where = where
        self.hint = hint
        super().__init__(self.format())

    def format(self) -> str:
        text = f"{self.where}: {self.message}" if self.where else self.message
        if self.hint:
            text += f"\n  hint: {self.hint}"
        return text


def did_you_mean(value: str, options) -> str | None:
    """Return 'did you mean ...?' text for the closest option, or None."""
    matches = difflib.get_close_matches(str(value), [str(o) for o in options], n=1, cutoff=0.6)
    return f"did you mean '{matches[0]}'?" if matches else None
