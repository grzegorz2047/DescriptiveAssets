"""Result types shared by asset checks and level checks."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CheckResult:
    kind: str  # check name as written in YAML, e.g. "visible"
    passed: bool
    measured: str  # human-readable measured value, e.g. "door 0.42 in front"
    expected: str  # human-readable expectation, e.g. ">= 0.80"
    message: str = ""  # extra detail on failure
    why: str | None = None  # designer intent from the `why:` key
    data: dict = field(default_factory=dict)  # machine-readable numbers for --json

    def line(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        text = f"{status}  {self.kind:<18} measured {self.measured}  expected {self.expected}"
        if not self.passed and self.message:
            text += f"  — {self.message}"
        if not self.passed and self.why:
            text += f"  (why: {self.why})"
        return text
