"""
Input prompt scanner — detects adversarial injection patterns before they
reach the target model.

Multi-layer detection:
  1. Role-override injection (DAN, jailbreak persona patterns)
  2. Instruction override markers ("Ignore previous instructions")
  3. Delimiter abuse (synthetic system prompts via markdown / XML)
  4. Encoding obfuscation (base64, ROT13, zero-width chars, homoglyphs)
  5. Crescendo escalation markers (topic drift phrasing)
  6. Known universal adversarial suffixes (GCG-style bracket/token spam)

Each fired rule contributes to a 0–1 risk_score. The scanner is intentionally
conservative — false positives are surfaced as low/medium severity so the caller
can decide whether to block or just log.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field


@dataclass
class ScanFlag:
    rule: str           # rule name
    severity: str       # low | medium | high
    matched: str        # the matched text (truncated)
    description: str    # human explanation


@dataclass
class ScanResult:
    is_suspicious: bool
    risk_score: float           # 0.0 – 1.0
    flags: list[ScanFlag] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.is_suspicious


# ── Rule weights (contribution to risk_score, capped at 1.0) ─────────────────

_RULES: list[tuple[str, float, str, re.Pattern | None, str]] = [
    # (name, weight, severity, pattern, description)
    (
        "role_override_dan",
        0.35,
        "high",
        re.compile(
            r"\b(DAN|do anything now|jailbreak|no restrictions|developer mode"
            r"|god mode|unrestricted|STAN|DUDE|AIM|KEVIN)\b",
            re.IGNORECASE,
        ),
        "Known jailbreak persona / role-override keyword",
    ),
    (
        "instruction_override",
        0.40,
        "high",
        re.compile(
            r"(ignore (all |your |previous |prior |above |these )*(instructions?|rules?|guidelines?|constraints?|policies?|restrictions?)"
            r"|disregard (all |your |previous |prior |above )*(instructions?|rules?|restrictions?)"
            r"|forget (all |your |previous )*(instructions?|rules?|guidelines?)"
            r"|override (your |the |all )*(instructions?|rules?|safety|alignment)"
            r"|you (are|must|will|shall) (now |no longer )?(follow|obey|act|pretend|roleplay|ignore))",
            re.IGNORECASE,
        ),
        "Instruction override / alignment bypass attempt",
    ),
    (
        "synthetic_system_prompt",
        0.35,
        "high",
        re.compile(
            r"(```\s*system|<\s*system\s*>|\[SYSTEM\]|#{2,}\s*SYSTEM|"
            r"---\s*SYSTEM\s*---|\[INST\].*<<SYS>>|\[\[system\]\])",
            re.IGNORECASE,
        ),
        "Synthetic system prompt injection via delimiter",
    ),
    (
        "base64_payload",
        0.25,
        "medium",
        re.compile(
            r"(?:base64|b64decode|atob)\s*[\(\[]?[A-Za-z0-9+/]{20,}={0,2}",
            re.IGNORECASE,
        ),
        "Base64-encoded payload — possible obfuscation",
    ),
    (
        "rot13_obfuscation",
        0.15,
        "low",
        re.compile(r"\brot13\b|\bcodecs\.decode\b", re.IGNORECASE),
        "ROT-13 obfuscation reference",
    ),
    (
        "hypothetical_framing",
        0.15,
        "low",
        re.compile(
            r"\b(hypothetically|in a fictional world|as a thought experiment"
            r"|imagine (you are|you were|if you had no)|for educational purposes only"
            r"|in a story where|in a film where)\b",
            re.IGNORECASE,
        ),
        "Hypothetical / fictional framing used to lower safety guardrails",
    ),
    (
        "crescendo_escalation",
        0.20,
        "medium",
        re.compile(
            r"(tell me more about|let's go deeper|expand on that|now explain"
            r"|and how would (someone|one|you) (actually|specifically|exactly)|"
            r"building on (that|what you said)|taking (this|it) (further|a step further))",
            re.IGNORECASE,
        ),
        "Crescendo-style gradual escalation phrasing",
    ),
    (
        "token_bracket_spam",
        0.20,
        "medium",
        re.compile(
            r"(\!{5,}|\[{3,}|\]{3,}|\!{3,}[A-Z]{2,}|"
            r"(\w+\s*){0,3}(\!{3,}|\*{5,}|={5,}|\+{5,})){1,}",
        ),
        "Token-level bracket / symbol spam (GCG-style universal suffix pattern)",
    ),
    (
        "zero_width_chars",
        0.30,
        "high",
        None,  # checked procedurally
        "Zero-width / invisible Unicode characters (homoglyph obfuscation)",
    ),
    (
        "excessive_unicode_escapes",
        0.20,
        "medium",
        re.compile(r"(\\u[0-9a-fA-F]{4}){5,}"),
        "Dense unicode escape sequences — possible homoglyph attack",
    ),
]

_ZERO_WIDTH_CATS = {"Cf", "Cs"}  # Unicode format / surrogate categories


def _has_zero_width(text: str) -> tuple[bool, str]:
    found = [ch for ch in text if unicodedata.category(ch) in _ZERO_WIDTH_CATS]
    return bool(found), "".join(f"\\u{ord(c):04x}" for c in found[:5])


def scan_input(prompt: str, block_threshold: float = 0.5) -> ScanResult:
    """
    Scan a prompt string for adversarial injection patterns.

    Parameters
    ----------
    prompt           : The raw user / attacker prompt.
    block_threshold  : risk_score >= this → is_suspicious = True.

    Returns
    -------
    ScanResult with risk_score, flags, and is_suspicious.
    """
    flags: list[ScanFlag] = []
    raw_score = 0.0

    for name, weight, severity, pattern, description in _RULES:
        if name == "zero_width_chars":
            found, sample = _has_zero_width(prompt)
            if found:
                flags.append(ScanFlag(
                    rule=name, severity=severity,
                    matched=sample or "<invisible>", description=description,
                ))
                raw_score += weight
        elif pattern is not None:
            m = pattern.search(prompt)
            if m:
                flags.append(ScanFlag(
                    rule=name, severity=severity,
                    matched=m.group(0)[:80], description=description,
                ))
                raw_score += weight

    risk_score = min(1.0, raw_score)
    return ScanResult(
        is_suspicious=risk_score >= block_threshold,
        risk_score=round(risk_score, 3),
        flags=flags,
    )
