# Security Policy

## What this tool does

PromptStrike is an **adversarial red teaming framework** for large language models. It automates the PAIR and TAP jailbreak algorithms from academic literature to measure how well an LLM resists manipulation.

**Authorized use only.** Only test LLMs you own or have explicit written permission to test. Unauthorized testing may violate the provider's terms of service and applicable law.

---

## Scope

PromptStrike is intended for:

- **Security research** — measuring LLM robustness against known attack algorithms
- **Red teaming engagements** — with explicit authorization from the model operator
- **AI safety evaluation** — internal testing of your own deployed models
- **CI/CD safety gates** — automated pre-deployment regression checks

It is **not** intended for:

- Attacking production systems without authorization
- Bypassing safety measures on public LLM APIs in violation of their ToS
- Generating harmful content for any purpose other than measuring model behavior

---

## Reporting vulnerabilities in PromptStrike itself

If you discover a security vulnerability in the PromptStrike codebase (e.g. a dependency with a known CVE, an injection flaw in the dashboard, unintended credential exposure), please report it responsibly:

1. **Do not open a public issue.** GitHub issues are indexed and may expose the vulnerability before a fix is available.
2. **Open a [GitHub Security Advisory](https://github.com/Pyhroff/promptstrike/security/advisories/new)** — this is private between you and the maintainer.
3. Include: a description of the issue, reproduction steps, and your assessment of impact.

I aim to acknowledge reports within **48 hours** and publish a fix within **7 days** for critical issues.

---

## Dependency security

Dependencies are pinned with minimum versions in `pyproject.toml`. Run `pip audit` or `safety check` periodically to check for CVEs in the dependency tree.

```bash
pip install pip-audit
pip-audit
```

---

## Responsible disclosure timeline

| Day | Action |
|-----|--------|
| 0 | Report received |
| 1–2 | Acknowledgement sent |
| 3–7 | Patch developed and tested |
| 7–14 | Fix released, CVE filed if applicable |
| 14+ | Public disclosure (coordinated with reporter) |
