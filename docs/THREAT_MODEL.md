# PromptStrike Threat Model

## Security objective

Evaluate and defend AI systems against direct jailbreaks and indirect instruction injection while preserving a reproducible boundary between attacker-controlled content and trusted system instructions.

## Assets

- System/developer instructions
- User and tool-provided content
- Model outputs
- Agent tool-call decisions
- Safety policy enforcement

## Attack surfaces

```
direct user input ───────┐
RAG / retrieved content ─┼──> target model ──> tool/action
tool output ─────────────┘
```

PromptStrike explicitly distinguishes direct user-channel attacks from indirect tool-mediated attacks.

## Threats in scope

- Instruction override / role manipulation
- Jailbreak escalation
- Obfuscated or encoded attack content
- Multi-turn escalation
- Indirect injection through tool output
- Adversarial suffix research in controlled local models

## Evaluation boundary

The attacker should control only the channel being studied. In tool-output mode, the attack payload must not be smuggled into the direct user message. This preserves the experiment's causal interpretation.

## Out of scope

- Universal claims about model safety
- Production abuse against systems without authorization
- Complete semantic understanding of arbitrary natural-language intent
- Provider-specific hidden system prompts

## Defensive objective

The defense layer should provide:

- transparent risk scores
- explainable rule hits
- configurable thresholds
- pre-call and post-call checks
- machine-readable results suitable for CI

A blocked input is evidence that a configured detector fired; it is not proof that the underlying model would otherwise have been compromised.
