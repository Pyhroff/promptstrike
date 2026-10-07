# PromptStrike Evaluation Methodology

## Goal

Compare attack strategies and defensive controls under a controlled, reproducible experiment design.

## Attack matrix

Track at least:

- PAIR
- TAP
- Crescendo
- GCG where local white-box access is available
- direct user channel
- indirect tool-output channel

## Required experiment metadata

Every campaign should record:

- target model identifier
- provider/adapter
- attack algorithm
- channel
- seed where supported
- iteration/query budget
- defense configuration
- number of trials
- success criterion

## Metrics

For attack evaluation:

- attack success rate
- median/mean queries to success
- failure rate
- defense block rate

For defense evaluation:

- true positives / false positives
- precision, recall, F1
- false-positive rate
- latency overhead

Report confidence intervals when trial counts are large enough to justify them.

## Causal integrity

Direct and indirect injection experiments should be evaluated separately. A result from a direct user jailbreak should not be presented as evidence that the same payload succeeds when delivered through tool output.

## Reproducibility contract

A published result should identify:

1. PromptStrike commit/version
2. target model/version
3. attack implementation/version
4. campaign configuration
5. trial count
6. success/judge rubric
7. defense configuration

## Research questions

- Which attacks transfer from direct user input to tool-mediated channels?
- How much does a heuristic shield reduce attack success?
- What false-positive cost appears as input defenses become stricter?
- How stable are results across model families and providers?
