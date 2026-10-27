# Architecture Decision Records

This directory holds the project's Architecture Decision Records (ADRs). An ADR captures a single significant decision, its context, and its consequences, so that the reasoning behind the codebase is not lost across time or chat sessions. ADRs are the single source of truth for architecture; when an implementation detail is ambiguous, consult these first.

## Conventions

Records are numbered sequentially and named `NNNN-short-title.md`. Each follows the same skeleton: Status, Date, Context, Decision, Consequences. A decision that overturns an earlier one supersedes it rather than editing history, and both records note the relationship.

Status values used in this project:

| Status | Meaning |
| --- | --- |
| Proposed | Under discussion, not yet adopted |
| Accepted | Adopted and in force |
| Superseded | Replaced by a later ADR (which is referenced) |
| Deprecated | No longer relevant, not replaced |

## Index

| ADR | Title | Status |
| --- | --- | --- |
| [0001](0001-code-maintainability.md) | Code Maintainability and Quality Gates | Accepted |
| [0002](0002-tfb-uncertainty-for-safety-classification.md) | Training-Free Bayesianization for Safety-Classification Uncertainty | Proposed |