# Repository guidance

## Repository purpose and owner

This repository belongs to Chenxi Wan, a physics student and beginner programmer taking AMAT5315 in Fall 2026. It stores coursework and exercises for that course.

This repository holds Chenxi's weekly exercises in `week1/`, `week2/`, `week3/`, and so on. Treat each `weekN/` directory as the work area for that week's exercise. Week 1 covers AI agents and Git, including repository and Git-history setup, a tested Monte Carlo pi estimator, persistent agent memory, and a reusable tutor skill.

Memory probe: W1-MEMORY-5315

## How to work with Chenxi

- Guide Chenxi step by step as a beginner.
- Before making a significant change, explain unfamiliar commands and why each important step is needed.
- For a straightforward task that Chenxi explicitly requests, implement it directly and explain afterward what changed and why.
- Explain unfamiliar Python syntax and programming concepts in beginner-friendly language.
- When mathematics or numerical methods are involved, explain the mathematical idea and how it maps to the code step by step.
- The course allows an agent to write most of the code, but Chenxi remains responsible for stating the problem, defining or reviewing correctness, independently checking the agent's work, personally performing any verification assigned to the student, and deciding what to trust.
- Do not perform a step that an exercise assigns personally to Chenxi, such as writing a specification, carrying out a manual negative control, reviewing a draft, or running an independent verification.

## Scope and sources of truth

- Before editing, inspect `git status` and the files relevant to the requested task. Preserve existing work and avoid unrelated changes.
- Normally limit changes to the requested `weekN/` directory. Preserve completed weeks unless Chenxi explicitly requests a cross-week or repository-level change.
- Modify course-wide files such as `AGENTS.md`, `README.md`, `setup.txt`, or project skills only when an exercise requires it or Chenxi explicitly requests it.
- Treat each weekly `SPEC.md` as the authoritative statement of requirements. Do not modify it unless Chenxi explicitly asks.
- Tests should implement and protect the corresponding `SPEC.md`. Do not weaken or change a test merely to make an implementation pass.
- Modify a test only when Chenxi explicitly requests it or when the test is demonstrably inconsistent with the `SPEC.md`. Explain and demonstrate that inconsistency before changing the test.
- Normally edit the exercise implementation while preserving its specification and tests.

## Python workflow and validation

- For Python work, use the course virtual environment at `~/.venvs/AMAT5315` and verify that its Python interpreter is active before running tests.
- Run the relevant `pytest` checks after making code changes. Report what passed, what failed, and any validation that could not be completed.
- After changes, inspect `git diff` and `git status` to verify and report exactly what changed.
- Do not install, remove, or upgrade dependencies unless the task requires it. Ask Chenxi before adding or changing a dependency.

## Implementation style

- Prefer simple, beginner-readable Python. Avoid advanced abstractions unless the exercise specifically requires them.
- Keep implementations as simple as possible while satisfying the `SPEC.md` and tests.
- Use clear variable and function names.
- Add type hints and short docstrings when they improve readability without adding unnecessary complexity.
- Comment non-obvious logic; do not comment every line.

## Git discipline

- Do not stage, commit, push, merge, or change remotes unless an exercise or Chenxi explicitly requests the specific action.
- When a commit is requested, keep it small and aligned with the relevant exercise step.
- Never discard or overwrite Chenxi's unrelated changes.
