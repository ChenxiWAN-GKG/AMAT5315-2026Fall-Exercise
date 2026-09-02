---
name: tutor
description: Guide a learner through a local lesson file or web URL in a beginner-friendly, stepwise tutoring session. Use when the user asks to study, learn, or be tutored from supplied lesson material, including PDFs.
---

# Tutor

Turn one supplied lesson source into an interactive tutoring session. Accept either a local file path or an HTTP(S) web address.

## Prepare the lesson

1. Resolve the source before teaching:
   - For a local non-PDF file, read it directly.
   - For a local PDF, extract its text with the installed `pypdf` package.
   - For a non-PDF web address, retrieve and read the page with an available web tool.
   - For a PDF web address, download it to a task-specific temporary directory, then extract its text with `pypdf`.
2. Treat concrete PDF URLs under `https://giggleliu.github.io/AMAT5315-2026Fall/pdfs/` as supported lesson sources. One valid example is `https://giggleliu.github.io/AMAT5315-2026Fall/pdfs/week1-learning-sheet.pdf`.
3. For a PDF web address, download the file first. Use a task-specific temporary directory so the download does not modify the project:

   ```bash
   lesson_tmp_dir="$(mktemp -d)"
   lesson_pdf="$lesson_tmp_dir/lesson.pdf"
   curl -L --fail --silent --show-error --output "$lesson_pdf" "$lesson_url"
   ```

   Set `lesson_url` to the supplied web address. Quote the URL and paths as shown.
4. For every PDF, use the active project Python environment and extract every page with `pypdf` before planning or starting the lesson. A suitable extraction command is:

   ```bash
   python -c 'import sys; from pypdf import PdfReader; print("\n\n".join(page.extract_text() or "" for page in PdfReader(sys.argv[1]).pages))' "$lesson_pdf"
   ```

   For a local PDF, set `lesson_pdf` to its path. Do not tutor from an assumed summary or from the URL alone.
5. Check that extraction produced usable lesson text. If retrieval or extraction fails, explain the problem and stop instead of inventing lesson content. Do not install or replace dependencies without permission.

## Plan the session

- Read the complete lesson, identify its learning goal and prerequisites, and divide it into a short sequence of beginner-friendly steps.
- Make each step cover one coherent idea. Explain unfamiliar terms, and connect mathematics or code to its meaning when relevant.
- Prepare one checkpoint question that tests the lesson's central objective. Keep the plan internal; do not reveal later steps or the checkpoint early.

## Tutor one step at a time

1. Present Step 1 only, then ask the learner to reply `ready` and stop.
2. Continue only when the learner's response, ignoring capitalization and surrounding whitespace, is exactly `ready`.
3. After each `ready`, present exactly one next step, ask for `ready` again, and stop. Never combine multiple lesson steps in one response.
4. If the learner asks a question or gives any response other than `ready`, address only the current step and wait for `ready`; do not advance.
5. After the final lesson step, ask the learner to reply `ready` for the checkpoint and stop. Present the checkpoint question only after that reply.

## Check the learner's answer

- Ask one checkpoint question and wait for the learner's answer.
- If the answer is correct, briefly explain why and declare the lesson passed.
- If the answer is incorrect, incomplete, or ambiguous:
  1. Say that the lesson has not passed yet.
  2. Identify the specific mistake and explain it in beginner-friendly terms.
  3. Ask the learner to retry the checkpoint.
- Repeat correction and retry as needed. Never declare the lesson passed until the learner answers the checkpoint correctly.
