# Prompt Engineering Assignment

This directory contains the complete submission for the NLP prompt engineering assignment.

## Deliverables

- `notebooks/Prompt_Engineering_Assignment.ipynb`
- `reports/Prompt_Engineering_Report.pdf`

## Supporting Files

- `run_all.py`: runs all Groq experiments and generates result tables and figures
- `build_notebook.py`: builds and executes the final notebook
- `build_report.py`: generates the PDF report
- `src/assignment_core.py`: shared experiment logic and Groq API wrapper
- `results/`: generated CSV/JSON outputs
- `figures/`: charts used in the report and notebook

## Environment

This assignment expects `GROQ_API_KEY` to be present in the environment.

Example local setup:

```bash
python3 -m venv .venv_prompt
. .venv_prompt/bin/activate
pip install -r prompt_engineering_assignment/requirements.txt
export GROQ_API_KEY=your_key_here
```

## Re-run Workflow

```bash
. .venv_prompt/bin/activate
python prompt_engineering_assignment/run_all.py
python prompt_engineering_assignment/build_notebook.py
python prompt_engineering_assignment/build_report.py
```
