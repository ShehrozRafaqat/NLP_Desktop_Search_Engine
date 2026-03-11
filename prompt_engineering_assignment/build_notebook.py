"""Build and execute the prompt engineering assignment notebook."""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

from src.assignment_core import PARAMETER_PROMPT, load_results_bundle


def _build_notebook_cells(base_dir: Path) -> list:
    bundle = load_results_bundle(base_dir)
    best_exp = bundle["parameter_sensitivity"]["summary"]["best_balance_experiment"]
    best_technique = bundle["prompt_optimization"]["summary"]["best_technique"]

    return [
        new_markdown_cell(
            "# Prompt Engineering Assignment\n"
            "## Natural Language Processing\n\n"
            "This notebook documents four Groq-based experiments:\n"
            "1. Parameter sensitivity\n"
            "2. Prompt optimization\n"
            "3. LLM failure analysis\n"
            "4. A multi-step prompt pipeline"
        ),
        new_markdown_cell(
            "### Environment and API Setup\n"
            "The assignment uses Groq's OpenAI-compatible API with the model "
            f"`{bundle['metadata']['model']}`."
        ),
        new_code_cell(
            "from pathlib import Path\n"
            "import sys\n"
            "import inspect\n"
            "import json\n"
            "import pandas as pd\n"
            "from IPython.display import display, Image, Markdown\n\n"
            "CWD = Path.cwd().resolve()\n"
            "BASE_DIR = CWD if (CWD / 'results').exists() else CWD.parent\n"
            "SRC_DIR = BASE_DIR / 'src'\n"
            "if str(SRC_DIR) not in sys.path:\n"
            "    sys.path.insert(0, str(SRC_DIR))\n\n"
            "from assignment_core import (\n"
            "    GroqRunner,\n"
            "    PARAMETER_PROMPT,\n"
            "    PROMPT_VARIANTS,\n"
            "    FAILURE_CASES,\n"
            "    PIPELINE_ARTICLE,\n"
            "    load_results_bundle,\n"
            ")\n\n"
            "bundle = load_results_bundle(BASE_DIR)\n"
            "bundle['metadata']"
        ),
        new_code_cell(
            "print(inspect.getsource(GroqRunner.get_completion))"
        ),
        new_markdown_cell(
            "## Part 1: Parameter Sensitivity Experiment\n\n"
            f"Prompt used:\n\n> {PARAMETER_PROMPT}\n\n"
            "Eight parameter configurations were executed while varying temperature, top-p, "
            "max tokens, frequency penalty, and presence penalty."
        ),
        new_code_cell(
            "parameter_df = pd.DataFrame(bundle['parameter_sensitivity']['experiments'])\n"
            "display(parameter_df[[\n"
            "    'experiment_id', 'temperature', 'top_p', 'max_tokens',\n"
            "    'frequency_penalty', 'presence_penalty', 'word_count',\n"
            "    'coherence', 'creativity', 'repetition', 'topic_drift',\n"
            "    'length_assessment'\n"
            "]])"
        ),
        new_code_cell(
            "for exp_id in ['exp_01', 'exp_04', 'exp_07']:\n"
            "    row = parameter_df.loc[parameter_df['experiment_id'] == exp_id].iloc[0]\n"
            "    print(f'--- {exp_id} ---')\n"
            "    print(row['output_text'])\n"
            "    print()"
        ),
        new_markdown_cell(
            "Parameter observations:\n\n"
            f"- The best balance across the evaluation criteria came from `{best_exp}`.\n"
            "- Lower temperatures tended to produce more stable, coherent responses.\n"
            "- Higher temperatures and higher presence penalties increased novelty but also raised drift risk.\n"
            "- Lower max token limits reduced response length and sometimes forced abrupt endings."
        ),
        new_code_cell(
            "display(Image(filename=str(BASE_DIR / 'figures' / 'parameter_sensitivity.png')))"
        ),
        new_markdown_cell(
            "## Part 2: Prompt Optimization Study\n\n"
            "Base task:\n\n> Explain the causes of World War I.\n\n"
            "Five prompt styles were tested: basic, role-based, structured, few-shot, and chain-of-thought."
        ),
        new_code_cell(
            "prompt_df = pd.DataFrame(bundle['prompt_optimization']['comparisons'])\n"
            "display(prompt_df[[\n"
            "    'technique', 'word_count', 'clarity', 'structure',\n"
            "    'completeness', 'factual_accuracy', 'judge_note'\n"
            "]])"
        ),
        new_code_cell(
            "for _, row in prompt_df[['technique', 'prompt', 'output_text']].iterrows():\n"
            "    print(f\"--- {row['technique']} ---\")\n"
            "    print('PROMPT:')\n"
            "    print(row['prompt'])\n"
            "    print('\\nOUTPUT:')\n"
            "    print(row['output_text'])\n"
            "    print()"
        ),
        new_markdown_cell(
            "Prompt optimization observations:\n\n"
            f"- The highest overall scoring technique was `{best_technique}`.\n"
            "- Structured prompts improved organization and completeness.\n"
            "- Few-shot prompting consistently encouraged a cleaner answer template.\n"
            "- Chain-of-thought prompting tended to add detail, but structure still depended on the final formatting instruction."
        ),
        new_code_cell(
            "display(Image(filename=str(BASE_DIR / 'figures' / 'prompt_optimization.png')))"
        ),
        new_markdown_cell(
            "## Part 3: LLM Failure Analysis\n\n"
            "Three prompts were designed to induce different failure modes: hallucination, ambiguity, "
            "and confident but incorrect reasoning."
        ),
        new_code_cell(
            "failure_df = pd.DataFrame(bundle['failure_analysis']['cases'])\n"
            "display(failure_df[['failure_type', 'prompt', 'why_failure_happened', 'improved_prompt']])"
        ),
        new_code_cell(
            "for _, row in failure_df[['failure_type', 'prompt', 'output_text']].iterrows():\n"
            "    print(f\"--- {row['failure_type']} ---\")\n"
            "    print('PROMPT:')\n"
            "    print(row['prompt'])\n"
            "    print('\\nMODEL OUTPUT:')\n"
            "    print(row['output_text'])\n"
            "    print()"
        ),
        new_markdown_cell(
            "Failure analysis summary:\n\n"
            "- Hallucinations became much more likely when the prompt explicitly rewarded invention.\n"
            "- Ambiguity came from missing referents, not just factual uncertainty.\n"
            "- Confidently wrong answers emerged when the prompt discouraged correction and favored fluent explanation over verification."
        ),
        new_markdown_cell(
            "## Part 4: Multi-Step Prompt Pipeline\n\n"
            "The pipeline uses a short news-style article and performs four API calls:\n"
            "1. Summarize the article\n"
            "2. Extract key facts as JSON\n"
            "3. Classify the topic\n"
            "4. Generate a tweet-length summary"
        ),
        new_code_cell(
            "pipeline = bundle['multi_step_pipeline']\n"
            "print(pipeline['article_text'])"
        ),
        new_code_cell(
            "for stage in pipeline['stages']:\n"
            "    print(f\"--- Stage {stage['stage']}: {stage['name']} ---\")\n"
            "    print('PROMPT:')\n"
            "    print(stage['prompt'])\n"
            "    print('\\nOUTPUT:')\n"
            "    if isinstance(stage['output'], dict):\n"
            "        print(json.dumps(stage['output'], indent=2))\n"
            "    else:\n"
            "        print(stage['output'])\n"
            "    print()"
        ),
        new_markdown_cell(
            "## Conclusions\n\n"
            "- Parameter tuning materially changed creativity, stability, length, and drift.\n"
            "- Prompt structure was a strong lever for answer quality on historical explanation tasks.\n"
            "- Failure modes were often caused by the prompt itself rather than the topic alone.\n"
            "- Multi-step prompting made it easier to control transformations from raw text to structured outputs."
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the prompt engineering notebook")
    parser.add_argument(
        "--base-dir",
        default=str(Path(__file__).resolve().parent),
        help="Prompt assignment base directory",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()
    notebook_path = base_dir / "notebooks" / "Prompt_Engineering_Assignment.ipynb"
    notebook_path.parent.mkdir(parents=True, exist_ok=True)

    nb = new_notebook(cells=_build_notebook_cells(base_dir))
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3 (Prompt Assignment)",
        "language": "python",
        "name": "prompt_assignment_kernel",
    }
    nb.metadata["language_info"] = {"name": "python", "version": "3.12"}

    with open(notebook_path, "w", encoding="utf-8") as handle:
        nbformat.write(nb, handle)

    with open(notebook_path, "r", encoding="utf-8") as handle:
        executed = nbformat.read(handle, as_version=4)
    client = NotebookClient(executed, timeout=600, kernel_name="prompt_assignment_kernel")
    executed = client.execute(cwd=str(notebook_path.parent))
    with open(notebook_path, "w", encoding="utf-8") as handle:
        nbformat.write(executed, handle)

    print(f"Notebook generated at {notebook_path}")


if __name__ == "__main__":
    main()
