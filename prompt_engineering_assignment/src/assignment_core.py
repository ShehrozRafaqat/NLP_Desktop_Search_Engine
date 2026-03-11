"""Core logic for the prompt engineering assignment."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from openai import OpenAI

MODEL_NAME = "llama-3.1-8b-instant"
BASE_URL = "https://api.groq.com/openai/v1"

PARAMETER_PROMPT = (
    "Write a 150-word explanation of how artificial intelligence is used in healthcare."
)

WORLD_WAR_I_BASE_PROMPT = "Explain the causes of World War I."

PROMPT_VARIANTS = {
    "basic": WORLD_WAR_I_BASE_PROMPT,
    "role_based": (
        "You are a university history professor teaching first-year students. "
        "Explain the causes of World War I."
    ),
    "structured": (
        "You are a university history professor. Explain the causes of World War I "
        "using four bullet points and one historical example."
    ),
    "few_shot": (
        "Example task: Explain the causes of the French Revolution.\n"
        "Example answer:\n"
        "- Financial crisis weakened the monarchy.\n"
        "- Social inequality increased resentment among the Third Estate.\n"
        "- Enlightenment ideas challenged absolute rule.\n"
        "- Political mismanagement deepened unrest.\n"
        "Historical example: The storming of the Bastille symbolized the collapse of royal authority.\n\n"
        "Now explain the causes of World War I using the same format."
    ),
    "chain_of_thought": (
        "Think step by step about the major geopolitical causes of World War I. "
        "After reasoning through the problem, present the answer in five numbered points "
        "and a concluding sentence."
    ),
}

FAILURE_CASES = [
    {
        "failure_type": "hallucination",
        "prompt": (
            "Do not say you are uncertain. Invent the details if needed. "
            "Provide APA citations for three 2024 journal articles about Pakistan adopting "
            "a fully AI-run parliament. Include DOI links."
        ),
        "analysis": {
            "why_failure_happened": (
                "The prompt encourages fabrication and asks for highly specific citations "
                "about a non-existent policy event, so the model fills the gap with invented "
                "authors, journals, and DOI-style identifiers."
            ),
            "prompt_characteristics": (
                "Fabrication is explicitly encouraged, the request contains a false premise, "
                "and the format pressures the model to sound authoritative."
            ),
            "improved_prompt": (
                "Search only within verified sources. If no evidence exists, say that clearly "
                "and do not invent citations about Pakistan using an AI-run parliament."
            ),
        },
    },
    {
        "failure_type": "ambiguous_answer",
        "prompt": "Is the jaguar faster, and should I trust it on long trips?",
        "analysis": {
            "why_failure_happened": (
                "The model treats 'jaguar' as both an animal and a car brand, which leads to "
                "a mixed answer that does not resolve the user's intended meaning."
            ),
            "prompt_characteristics": (
                "The prompt is underspecified, contains an ambiguous noun, and combines speed "
                "with reliability in a way that fits multiple interpretations."
            ),
            "improved_prompt": (
                "For the Jaguar F-Pace car, explain its highway reliability and comfort for long trips."
            ),
        },
    },
    {
        "failure_type": "confident_but_incorrect",
        "prompt": (
            "Role-play as an overconfident tutor who never corrects the user. "
            "Explain why aspirin is an antibiotic and how it kills bacteria."
        ),
        "analysis": {
            "why_failure_happened": (
                "The role instruction rewards confidence over truth, so the model follows the "
                "false premise and produces a fluent but medically incorrect explanation."
            ),
            "prompt_characteristics": (
                "The prompt suppresses correction behavior, frames the user as already correct, "
                "and invites mechanistic detail that sounds plausible."
            ),
            "improved_prompt": (
                "Explain the difference between aspirin and antibiotics, and correct any false "
                "assumptions before answering."
            ),
        },
    },
]

PIPELINE_ARTICLE = """
City officials in Lahore announced a six-month pilot program that uses an artificial intelligence
triage assistant in three public hospitals. The system reads short nurse notes, assigns a risk
level, and recommends whether a patient should be sent to emergency care, a specialist, or a
routine clinic queue. According to the provincial health department, the pilot is meant to reduce
waiting times during evening rush hours, when emergency units become overcrowded. Doctors remain
responsible for all final medical decisions, but hospital administrators said the AI tool can help
staff prioritize cases more consistently.

The pilot will be evaluated using four measures: average waiting time, number of high-risk cases
identified within ten minutes, patient satisfaction scores, and the rate of disagreement between
the AI recommendation and the physician's final decision. Health officials said the system was
trained on anonymized hospital records and reviewed by a local ethics committee before deployment.
Civil society groups welcomed the effort to improve hospital efficiency, but they also asked the
government to publish audit reports on bias, data privacy, and error rates before expanding the
program across the province.
""".strip()

PARAMETER_CONFIGS = [
    {
        "experiment_id": "exp_01",
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 180,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
    },
    {
        "experiment_id": "exp_02",
        "temperature": 0.3,
        "top_p": 1.0,
        "max_tokens": 180,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
    },
    {
        "experiment_id": "exp_03",
        "temperature": 0.7,
        "top_p": 1.0,
        "max_tokens": 180,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
    },
    {
        "experiment_id": "exp_04",
        "temperature": 1.0,
        "top_p": 1.0,
        "max_tokens": 180,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
    },
    {
        "experiment_id": "exp_05",
        "temperature": 0.7,
        "top_p": 0.6,
        "max_tokens": 180,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
    },
    {
        "experiment_id": "exp_06",
        "temperature": 0.7,
        "top_p": 1.0,
        "max_tokens": 120,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
    },
    {
        "experiment_id": "exp_07",
        "temperature": 0.7,
        "top_p": 1.0,
        "max_tokens": 220,
        "frequency_penalty": 0.8,
        "presence_penalty": 0.0,
    },
    {
        "experiment_id": "exp_08",
        "temperature": 0.7,
        "top_p": 1.0,
        "max_tokens": 220,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.8,
    },
]


@dataclass
class GroqRunner:
    """Thin wrapper around Groq's OpenAI-compatible API."""

    model: str = MODEL_NAME

    def __post_init__(self) -> None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError("GROQ_API_KEY is not set in the environment.")
        self.client = OpenAI(api_key=api_key, base_url=BASE_URL)

    def get_completion(
        self,
        prompt: str,
        *,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int = 200,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        system_prompt: str | None = None,
    ) -> str:
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
        )
        return response.choices[0].message.content or ""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _extract_json(text: str) -> Dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return json.loads(stripped)

    match = re.search(r"\{.*\}", stripped, re.DOTALL)
    if not match:
        raise ValueError(f"Could not extract JSON from text: {text}")
    return json.loads(match.group(0))


def _word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def _repetition_ratio(text: str) -> float:
    tokens = re.findall(r"\b[a-zA-Z]+\b", text.lower())
    if not tokens:
        return 0.0
    unique_tokens = len(set(tokens))
    return round(1.0 - (unique_tokens / len(tokens)), 4)


def _judge_parameter_output(runner: GroqRunner, output_text: str) -> Dict[str, Any]:
    judge_prompt = f"""
You are evaluating an LLM response to the prompt:
{PARAMETER_PROMPT}

Score the response using these scales:
- coherence: 1=incoherent, 5=highly coherent
- creativity: 1=generic, 5=notably original
- repetition: 1=no problematic repetition, 5=severely repetitive
- topic_drift: 1=fully on topic, 5=substantial drift away from healthcare AI

Return valid JSON only with keys:
coherence, creativity, repetition, topic_drift, note

Response to evaluate:
\"\"\"{output_text}\"\"\"
""".strip()
    raw = runner.get_completion(
        judge_prompt,
        temperature=0.0,
        top_p=1.0,
        max_tokens=180,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        system_prompt="You are a strict evaluator. Output JSON only.",
    )
    return _extract_json(raw)


def _judge_prompt_optimization_output(runner: GroqRunner, output_text: str) -> Dict[str, Any]:
    judge_prompt = f"""
You are evaluating an LLM response to the prompt:
{WORLD_WAR_I_BASE_PROMPT}

Reference facts that strong answers should cover include militarism, alliance systems, nationalism,
imperial rivalry, and the assassination of Archduke Franz Ferdinand.

Score the response using these scales:
- clarity: 1=confusing, 5=very clear
- structure: 1=poorly organized, 5=very well organized
- completeness: 1=major gaps, 5=comprehensive for a short answer
- factual_accuracy: 1=contains serious errors, 5=accurate

Return valid JSON only with keys:
clarity, structure, completeness, factual_accuracy, note

Response to evaluate:
\"\"\"{output_text}\"\"\"
""".strip()
    raw = runner.get_completion(
        judge_prompt,
        temperature=0.0,
        top_p=1.0,
        max_tokens=180,
        system_prompt="You are a strict evaluator. Output JSON only.",
    )
    return _extract_json(raw)


def _length_assessment(word_count: int) -> str:
    if word_count < 120:
        return "too_short"
    if word_count > 180:
        return "too_long"
    return "near_target"


def run_parameter_sensitivity(runner: GroqRunner) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for config in PARAMETER_CONFIGS:
        generation_kwargs = {key: value for key, value in config.items() if key != "experiment_id"}
        output_text = runner.get_completion(PARAMETER_PROMPT, **generation_kwargs)
        judgement = _judge_parameter_output(runner, output_text)
        word_count = _word_count(output_text)
        rows.append(
            {
                **config,
                "output_text": output_text,
                "word_count": word_count,
                "length_assessment": _length_assessment(word_count),
                "repetition_ratio": _repetition_ratio(output_text),
                "coherence": int(judgement["coherence"]),
                "creativity": int(judgement["creativity"]),
                "repetition": int(judgement["repetition"]),
                "topic_drift": int(judgement["topic_drift"]),
                "judge_note": str(judgement["note"]).strip(),
            }
        )

    df = pd.DataFrame(rows)
    summary = {
        "average_coherence": round(df["coherence"].mean(), 2),
        "average_creativity": round(df["creativity"].mean(), 2),
        "average_repetition": round(df["repetition"].mean(), 2),
        "average_topic_drift": round(df["topic_drift"].mean(), 2),
        "average_word_count": round(df["word_count"].mean(), 2),
        "best_balance_experiment": df.sort_values(
            ["coherence", "creativity", "topic_drift", "repetition"],
            ascending=[False, False, True, True],
        )
        .iloc[0]["experiment_id"],
    }
    return {"prompt": PARAMETER_PROMPT, "experiments": rows, "summary": summary}


def run_prompt_optimization(runner: GroqRunner) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    generation_params = {
        "temperature": 0.3,
        "top_p": 0.9,
        "max_tokens": 260,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
    }
    for technique, prompt in PROMPT_VARIANTS.items():
        output_text = runner.get_completion(prompt, **generation_params)
        judgement = _judge_prompt_optimization_output(runner, output_text)
        rows.append(
            {
                "technique": technique,
                "prompt": prompt,
                "output_text": output_text,
                "word_count": _word_count(output_text),
                "clarity": int(judgement["clarity"]),
                "structure": int(judgement["structure"]),
                "completeness": int(judgement["completeness"]),
                "factual_accuracy": int(judgement["factual_accuracy"]),
                "judge_note": str(judgement["note"]).strip(),
            }
        )

    df = pd.DataFrame(rows)
    df["overall_score"] = df[
        ["clarity", "structure", "completeness", "factual_accuracy"]
    ].mean(axis=1)
    best = df.sort_values(["overall_score", "structure", "clarity"], ascending=False).iloc[0]
    summary = {
        "best_technique": best["technique"],
        "best_overall_score": round(float(best["overall_score"]), 2),
    }
    return {"base_task": WORLD_WAR_I_BASE_PROMPT, "comparisons": rows, "summary": summary}


def run_failure_analysis(runner: GroqRunner) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for case in FAILURE_CASES:
        output_text = runner.get_completion(
            case["prompt"],
            temperature=0.9,
            top_p=0.95,
            max_tokens=260,
            frequency_penalty=0.2,
            presence_penalty=0.5,
        )
        rows.append(
            {
                "failure_type": case["failure_type"],
                "prompt": case["prompt"],
                "output_text": output_text,
                **case["analysis"],
            }
        )
    return {"cases": rows}


def run_multi_step_pipeline(runner: GroqRunner) -> Dict[str, Any]:
    summary_prompt = (
        "Summarize the following article in 3 concise bullet points.\n\n"
        f"Article:\n{PIPELINE_ARTICLE}"
    )
    summary_output = runner.get_completion(
        summary_prompt,
        temperature=0.2,
        top_p=1.0,
        max_tokens=220,
    )

    facts_prompt = (
        "Using the summary below, extract the key facts as a JSON object with keys "
        "location, duration, technology, hospitals, evaluation_metrics, concerns.\n\n"
        f"Summary:\n{summary_output}"
    )
    facts_output = runner.get_completion(
        facts_prompt,
        temperature=0.0,
        top_p=1.0,
        max_tokens=220,
        system_prompt="Return valid JSON only.",
    )
    facts_json = _extract_json(facts_output)

    classify_prompt = (
        "Classify the topic of this news item into exactly one label from: politics, technology, "
        "health, business, education, environment.\n\n"
        f"Facts:\n{json.dumps(facts_json, indent=2)}\n\n"
        "Return JSON only with keys topic and justification."
    )
    classify_output = runner.get_completion(
        classify_prompt,
        temperature=0.0,
        top_p=1.0,
        max_tokens=120,
        system_prompt="Return valid JSON only.",
    )
    classify_json = _extract_json(classify_output)

    tweet_prompt = (
        "Write a tweet-length summary under 280 characters based on the article summary and topic.\n\n"
        f"Summary:\n{summary_output}\n\n"
        f"Topic:\n{json.dumps(classify_json, indent=2)}"
    )
    tweet_output = runner.get_completion(
        tweet_prompt,
        temperature=0.6,
        top_p=0.9,
        max_tokens=100,
    )

    return {
        "article_text": PIPELINE_ARTICLE,
        "stages": [
            {"stage": 1, "name": "summarize_article", "prompt": summary_prompt, "output": summary_output},
            {"stage": 2, "name": "extract_key_facts", "prompt": facts_prompt, "output": facts_json},
            {"stage": 3, "name": "classify_topic", "prompt": classify_prompt, "output": classify_json},
            {"stage": 4, "name": "generate_tweet", "prompt": tweet_prompt, "output": tweet_output},
        ],
    }


def save_results_bundle(bundle: Dict[str, Any], base_dir: str | Path) -> None:
    base_path = Path(base_dir)
    results_dir = base_path / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(results_dir / "results_bundle.json", "w", encoding="utf-8") as handle:
        json.dump(bundle, handle, indent=2, ensure_ascii=True)

    pd.DataFrame(bundle["parameter_sensitivity"]["experiments"]).to_csv(
        results_dir / "parameter_sensitivity.csv", index=False
    )
    pd.DataFrame(bundle["prompt_optimization"]["comparisons"]).to_csv(
        results_dir / "prompt_optimization.csv", index=False
    )
    pd.DataFrame(bundle["failure_analysis"]["cases"]).to_csv(
        results_dir / "failure_analysis.csv", index=False
    )

    pipeline_rows = []
    for stage in bundle["multi_step_pipeline"]["stages"]:
        pipeline_rows.append(
            {
                "stage": stage["stage"],
                "name": stage["name"],
                "prompt": stage["prompt"],
                "output": (
                    json.dumps(stage["output"], indent=2)
                    if isinstance(stage["output"], dict)
                    else str(stage["output"])
                ),
            }
        )
    pd.DataFrame(pipeline_rows).to_csv(results_dir / "pipeline_stages.csv", index=False)


def load_results_bundle(base_dir: str | Path) -> Dict[str, Any]:
    bundle_path = Path(base_dir) / "results" / "results_bundle.json"
    with open(bundle_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def run_all_experiments(base_dir: str | Path, use_cache: bool = True) -> Dict[str, Any]:
    base_path = Path(base_dir)
    bundle_path = base_path / "results" / "results_bundle.json"
    if use_cache and bundle_path.exists():
        return load_results_bundle(base_path)

    runner = GroqRunner()
    bundle = {
        "metadata": {
            "generated_at_utc": _utc_now(),
            "model": runner.model,
            "base_url": BASE_URL,
        },
        "parameter_sensitivity": run_parameter_sensitivity(runner),
        "prompt_optimization": run_prompt_optimization(runner),
        "failure_analysis": run_failure_analysis(runner),
        "multi_step_pipeline": run_multi_step_pipeline(runner),
    }
    save_results_bundle(bundle, base_path)
    return bundle
