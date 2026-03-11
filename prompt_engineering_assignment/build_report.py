"""Generate the PDF report for the prompt engineering assignment."""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from src.assignment_core import load_results_bundle


def _clean(text: str, width: int = 110) -> str:
    return textwrap.fill(text.replace("\n", " "), width=width)


def _paragraph(text: str, style: ParagraphStyle) -> Paragraph:
    escaped = (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", "<br/>")
    )
    return Paragraph(escaped, style)


def _table(data, col_widths=None):
    tbl = Table(data, colWidths=col_widths, repeatRows=1)
    tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#264653")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.HexColor("#f2f2f2")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return tbl


def _add_page_number(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 9)
    canvas.drawRightString(560, 20, f"Page {doc.page}")
    canvas.restoreState()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the prompt engineering PDF report")
    parser.add_argument(
        "--base-dir",
        default=str(Path(__file__).resolve().parent),
        help="Prompt assignment base directory",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()
    bundle = load_results_bundle(base_dir)
    param_df = pd.DataFrame(bundle["parameter_sensitivity"]["experiments"])
    prompt_df = pd.DataFrame(bundle["prompt_optimization"]["comparisons"])
    failure_df = pd.DataFrame(bundle["failure_analysis"]["cases"])
    pipeline = bundle["multi_step_pipeline"]

    report_path = base_dir / "reports" / "Prompt_Engineering_Report.pdf"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(report_path),
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=42,
        bottomMargin=35,
    )

    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="BodySmall",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=10,
            leading=14,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="HeadingSmall",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=14,
            leading=18,
            spaceBefore=6,
            spaceAfter=8,
        )
    )

    story = []

    story.append(_paragraph("Prompt Engineering Assignment Report", styles["Title"]))
    story.append(_paragraph("Course: Natural Language Processing", styles["Heading3"]))
    story.append(
        _paragraph(
            "This report presents four Groq-based experiments designed to study how prompt design "
            "and generation parameters affect large language model behavior. All experiments were "
            "executed with the OpenAI-compatible Groq API using the llama-3.1-8b-instant model.",
            styles["BodySmall"],
        )
    )
    story.append(Spacer(1, 0.15 * inch))

    story.append(_paragraph("1. Introduction", styles["HeadingSmall"]))
    story.append(
        _paragraph(
            "The objective of this assignment was to study prompt engineering as an empirical process. "
            "Instead of treating prompting as a single instruction-writing task, the experiments varied "
            "sampling parameters, prompt structure, and multi-stage workflows to observe how these choices "
            "change coherence, creativity, factual reliability, and task control.",
            styles["BodySmall"],
        )
    )
    story.append(
        _paragraph(
            "The full implementation was written in Python, and the workflow generated structured CSV "
            "results, figures, an executed notebook, and this PDF report. Each section below links the "
            "experimental design to concrete outputs collected from the model.",
            styles["BodySmall"],
        )
    )

    story.append(_paragraph("2. Experimental Setup", styles["HeadingSmall"]))
    story.append(
        _paragraph(
            f"Model: {bundle['metadata']['model']}. API provider: Groq. "
            "The experiments used the chat completion endpoint through Groq's OpenAI-compatible base URL. "
            "Evaluation tables were generated programmatically with pandas, and figures were created with matplotlib.",
            styles["BodySmall"],
        )
    )
    story.append(
        _paragraph(
            "For Part 1, eight parameter configurations were tested while varying temperature, top_p, "
            "max_tokens, frequency_penalty, and presence_penalty. For Part 2, five prompt styles were "
            "compared on a common history question. Part 3 probed three failure modes, and Part 4 built "
            "a four-stage prompt pipeline over a news-style article.",
            styles["BodySmall"],
        )
    )

    story.append(PageBreak())

    story.append(_paragraph("3. Parameter Sensitivity Analysis", styles["HeadingSmall"]))
    story.append(
        _paragraph(
            "The parameter sensitivity experiment used the prompt: "
            "\"Write a 150-word explanation of how artificial intelligence is used in healthcare.\" "
            "Each run was rated on coherence, creativity, repetition, topic drift, and output length. "
            "The table below summarizes the quantitative results.",
            styles["BodySmall"],
        )
    )
    param_table = [["Exp", "Temp", "Top_p", "MaxTok", "Freq", "Presence", "Words", "Coh", "Creat", "Rep", "Drift"]]
    for row in param_df[
        [
            "experiment_id",
            "temperature",
            "top_p",
            "max_tokens",
            "frequency_penalty",
            "presence_penalty",
            "word_count",
            "coherence",
            "creativity",
            "repetition",
            "topic_drift",
        ]
    ].itertuples(index=False):
        param_table.append(list(row))
    story.append(_table(param_table, col_widths=[50, 36, 36, 42, 42, 48, 40, 28, 32, 28, 32]))
    story.append(Spacer(1, 0.15 * inch))

    story.append(
        _paragraph(
            f"The best overall balance came from {bundle['parameter_sensitivity']['summary']['best_balance_experiment']}. "
            "Lower temperatures consistently improved coherence, while higher temperatures and higher "
            "presence penalties made responses more varied but also more prone to topic drift. "
            "Shorter max token limits reduced length and sometimes truncated supporting details.",
            styles["BodySmall"],
        )
    )
    story.append(
        Image(str(base_dir / "figures" / "parameter_sensitivity.png"), width=6.5 * inch, height=5.1 * inch)
    )

    story.append(PageBreak())

    story.append(_paragraph("4. Prompt Engineering Experiments", styles["HeadingSmall"]))
    story.append(
        _paragraph(
            "The prompt optimization study compared five prompting techniques on the base task "
            "\"Explain the causes of World War I.\" The goal was to test whether prompt structure alone "
            "could improve clarity, organization, completeness, and factual quality.",
            styles["BodySmall"],
        )
    )
    prompt_table = [["Technique", "Words", "Clarity", "Structure", "Complete", "Accuracy"]]
    for row in prompt_df[
        ["technique", "word_count", "clarity", "structure", "completeness", "factual_accuracy"]
    ].itertuples(index=False):
        prompt_table.append(list(row))
    story.append(_table(prompt_table, col_widths=[95, 45, 45, 50, 55, 55]))
    story.append(Spacer(1, 0.12 * inch))
    story.append(
        _paragraph(
            f"The highest scoring technique was {bundle['prompt_optimization']['summary']['best_technique']}. "
            "Structured and few-shot prompts produced the clearest improvements in answer organization. "
            "Role-based prompting improved tone and pedagogical framing, while chain-of-thought prompting "
            "increased detail but still benefited from explicit formatting constraints.",
            styles["BodySmall"],
        )
    )
    story.append(
        _paragraph(
            "A key observation is that prompt structure often mattered more than temperature changes for this "
            "fact-explanation task. The best answers did not simply add more information; they presented the "
            "same core facts in a more legible form.",
            styles["BodySmall"],
        )
    )
    story.append(
        Image(str(base_dir / "figures" / "prompt_optimization.png"), width=6.5 * inch, height=5.1 * inch)
    )

    story.append(PageBreak())

    story.append(_paragraph("5. Failure Analysis", styles["HeadingSmall"]))
    story.append(
        _paragraph(
            "Three prompts were intentionally designed to provoke incorrect or misleading behavior. "
            "The resulting outputs demonstrate that failure modes are often prompt-induced rather than purely model-internal.",
            styles["BodySmall"],
        )
    )
    for row in failure_df.itertuples(index=False):
        story.append(_paragraph(f"Failure type: {row.failure_type}", styles["Heading3"]))
        story.append(_paragraph(f"Prompt: {_clean(row.prompt, 95)}", styles["BodySmall"]))
        story.append(
            _paragraph(
                f"Observed output excerpt: {_clean(str(row.output_text)[:500], 95)}",
                styles["BodySmall"],
            )
        )
        story.append(
            _paragraph(f"Why it failed: {_clean(row.why_failure_happened, 95)}", styles["BodySmall"])
        )
        story.append(
            _paragraph(
                f"Prompt characteristics: {_clean(row.prompt_characteristics, 95)}",
                styles["BodySmall"],
            )
        )
        story.append(
            _paragraph(f"Improved prompt: {_clean(row.improved_prompt, 95)}", styles["BodySmall"])
        )

    story.append(PageBreak())

    story.append(_paragraph("6. Multi-Step Prompt Pipeline", styles["HeadingSmall"]))
    story.append(
        _paragraph(
            "The final experiment implemented a four-stage prompt pipeline over a news-style article about "
            "an AI triage pilot in Lahore hospitals. Each stage consumed the previous stage's output, which "
            "demonstrated how prompt chaining can transform unstructured text into compact downstream artifacts.",
            styles["BodySmall"],
        )
    )
    pipeline_table = [["Stage", "Name", "Output summary"]]
    for stage in pipeline["stages"]:
        pipeline_table.append(
            [
                stage["stage"],
                stage["name"],
                _clean(
                    stage["output"] if isinstance(stage["output"], str) else str(stage["output"]),
                    70,
                ),
            ]
        )
    story.append(_table(pipeline_table, col_widths=[35, 110, 310]))
    story.append(Spacer(1, 0.12 * inch))
    story.append(
        _paragraph(
            "The pipeline showed clear information flow: a summary became structured facts, the facts became "
            "a topic label, and both together produced a short social-media style summary. This is a practical "
            "pattern for building controllable LLM applications because each stage narrows the task definition.",
            styles["BodySmall"],
        )
    )
    story.append(
        _paragraph(
            "The main risk in chained prompting is error propagation. If the first summary omits a critical fact, "
            "every later stage inherits that omission. Even so, separating tasks by stage improved output clarity "
            "and reduced the chance of mixing extraction, classification, and stylistic compression in one prompt.",
            styles["BodySmall"],
        )
    )

    story.append(_paragraph("7. Observations and Conclusions", styles["HeadingSmall"]))
    story.append(
        _paragraph(
            "The assignment produced four broad conclusions. First, generation parameters directly affect answer "
            "style, length, and stability. Second, explicit prompt structure is a strong quality lever for many "
            "educational tasks. Third, misleading prompts can reliably trigger hallucination, ambiguity, and "
            "confidently wrong explanations. Fourth, multi-step pipelines provide better control than monolithic prompts "
            "when the task involves multiple transformations.",
            styles["BodySmall"],
        )
    )
    story.append(
        _paragraph(
            "Overall, the experiments show that prompt engineering is best treated as a design-and-evaluation workflow. "
            "Effective prompts specify task, role, structure, and guardrails clearly, while robust LLM applications "
            "benefit from staged processing, explicit verification, and conservative defaults for factual tasks.",
            styles["BodySmall"],
        )
    )

    doc.build(story, onFirstPage=_add_page_number, onLaterPages=_add_page_number)
    print(f"Report generated at {report_path}")


if __name__ == "__main__":
    main()
