import json
from pathlib import Path

import ollama

USER_ROLE = "user"
ASSISTANT_ROLE = "assistant"
SYSTEM_ROLE = "system"

DEFAULT_MODEL = "qwen2.5"
PASS_THRESHOLD = 7.0

RUNNER_SYSTEM_PROMPT = """Eres un asistente experto que responde preguntas de forma clara y precisa.
Responde directamente a la pregunta sin añadir introducciones, despedidas ni explicaciones adicionales."""

JUDGE_SYSTEM_PROMPT = """Eres un evaluador experto de respuestas de modelos de lenguaje.
Tu tarea es comparar una respuesta generada por un modelo con la respuesta esperada y asignar una puntuación.

Criterios de evaluación:
- Corrección factual: ¿La información es precisa?
- Completitud: ¿Cubre los puntos clave de la respuesta esperada?
- Relevancia: ¿Responde directamente a la pregunta?

Responde ÚNICAMENTE con un objeto JSON con esta estructura exacta:
{"score": <número entero entre 0 y 10>, "rationale": "<explicación breve en una oración>", "passed": <true si score >= 7, false si no>}"""

JUDGE_PREFILL = '{"score": '
JUDGE_STOP_SEQUENCES = ["}\n", "} "]


class ModelRunner:
    """Runs each eval input through the model under test and returns its response."""

    def __init__(self, model: str = DEFAULT_MODEL, system_prompt: str = RUNNER_SYSTEM_PROMPT):
        self.model = model
        self.system_prompt = system_prompt

    def run(self, input_text: str) -> str:
        response = ollama.chat(
            model=self.model,
            messages=[
                {"role": SYSTEM_ROLE, "content": self.system_prompt},
                {"role": USER_ROLE, "content": input_text},
            ],
        )
        return response["message"]["content"].strip()


class EvalJudge:
    """LLM-as-a-judge: scores model output against the expected output.

    Uses the same prefill + stop-sequence technique from eval-data-assets to
    force structured JSON output from the judge model.
    """

    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model

    def judge(self, input_text: str, expected: str, actual: str) -> dict:
        user_message = (
            f"Pregunta: {input_text}\n\n"
            f"Respuesta esperada:\n{expected}\n\n"
            f"Respuesta del modelo:\n{actual}\n\n"
            "Evalúa qué tan bien la respuesta del modelo captura la información de la respuesta esperada."
        )

        response = ollama.chat(
            model=self.model,
            messages=[
                {"role": SYSTEM_ROLE, "content": JUDGE_SYSTEM_PROMPT},
                {"role": USER_ROLE, "content": user_message},
                {"role": ASSISTANT_ROLE, "content": JUDGE_PREFILL},
            ],
            options={"stop": JUDGE_STOP_SEQUENCES},
        )

        raw = JUDGE_PREFILL + response["message"]["content"]
        if not raw.rstrip().endswith("}"):
            raw = raw.rstrip() + "}"

        try:
            parsed = json.loads(raw)
            score = float(parsed.get("score", 0))
            return {
                "score": score,
                "rationale": parsed.get("rationale", ""),
                "passed": parsed.get("passed", score >= PASS_THRESHOLD),
            }
        except json.JSONDecodeError:
            return {"score": 0.0, "rationale": raw, "passed": False}


class EvalWorkflow:
    """Orchestrates the full eval loop: run → judge → summarise → persist."""

    def __init__(self, model: str = DEFAULT_MODEL):
        self.runner = ModelRunner(model=model)
        self.judge = EvalJudge(model=model)

    def run(self, dataset_path: str, output_path: str = "eval_results.json") -> dict:
        with open(dataset_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)

        print(f"Loaded {len(dataset)} records from {dataset_path}\n")

        results = []
        for i, record in enumerate(dataset):
            label = record.get("input", "")[:60]
            print(f"[{i + 1}/{len(dataset)}] {label}...")

            actual = self.runner.run(record["input"])
            judgment = self.judge.judge(record["input"], record["expected_output"], actual)

            results.append(
                {
                    "id": record.get("id", i + 1),
                    "input": record["input"],
                    "expected_output": record["expected_output"],
                    "actual_output": actual,
                    "category": record.get("category", ""),
                    "difficulty": record.get("difficulty", ""),
                    "score": judgment["score"],
                    "rationale": judgment["rationale"],
                    "passed": judgment["passed"],
                }
            )
            status = "PASS" if judgment["passed"] else "FAIL"
            print(f"  [{status}] score={judgment['score']}/10 — {judgment['rationale']}")

        summary = self._summarise(results)
        output = {"summary": summary, "results": results}

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        self._print_summary(summary, output_path)
        return output

    def _summarise(self, results: list) -> dict:
        total = len(results)
        passed = sum(1 for r in results if r["passed"])
        scores = [r["score"] for r in results]

        by_category: dict[str, dict] = {}
        by_difficulty: dict[str, dict] = {}

        for r in results:
            for key, bucket in [("category", by_category), ("difficulty", by_difficulty)]:
                label = r[key]
                if label not in bucket:
                    bucket[label] = {"total": 0, "passed": 0, "scores": []}
                bucket[label]["total"] += 1
                bucket[label]["passed"] += int(r["passed"])
                bucket[label]["scores"].append(r["score"])

        def _aggregate(bucket: dict) -> dict:
            return {
                k: {
                    "total": v["total"],
                    "passed": v["passed"],
                    "pass_rate": round(v["passed"] / v["total"], 3),
                    "avg_score": round(sum(v["scores"]) / len(v["scores"]), 2),
                }
                for k, v in bucket.items()
            }

        return {
            "total": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": round(passed / total, 3) if total else 0,
            "avg_score": round(sum(scores) / len(scores), 2) if scores else 0,
            "by_category": _aggregate(by_category),
            "by_difficulty": _aggregate(by_difficulty),
        }

    def _print_summary(self, summary: dict, output_path: str) -> None:
        print("\n" + "=" * 40)
        print("EVAL SUMMARY")
        print("=" * 40)
        print(f"Total   : {summary['total']}")
        print(f"Passed  : {summary['passed']}")
        print(f"Failed  : {summary['failed']}")
        print(f"Pass rate: {summary['pass_rate']:.1%}")
        print(f"Avg score: {summary['avg_score']:.2f}/10")

        if summary["by_category"]:
            print("\nBy category:")
            for cat, s in summary["by_category"].items():
                print(f"  {cat:<12} pass={s['pass_rate']:.0%}  avg={s['avg_score']:.1f}")

        if summary["by_difficulty"]:
            print("\nBy difficulty:")
            for diff, s in summary["by_difficulty"].items():
                print(f"  {diff:<12} pass={s['pass_rate']:.0%}  avg={s['avg_score']:.1f}")

        print(f"\nResults saved to {output_path}")


def main():
    print("=== Prompt Eval Workflow ===\n")

    default_dataset = "../eval-data-assets/eval_dataset.json"
    raw = input(f"Path to eval dataset [{default_dataset}]: ").strip()
    dataset_path = raw or default_dataset

    if not Path(dataset_path).exists():
        print(f"Error: dataset not found at '{dataset_path}'")
        return

    raw = input("Output path [eval_results.json]: ").strip()
    output_path = raw or "eval_results.json"

    workflow = EvalWorkflow()
    workflow.run(dataset_path, output_path)
