"""
Prompt Eval Engineer — Workflow de evaluación y prompt engineering iterativo.

Porta el flujo completo del notebook 002_prompting_completed.ipynb a un paquete
ejecutable con Ollama como backend local.

Flujo:
  1. generate_dataset  — genera test cases para una tarea dada
  2. run_evaluation    — ejecuta el prompt contra cada test case y puntúa el resultado
  3. informe HTML      — escribe un reporte visual con scores y razonamientos

Técnicas de prompt engineering aplicadas:
  - Prefill + stop sequences para forzar salida JSON fiable
  - System prompt para especializar el rol del modelo
  - Few-shot examples en el prompt de generación de test cases
  - Temperature diferenciada (alta para generación creativa, baja para evaluación)
  - Concurrent execution para acelerar la generación y evaluación
"""

import json
import concurrent.futures
from statistics import mean
from textwrap import dedent

import ollama

# ──────────────────────────────────────────────
# Configuración global
# ──────────────────────────────────────────────

MODEL: str = "qwen2.5"  # Modelo Ollama local


# ──────────────────────────────────────────────
# Helpers de mensajes (formato Ollama)
# ──────────────────────────────────────────────

def _msg(role: str, text: str) -> dict:
    return {"role": role, "content": text}


def add_user_message(messages: list, text: str) -> None:
    messages.append(_msg("user", text))


def add_assistant_message(messages: list, text: str) -> None:
    """Añade un mensaje de asistente.  Con Ollama/qwen2.5 actúa como prefill:
    el modelo continúa desde este texto en lugar de tratarlo como turno completo."""
    messages.append(_msg("assistant", text))


# ──────────────────────────────────────────────
# Función de chat (wrapper Ollama)
# ──────────────────────────────────────────────

def chat(
    messages: list,
    system: str | None = None,
    temperature: float = 1.0,
    stop_sequences: list[str] | None = None,
) -> str:
    """Llama al modelo local vía Ollama y devuelve el texto de la respuesta.

    Si se pasa `system`, se inyecta como primer mensaje con role "system".
    Los stop_sequences se pasan como opción de Ollama para cortar la generación.
    """
    ollama_messages: list[dict] = []

    if system:
        ollama_messages.append(_msg("system", system))

    for msg in messages:
        # Soporta tanto el formato Bedrock (content=[{text}]) como el plano (content=str)
        content = msg["content"]
        if isinstance(content, list):
            content = content[0].get("text", "")
        ollama_messages.append(_msg(msg["role"], content))

    options: dict = {"temperature": temperature}
    if stop_sequences:
        options["stop"] = stop_sequences

    response = ollama.chat(model=MODEL, messages=ollama_messages, options=options)
    return response["message"]["content"]


# ──────────────────────────────────────────────
# Generador de reporte HTML
# ──────────────────────────────────────────────

def generate_prompt_evaluation_report(evaluation_results: list) -> str:
    total_tests = len(evaluation_results)
    scores = [r["score"] for r in evaluation_results]
    avg_score = mean(scores) if scores else 0
    pass_rate = (
        100 * len([s for s in scores if s >= 7]) / total_tests if total_tests else 0
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Prompt Evaluation Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; color: #333; }}
    .header {{ background: #f0f0f0; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
    .summary-stats {{ display: flex; gap: 10px; flex-wrap: wrap; }}
    .stat-box {{ background: #fff; border-radius: 5px; padding: 15px; box-shadow: 0 2px 5px rgba(0,0,0,.1); flex-basis: 30%; min-width: 180px; }}
    .stat-value {{ font-size: 24px; font-weight: bold; margin-top: 5px; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
    th {{ background: #4a4a4a; color: #fff; text-align: left; padding: 12px; }}
    td {{ padding: 10px; border-bottom: 1px solid #ddd; vertical-align: top; width: 20%; }}
    tr:nth-child(even) {{ background: #f9f9f9; }}
    .score {{ font-weight: bold; padding: 5px 10px; border-radius: 3px; display: inline-block; }}
    .score-high {{ background: #c8e6c9; color: #2e7d32; }}
    .score-medium {{ background: #fff9c4; color: #f57f17; }}
    .score-low {{ background: #ffcdd2; color: #c62828; }}
    .output pre {{ background: #f5f5f5; border: 1px solid #ddd; border-radius: 4px; padding: 10px;
                   font-family: monospace; font-size: 13px; white-space: pre-wrap; word-wrap: break-word; }}
    .score-col {{ width: 80px; }}
  </style>
</head>
<body>
  <div class="header">
    <h1>Prompt Evaluation Report</h1>
    <div class="summary-stats">
      <div class="stat-box"><div>Total Test Cases</div><div class="stat-value">{total_tests}</div></div>
      <div class="stat-box"><div>Average Score</div><div class="stat-value">{avg_score:.1f} / 10</div></div>
      <div class="stat-box"><div>Pass Rate (&ge;7)</div><div class="stat-value">{pass_rate:.1f}%</div></div>
    </div>
  </div>
  <table>
    <thead>
      <tr>
        <th>Scenario</th>
        <th>Prompt Inputs</th>
        <th>Solution Criteria</th>
        <th>Output</th>
        <th>Score</th>
        <th>Reasoning</th>
      </tr>
    </thead>
    <tbody>
"""

    for result in evaluation_results:
        inputs_html = "<br>".join(
            f"<strong>{k}:</strong> {v}"
            for k, v in result["test_case"]["prompt_inputs"].items()
        )
        criteria_html = "<br>• ".join(result["test_case"]["solution_criteria"])
        score = result["score"]
        cls = "score-high" if score >= 8 else ("score-low" if score <= 5 else "score-medium")

        html += f"""
      <tr>
        <td>{result["test_case"]["scenario"]}</td>
        <td>{inputs_html}</td>
        <td>• {criteria_html}</td>
        <td class="output"><pre>{result["output"]}</pre></td>
        <td class="score-col"><span class="score {cls}">{score}</span></td>
        <td>{result["reasoning"]}</td>
      </tr>"""

    html += """
    </tbody>
  </table>
</body>
</html>"""
    return html


# ──────────────────────────────────────────────
# PromptEvaluator
# ──────────────────────────────────────────────

class PromptEvaluator:
    """Genera datasets de evaluación, ejecuta prompts y puntúa los resultados.

    Técnicas de prompt engineering usadas internamente:
    - Prefill (```json) + stop_sequence (```) para extraer JSON fiable del modelo.
    - System prompts especializados por tarea (diseñador de escenarios, evaluador estricto).
    - Temperature alta (1.0) para ideas diversas; baja (0.0) para evaluación determinista.
    - Few-shot examples en el prompt de generación de test cases.
    - Concurrent futures para paralelizar llamadas al modelo.
    """

    PREFILL = "```json"
    STOP = ["```"]

    def __init__(self, max_concurrent_tasks: int = 3):
        self.max_concurrent_tasks = max_concurrent_tasks

    # ── Utilidades ──────────────────────────────

    def _render(self, template: str, variables: dict) -> str:
        """Reemplaza {placeholder} por el valor correspondiente en variables.
        Los {{ y }} literales se convierten en { y } después del reemplazo."""
        for key, value in variables.items():
            template = template.replace("{" + key + "}", str(value))
        return template.replace("{{", "{").replace("}}", "}")

    def _chat_json(
        self,
        prompt: str,
        system: str,
        temperature: float = 1.0,
    ) -> str:
        """Llama al modelo con prefill JSON y stop sequence, devuelve el texto JSON."""
        messages: list = []
        add_user_message(messages, prompt)
        add_assistant_message(messages, self.PREFILL)
        return chat(messages, system=system, temperature=temperature, stop_sequences=self.STOP)

    # ── Generación de dataset ────────────────────

    def generate_unique_ideas(
        self,
        task_description: str,
        prompt_inputs_spec: dict,
        num_cases: int,
    ) -> list[str]:
        """Genera N ideas distintas de escenarios de test para la tarea dada.

        Usa temperature alta para maximizar diversidad.
        El prefill + stop fuerza salida como array JSON de strings.
        """
        inputs_str = ", ".join(
            f'"{k}": str  # {v.replace(chr(10), " ")}' for k, v in prompt_inputs_spec.items()
        )
        prompt = self._render(
            dedent("""
            Generate {num_cases} unique, diverse ideas for testing a prompt that accomplishes this task:

            <task_description>
            {task_description}
            </task_description>

            The prompt will receive the following inputs:
            <prompt_inputs>
            {inputs_str}
            </prompt_inputs>

            Each idea should represent a distinct scenario that tests different aspects of the task.

            Output Format:
            Respond with a JSON array of brief descriptions.

            Example:
            ```json
            [
                "Testing with technical computer science terminology",
                "Testing with medical research findings",
                "Testing with complex mathematical concepts"
            ]
            ```

            Generate exactly {num_cases} unique ideas. Each must be:
            - Clearly distinct from the others
            - Specific enough to guide generation of a full test case
            - Solvable with no more than 400 tokens of output
            """),
            {"num_cases": num_cases, "task_description": task_description, "inputs_str": inputs_str},
        )
        text = self._chat_json(
            prompt,
            system="You are a test scenario designer specialized in creating diverse, unique testing scenarios.",
            temperature=1.0,
        )
        return json.loads(text)

    def generate_test_case(
        self,
        task_description: str,
        idea: str,
        prompt_inputs_spec: dict,
    ) -> dict:
        """Genera un test case completo (prompt_inputs + solution_criteria) para una idea dada.

        Usa temperature moderada para creatividad controlada.
        El prefill + stop fuerza salida como objeto JSON.
        """
        example_inputs = "".join(
            f'"{k}": "EXAMPLE_VALUE",\n'
            for k in prompt_inputs_spec
        )
        allowed_keys = ", ".join(f'"{k}"' for k in prompt_inputs_spec)

        prompt = self._render(
            dedent("""
            Generate a single detailed test case for a prompt evaluation based on:

            <task_description>
            {task_description}
            </task_description>

            <specific_idea>
            {idea}
            </specific_idea>

            <allowed_input_keys>
            {allowed_keys}
            </allowed_input_keys>

            Output Format:
            ```json
            {{
                "prompt_inputs": {{
                    {example_inputs}
                }},
                "solution_criteria": ["criterion 1", "criterion 2"]
            }}
            ```

            IMPORTANT REQUIREMENTS:
            - ONLY use these exact input keys: {allowed_keys}
            - Include ALL keys listed in allowed_input_keys
            - solution_criteria: 1 to 4 concise, measurable items tied to the core task only
            - Solvable with no more than 400 tokens of output
            - DO NOT add fields beyond those specified in the output format

            Example (extract topic task):
            ```json
            {{
                "prompt_inputs": {{
                    "content": "The transition to renewable energy encompasses solar, wind and policy dimensions..."
                }},
                "solution_criteria": [
                    "Includes all main topics mentioned in the text"
                ]
            }}
            ```
            """),
            {
                "task_description": task_description,
                "idea": idea,
                "allowed_keys": allowed_keys,
                "example_inputs": example_inputs,
            },
        )
        text = self._chat_json(
            prompt,
            system="You are a test case creator specializing in designing evaluation scenarios.",
            temperature=0.7,
        )
        test_case = json.loads(text)
        test_case["task_description"] = task_description
        test_case["scenario"] = idea
        return test_case

    def generate_dataset(
        self,
        task_description: str,
        prompt_inputs_spec: dict | None = None,
        num_cases: int = 5,
        output_file: str = "dataset.json",
    ) -> list[dict]:
        """Genera un dataset completo de test cases y lo guarda en JSON.

        1. Llama a generate_unique_ideas para obtener N escenarios distintos.
        2. Para cada idea, genera un test case completo en paralelo.
        3. Persiste el dataset en output_file.
        """
        if prompt_inputs_spec is None:
            prompt_inputs_spec = {}

        ideas = self.generate_unique_ideas(task_description, prompt_inputs_spec, num_cases)

        dataset: list[dict] = []
        completed = 0
        total = len(ideas)
        last_milestone = 0

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_concurrent_tasks) as ex:
            futures = {
                ex.submit(self.generate_test_case, task_description, idea, prompt_inputs_spec): idea
                for idea in ideas
            }
            for future in concurrent.futures.as_completed(futures):
                try:
                    dataset.append(future.result())
                except Exception as e:
                    print(f"Error generating test case: {e}")
                finally:
                    completed += 1
                    milestone = (int(completed / total * 100) // 20) * 20
                    if milestone > last_milestone:
                        print(f"Generated {completed}/{total} test cases")
                        last_milestone = milestone

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)

        return dataset

    # ── Evaluación ───────────────────────────────

    def grade_output(
        self,
        test_case: dict,
        output: str,
        extra_criteria: str | None = None,
    ) -> dict:
        """Puntúa la salida del prompt contra los criterios del test case.

        Usa temperature 0.0 para evaluación determinista y reproducible.
        El prefill + stop fuerza salida como objeto JSON con score numérico.

        La escala 1-10:
          1-3  → falla criterios obligatorios
          4-6  → cumple obligatorios con deficiencias en secundarios
          7-8  → cumple todo con issues menores
          9-10 → cumple todos los criterios
        """
        inputs_str = "".join(
            f'"{k}": "{v.replace(chr(10), " ")}",\n'
            for k, v in test_case["prompt_inputs"].items()
        )

        extra_block = ""
        if extra_criteria:
            extra_block = self._render(
                dedent("""
                Mandatory Requirements — ANY VIOLATION MEANS AUTOMATIC FAILURE (score ≤ 3):
                <extra_important_criteria>
                {extra_criteria}
                </extra_important_criteria>
                """),
                {"extra_criteria": extra_criteria},
            )

        prompt = self._render(
            dedent("""
            Evaluate the following AI-generated solution with EXTREME RIGOR.

            Original task:
            <task_description>
            {task_description}
            </task_description>

            Task inputs:
            <task_inputs>
            {{ {inputs_str} }}
            </task_inputs>

            Solution to evaluate:
            <solution>
            {output}
            </solution>

            Evaluation criteria:
            <criteria>
            {criteria}
            </criteria>

            {extra_block}

            Scoring:
            * 1-3: fails one or more mandatory requirements
            * 4-6: meets mandatory, significant deficiencies in secondary criteria
            * 7-8: meets mandatory and most secondary, minor issues
            * 9-10: meets all mandatory and secondary criteria

            SCORING RULES:
            - Grade ONLY on the listed criteria. Do NOT add your own requirements.
            - If all criteria are met, give a 9 or 10.
            - ANY mandatory violation MUST result in ≤ 3.
            - Use the full 1-10 scale.

            Respond with a JSON object:
            ```json
            {{
                "strengths": ["..."],
                "weaknesses": ["..."],
                "reasoning": "concise explanation",
                "score": <number 1-10>
            }}
            ```
            """),
            {
                "task_description": test_case["task_description"],
                "inputs_str": inputs_str,
                "output": output,
                "criteria": "\n".join(test_case["solution_criteria"]),
                "extra_block": extra_block,
            },
        )
        text = self._chat_json(prompt, system="", temperature=0.0)
        return json.loads(text)

    def run_test_case(
        self,
        test_case: dict,
        run_prompt_function,
        extra_criteria: str | None = None,
    ) -> dict:
        """Ejecuta el prompt contra un test case y lo puntúa."""
        output = run_prompt_function(test_case["prompt_inputs"])
        grade = self.grade_output(test_case, output, extra_criteria)
        return {
            "output": output,
            "test_case": test_case,
            "score": grade["score"],
            "reasoning": grade["reasoning"],
        }

    def run_evaluation(
        self,
        run_prompt_function,
        dataset_file: str,
        extra_criteria: str | None = None,
        json_output_file: str = "output.json",
        html_output_file: str = "output.html",
    ) -> list[dict]:
        """Ejecuta la evaluación completa sobre todos los test cases del dataset.

        1. Carga el dataset desde dataset_file.
        2. Ejecuta run_prompt_function + grade_output en paralelo.
        3. Escribe resultados en JSON y reporte visual en HTML.
        """
        with open(dataset_file, "r", encoding="utf-8") as f:
            dataset = json.load(f)

        results: list[dict] = []
        completed = 0
        total = len(dataset)
        last_milestone = 0

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_concurrent_tasks) as ex:
            futures = {
                ex.submit(self.run_test_case, tc, run_prompt_function, extra_criteria): tc
                for tc in dataset
            }
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
                completed += 1
                milestone = (int(completed / total * 100) // 20) * 20
                if milestone > last_milestone:
                    print(f"Graded {completed}/{total} test cases")
                    last_milestone = milestone

        avg = mean(r["score"] for r in results)
        print(f"Average score: {avg:.2f}")

        with open(json_output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        with open(html_output_file, "w", encoding="utf-8") as f:
            f.write(generate_prompt_evaluation_report(results))

        print(f"Report saved to {html_output_file}")
        return results


# ──────────────────────────────────────────────
# Demo: plan de comidas para atletas
# (equivalente al ejemplo del notebook)
# ──────────────────────────────────────────────

TASK_DESCRIPTION = "Write a compact, concise 1 day meal plan for a single athlete"

PROMPT_INPUTS_SPEC = {
    "height": "Athlete's height in cm",
    "weight": "Athlete's weight in kg",
    "goal": "Goal of the athlete",
    "restrictions": "Dietary restrictions of the athlete",
}

EXTRA_CRITERIA = """
The output should include:
- Daily caloric total
- Macronutrient breakdown (protein, fat, carbs)
- Meals with exact foods, portions (in grams), and timing
"""


def run_prompt(prompt_inputs: dict) -> str:
    """Prompt bajo evaluación — aquí es donde se aplica prompt engineering iterativo.

    Técnicas aplicadas:
    - Instrucción clara y directa con contexto del atleta en XML tags
    - Guía de formato con reglas numeradas
    - Few-shot example con input/output de referencia para anclar el estilo
    """
    prompt = f"""
Generate a one-day meal plan for an athlete that meets their dietary restrictions.

<athlete_information>
- Height: {prompt_inputs["height"]}
- Weight: {prompt_inputs["weight"]}
- Goal: {prompt_inputs["goal"]}
- Dietary restrictions: {prompt_inputs["restrictions"]}
</athlete_information>

Guidelines:
1. Include accurate daily calorie amount
2. Show protein, fat, and carb amounts
3. Specify when to eat each meal
4. Use only foods that fit restrictions
5. List all portion sizes in grams
6. Keep budget-friendly if mentioned

Example:
<sample_input>
height: 170 cm | weight: 70 kg | goal: Maintain fitness | restrictions: High cholesterol
</sample_input>
<ideal_output>
**Calorie Target:** ~2500 kcal | **Macros:** Protein 140g, Fat 70g, Carbs 340g

- **07:00 Breakfast** — Oatmeal (80g) + berries (100g) + walnuts (15g) + skim milk (240g) — P:15g F:15g C:60g
- **10:00 Snack** — Apple (150g) + almond butter (30g) — P:7g F:18g C:25g
- **13:00 Lunch** — Grilled chicken (120g) + salad (150g) + vinaigrette (30g) + whole wheat bread (60g) — P:40g F:15g C:70g
- **16:00 Snack** — Greek yogurt (170g) + banana (120g) — P:20g F:0g C:40g
- **19:00 Dinner** — Baked salmon (140g) + broccoli (200g) + quinoa (75g dry) — P:40g F:20g C:80g
- **21:00 Snack** — Almonds (20g) — P:8g F:12g C:15g
</ideal_output>
"""
    messages: list = []
    add_user_message(messages, prompt)
    return chat(messages)


def main() -> None:
    print("Prompt Eval Engineer — Demo: meal plan for athletes")
    print("====================================================")
    print(f"Model: {MODEL}")
    print()

    evaluator = PromptEvaluator(max_concurrent_tasks=1)

    print("Step 1: Generating dataset...")
    evaluator.generate_dataset(
        task_description=TASK_DESCRIPTION,
        prompt_inputs_spec=PROMPT_INPUTS_SPEC,
        num_cases=3,
        output_file="dataset.json",
    )
    print("Dataset saved to dataset.json\n")

    print("Step 2: Running evaluation...")
    evaluator.run_evaluation(
        run_prompt_function=run_prompt,
        dataset_file="dataset.json",
        extra_criteria=EXTRA_CRITERIA,
        json_output_file="output.json",
        html_output_file="output.html",
    )


if __name__ == "__main__":
    main()
