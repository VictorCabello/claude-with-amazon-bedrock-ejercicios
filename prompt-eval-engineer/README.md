# Prompt Eval Engineer — Workflow de Evaluación y Prompt Engineering Iterativo

Ejercicio que implementa el **flujo completo de evaluación de prompts con prompt engineering** inspirado en el notebook `002_prompting_completed.ipynb` (parte del curso [Claude in Amazon Bedrock](https://anthropic-partners.skilljar.com/claude-in-amazon-bedrock), no incluido en este repositorio), empaquetado como proyecto Python reutilizable.

## Qué hace

```
task_description + prompt_inputs_spec
         │
         ▼
  generate_dataset()
  ┌──────────────────────────────────────────┐
  │  generate_unique_ideas()                 │
  │    prefill: ```json  stop: ```           │
  │    → N ideas de escenarios distintos     │
  │                                          │
  │  generate_test_case() × N (paralelo)     │
  │    prefill: ```json  stop: ```           │
  │    → {prompt_inputs, solution_criteria}  │
  └──────────────────────────────────────────┘
         │ dataset.json
         ▼
  run_evaluation()
  ┌──────────────────────────────────────────┐
  │  run_prompt_function(prompt_inputs)      │
  │    → output del prompt bajo evaluación   │
  │                                          │
  │  grade_output()  temp=0.0                │
  │    prefill: ```json  stop: ```           │
  │    → {score, reasoning, strengths, ...}  │
  └──────────────────────────────────────────┘
         │
         ▼
  output.json  +  output.html (reporte visual)
```

## Técnicas de prompt engineering aplicadas

| Técnica | Dónde se usa | Efecto |
|---|---|---|
| **Prefill** (`\`\`\`json`) | generación de ideas, test cases, evaluación | Fuerza salida JSON desde el primer token |
| **Stop sequence** (`\`\`\``) | mismo | Corta tras el cierre del bloque JSON |
| **System prompt especializado** | por función del modelo (diseñador, evaluador) | Ancla el rol sin contaminar el prompt de usuario |
| **Temperature alta (1.0)** | `generate_unique_ideas` | Maximiza diversidad de escenarios |
| **Temperature baja (0.0)** | `grade_output` | Evaluación determinista y reproducible |
| **Few-shot example** | `generate_test_case`, `run_prompt` | Reduce ambigüedad en formato de salida |
| **XML tags** | separación de secciones en prompts | Delimita claramente inputs, criterios y ejemplos |
| **Concurrent futures** | generación y evaluación | Paraleliza llamadas al modelo |

---

## Implementación actual (Ollama)

> **Nota:** Este ejercicio no cuenta con acceso a Amazon Bedrock en este momento,
> por lo que el cliente de LLM fue implementado usando [Ollama](https://ollama.com/) como
> sustituto local del servicio.

```python
# src/prompt_eval_engineer/main.py — función chat con Ollama

import ollama

MODEL = "qwen2.5"

def chat(messages, system=None, temperature=1.0, stop_sequences=None):
    ollama_messages = []
    if system:
        ollama_messages.append({"role": "system", "content": system})
    for msg in messages:
        ollama_messages.append({"role": msg["role"], "content": msg["content"]})

    options = {"temperature": temperature}
    if stop_sequences:
        options["stop"] = stop_sequences

    response = ollama.chat(model=MODEL, messages=ollama_messages, options=options)
    return response["message"]["content"]
```

El prefill se implementa añadiendo un mensaje `role: "assistant"` al final de la lista
antes de llamar a `chat`. Con el modelo `qwen2.5` y Ollama esto funciona correctamente:
el modelo continúa la respuesta a partir del texto del asistente.

> **Limitación:** el soporte de prefill en Ollama depende del chat template del modelo.
> Con modelos distintos a `qwen2.5` el mensaje de asistente puede tratarse como turno
> completo en lugar de inicio de respuesta.

---

## Implementación equivalente con Amazon Bedrock (boto3)

Con la API de Anthropic el prefill está **garantizado de forma nativa**: el último mensaje
con `role: "assistant"` siempre es continuado por el modelo, independientemente del template.

```python
import boto3
import json

MODEL_ID = "us.anthropic.claude-3-5-sonnet-20241022-v2:0"

bedrock = boto3.client("bedrock-runtime", region_name="us-west-2")

def chat(messages, system=None, temperature=1.0, stop_sequences=None):
    # Convierte el formato plano al formato Bedrock (content como lista de dicts)
    bedrock_messages = [
        {"role": msg["role"], "content": [{"text": msg["content"]}]}
        for msg in messages
    ]

    params = {
        "modelId": MODEL_ID,
        "messages": bedrock_messages,
        "inferenceConfig": {
            "temperature": temperature,
        },
    }

    if system:
        params["system"] = [{"text": system}]

    if stop_sequences:
        params["inferenceConfig"]["stopSequences"] = stop_sequences

    response = bedrock.converse(**params)
    return response["output"]["message"]["content"][0]["text"]
```

### Diferencias clave Ollama vs Amazon Bedrock

| Aspecto | Ollama | Amazon Bedrock (boto3) |
|---|---|---|
| Cliente | `ollama.chat()` | `bedrock.converse()` |
| Modelo | `qwen2.5` (local) | `us.anthropic.claude-3-5-sonnet-20241022-v2:0` |
| System prompt | mensaje `role: "system"` dentro de la lista | parámetro `system` separado |
| Prefill | último mensaje `role: "assistant"` (depende del template) | último mensaje `role: "assistant"` (garantizado) |
| Stop sequences | `options={"stop": [...]}` | `inferenceConfig.stopSequences` |
| Stop reason | no expuesto directamente | `response["stopReason"]` → `"stop_sequence"` o `"end_turn"` |
| Formato `content` | `str` plano | lista de dicts `[{"text": "..."}]` |
| Coste | gratuito (local) | por token (ver pricing de Bedrock) |

### Para migrar este proyecto a Bedrock

1. Reemplazar la función `chat` por la versión boto3 de arriba.
2. Instalar `boto3`: `poetry add boto3`.
3. Configurar credenciales AWS (`aws configure` o variables de entorno).
4. Cambiar `MODEL_ID` al inference profile disponible en tu región.

No es necesario cambiar ninguna otra parte del código: `PromptEvaluator`,
`generate_prompt_evaluation_report` y `run_prompt` son agnósticos al backend.

---

## Instalación y ejecución

### Requisitos

- Python ≥ 3.14
- [Poetry](https://python-poetry.org/)
- [Ollama](https://ollama.com/) instalado y corriendo localmente
- Modelo `qwen2.5` descargado: `ollama pull qwen2.5`

### Setup

```bash
cd prompt-eval-engineer
poetry install
```

### Ejecutar el demo

```bash
poetry run eval
```

El demo ejecuta el flujo completo con un caso de uso de planes de comida para atletas:
1. Genera 3 test cases en `dataset.json`
2. Ejecuta el prompt de ejemplo contra cada test case
3. Puntúa los resultados y escribe el reporte en `output.html`

### Usar en tu propio proyecto

```python
from prompt_eval_engineer.main import PromptEvaluator, chat, add_user_message

evaluator = PromptEvaluator(max_concurrent_tasks=2)

# 1. Generar dataset
evaluator.generate_dataset(
    task_description="Summarize a news article in one sentence",
    prompt_inputs_spec={"article": "The full text of the news article"},
    num_cases=5,
    output_file="my_dataset.json",
)

# 2. Definir el prompt a evaluar
def run_prompt(prompt_inputs):
    messages = []
    add_user_message(messages, f"Summarize in one sentence:\n\n{prompt_inputs['article']}")
    return chat(messages)

# 3. Evaluar
results = evaluator.run_evaluation(
    run_prompt_function=run_prompt,
    dataset_file="my_dataset.json",
    extra_criteria="The summary must be a single sentence under 30 words.",
    html_output_file="my_report.html",
)
```

---

## Estructura del proyecto

```
prompt-eval-engineer/
├── pyproject.toml
├── README.md
├── .gitignore
├── src/
│   └── prompt_eval_engineer/
│       ├── __init__.py
│       └── main.py          ← PromptEvaluator + chat + reporte HTML
└── tests/
    └── __init__.py
```
