# eval-prompt-workflow

Automatiza el **ciclo de evaluación de prompts**: dado un dataset dorado producido por
[eval-data-assets](../eval-data-assets/), ejecuta cada entrada a través de un modelo,
puntúa cada respuesta con un LLM-como-juez y escribe un informe estructurado.

```
eval_dataset.json  ──►  ModelRunner  ──►  EvalJudge  ──►  eval_results.json
 (de eval-data-assets)   (respuestas)      (puntuaciones)   (informe)
```

---

## Pasos del flujo

| Paso | Componente | Qué hace |
|------|------------|----------|
| 1 | `ModelRunner` | Envía cada `input` del dataset al modelo y captura el `actual_output` |
| 2 | `EvalJudge` | Compara `actual_output` vs `expected_output` mediante una segunda llamada LLM y devuelve una puntuación 0-10 más una justificación de una línea |
| 3 | `EvalWorkflow` | Orquesta ejecutar → juzgar → agregar → persistir |
| 4 | Resumen | Tasa de aprobación, puntuación media, desglose por categoría y dificultad |

### Formato de salida

```json
{
  "summary": {
    "total": 9,
    "passed": 7,
    "failed": 2,
    "pass_rate": 0.778,
    "avg_score": 7.89,
    "by_category": { "ec2": { "total": 3, "passed": 3, "pass_rate": 1.0, "avg_score": 8.3 } },
    "by_difficulty": { "easy": { "total": 3, "passed": 3, "pass_rate": 1.0, "avg_score": 9.0 } }
  },
  "results": [
    {
      "id": 1,
      "input": "¿Qué es Amazon EC2?",
      "expected_output": "Amazon EC2 es ...",
      "actual_output": "Amazon EC2 (Elastic Compute Cloud) es ...",
      "category": "ec2",
      "difficulty": "easy",
      "score": 9,
      "rationale": "La respuesta cubre correctamente los puntos clave de la respuesta esperada.",
      "passed": true
    }
  ]
}
```

---

## Configuración local (Qwen vía Ollama — alternativa)

> **Nota:** Este proyecto está diseñado para ejecutarse en **AWS Bedrock** (ver más abajo).
> La implementación local usa [Ollama](https://ollama.com/) con el modelo `qwen2.5`
> como reemplazo directo porque el acceso a Bedrock no está disponible en este entorno.
> Las técnicas de ingeniería de prompts (prefill + stop sequences) y la arquitectura
> general son idénticas.

### Requisitos previos

```bash
# Instalar Ollama
brew install ollama        # macOS

# Descargar el modelo
ollama pull qwen2.5

# Iniciar el daemon (si no está en ejecución)
ollama serve
```

### Instalar y ejecutar

```bash
cd eval-prompt-workflow
poetry install
poetry run eval
```

La CLI solicitará:
- Ruta al dataset de evaluación (por defecto `../eval-data-assets/eval_dataset.json`)
- Ruta de salida para los resultados (por defecto `eval_results.json`)

---

## Implementación en AWS Bedrock

En Bedrock, reemplaza las dos llamadas `ollama.chat(...)` con llamadas `boto3` a
`bedrock-runtime`. La estructura del prompt y la lógica del juez permanecen igual.

### Runner — responder preguntas de evaluación

```python
import boto3
import json

bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")

def run_on_bedrock(input_text: str) -> str:
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 512,
        "system": RUNNER_SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": input_text}],
    }
    response = bedrock.invoke_model(
        modelId="anthropic.claude-3-5-sonnet-20241022-v2:0",
        body=json.dumps(body),
        contentType="application/json",
        accept="application/json",
    )
    result = json.loads(response["body"].read())
    return result["content"][0]["text"].strip()
```

### Judge — LLM-como-juez con prefill

Bedrock soporta nativamente el patrón de prefill en el turno del asistente
mediante el array `messages`. Las stop sequences se pasan en el cuerpo de la petición.

```python
def judge_on_bedrock(input_text: str, expected: str, actual: str) -> dict:
    user_message = (
        f"Pregunta: {input_text}\n\n"
        f"Respuesta esperada:\n{expected}\n\n"
        f"Respuesta del modelo:\n{actual}\n\n"
        "Evalúa qué tan bien la respuesta del modelo captura la información de la respuesta esperada."
    )
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 256,
        "system": JUDGE_SYSTEM_PROMPT,
        "messages": [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": JUDGE_PREFILL},   # prefill fuerza JSON
        ],
        "stop_sequences": JUDGE_STOP_SEQUENCES,
    }
    response = bedrock.invoke_model(
        modelId="anthropic.claude-3-5-sonnet-20241022-v2:0",
        body=json.dumps(body),
        contentType="application/json",
        accept="application/json",
    )
    result = json.loads(response["body"].read())
    raw = JUDGE_PREFILL + result["content"][0]["text"]
    if not raw.rstrip().endswith("}"):
        raw = raw.rstrip() + "}"
    parsed = json.loads(raw)
    score = float(parsed["score"])
    return {"score": score, "rationale": parsed["rationale"], "passed": score >= PASS_THRESHOLD}
```

### Permisos IAM requeridos

```json
{
  "Effect": "Allow",
  "Action": ["bedrock:InvokeModel"],
  "Resource": [
    "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-3-5-sonnet-20241022-v2:0"
  ]
}
```

### Modelos Bedrock recomendados

| Caso de uso | Model ID |
|-------------|----------|
| Juez de alta precisión | `anthropic.claude-3-5-sonnet-20241022-v2:0` |
| Runner eficiente en costes | `anthropic.claude-haiku-4-5-20251001` |
| Lotes de alto volumen | Usar [Inferencia por lotes](https://docs.aws.amazon.com/bedrock/latest/userguide/batch-inference.html) |

---

## Técnicas clave de ingeniería de prompts

### Prefill + stop sequences (juez)

El juez usa la misma técnica introducida en `eval-data-assets` para
garantizar salida JSON sin post-procesamiento:

```python
# Inyectar la apertura del JSON como primer token del asistente
{"role": "assistant", "content": '{"score": '}

# Detener la generación tras el cierre del objeto
"stop_sequences": ["}\n", "} "]
```

Esto elimina texto envolvente ("¡Claro! Aquí está la evaluación: ...") y
hace que la salida sea directamente parseable con `json.loads()` tras añadir `}`.

### Umbral de aprobación

Un registro se considera **aprobado** cuando `score >= 7`. Este umbral es
configurable mediante la constante `PASS_THRESHOLD` en `main.py`.
