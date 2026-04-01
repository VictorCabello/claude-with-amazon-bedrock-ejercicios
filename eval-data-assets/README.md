# Eval Data Assets — Generador de Golden Datasets para Evaluation Prompt Workflows

Ejercicio que demuestra cómo generar **data assets estructurados** (golden datasets)
para usar en workflows de evaluación de prompts, combinando **prefill** y **stop sequences**
para forzar salida JSON fiable.

## Descripción

### El problema

Los evaluation prompt workflows (PromptFlow, LangSmith, RAGAS, etc.) necesitan un
**golden dataset**: una colección de pares `input / expected_output` con metadatos
(categoría, dificultad) que sirven de referencia para medir la calidad del modelo bajo prueba.

Crear estos datasets manualmente es lento. Este ejercicio usa el propio modelo para
**generar los registros de evaluación** de forma automatizada.

### Técnicas aplicadas

**Prefill:** el generador inyecta `{"input": "` como inicio forzado de la respuesta del
asistente. El modelo continúa desde ahí, garantizando que la salida sea un objeto JSON
que empieza con la clave `input`. Sin prefill, el modelo podría añadir texto introductorio
o usar markdown que rompería el parse.

**Stop sequences:** la generación se detiene en cuanto el modelo cierra el objeto JSON
(`}\n` o `} `). Esto evita que el modelo añada explicaciones, comentarios o múltiples
registros de golpe. El `}` consumido por la stop sequence se reincorpora manualmente
antes del `json.loads`.

### Estructura del registro generado

```json
{
  "id": 1,
  "input": "¿Cuál es el procedimiento para escalar una queja de nivel 2?",
  "expected_output": "El agente debe transferir la llamada al supervisor de turno e informar al cliente del tiempo de respuesta estimado.",
  "category": "procedimientos",
  "difficulty": "medium"
}
```

### Flujo completo

```
dominio + categoría + dificultad
        │
        ▼
  EvalDataGenerator.generate()
        │  prefill: '{"input": "'
        │  stop: ['}\n', '} ']
        ▼
  ollama.chat(qwen2.5)
        │
        ▼
  prefill + continuation + '}'  →  json.loads()
        │
        ▼
  dict con id, input, expected_output, category, difficulty
        │
        ▼
  eval_dataset.json  ←── listo para el evaluation framework
```

---

## Implementación actual (Ollama)

> **Nota:** Este ejercicio no cuenta con acceso a Amazon Bedrock en este momento,
> por lo que `generate` fue implementado usando [Ollama](https://ollama.com/) como
> sustituto local del servicio.

```python
# src/claude_with_amazon_bedrock_ejercicio_eval_data_assets/main.py

import ollama

def generate(self, domain: str, category: str, difficulty: str) -> dict:
    messages_with_prefill = [
        {"role": "system",    "content": self.system_prompt},
        {"role": "user",      "content": f"Genera un registro para dominio '{domain}', categoría '{category}', dificultad '{difficulty}'."},
        {"role": "assistant", "content": self.prefill},   # prefill: '{"input": "'
    ]

    response = ollama.chat(
        model='qwen2.5',
        messages=messages_with_prefill,
        options={"stop": self.stop_sequences},   # stop: ['}\n', '} ']
    )

    continuation = response['message']['content']
    raw_json = self.prefill + continuation + "}"   # reincorporamos el '}' consumido
    return json.loads(raw_json)
```

> **Limitación con Ollama:** el soporte de prefill depende del *chat template* del modelo.
> Con `qwen2.5` funciona correctamente. Con otros modelos el mensaje de asistente puede
> ser tratado como turno completo previo en lugar de como inicio de la respuesta actual.

---

## Implementación equivalente con Amazon Bedrock (boto3)

Con la API de Anthropic el prefill está garantizado de forma nativa: el último mensaje
con `role: "assistant"` y contenido parcial es siempre continuado por el modelo,
independientemente del template.

```python
import boto3
import json

MODEL_ID = "anthropic.claude-3-5-sonnet-20241022-v2:0"

bedrock = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1",
)

def generate(self, domain: str, category: str, difficulty: str) -> dict:
    user_message = (
        f"Genera un registro de evaluación para el dominio: '{domain}'. "
        f"Categoría: '{category}'. Dificultad: '{difficulty}'."
    )

    # La API de Bedrock/Anthropic separa el system prompt del resto de mensajes.
    # El prefill va como último elemento de messages con role "assistant".
    messages = [
        {"role": "user",      "content": user_message},
        {"role": "assistant", "content": self.prefill},   # prefill garantizado
    ]

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 512,
        "system": self.system_prompt,
        "messages": messages,
        "stop_sequences": self.stop_sequences,   # ['}\n', '} ']
    }

    response = bedrock.invoke_model(
        modelId=MODEL_ID,
        body=json.dumps(body),
    )

    result = json.loads(response["body"].read())
    continuation: str = result["content"][0]["text"]

    # stop_reason puede ser "stop_sequence" (detenido por '}') o "end_turn"
    stop_reason: str = result["stop_reason"]
    stopped_by: str = result.get("stop_sequence", "—")
    print(f"  [stop_reason={stop_reason}, stopped_by={stopped_by!r}]")

    raw_json = self.prefill + continuation + "}"
    return json.loads(raw_json)
```

### Generar un dataset completo con Bedrock

```python
generator = EvalDataGenerator()

dataset = generator.generate_dataset(
    domain="atención al cliente",
    categories=["comprensión", "razonamiento", "extracción"],
    difficulties=["easy", "medium", "hard"],
    samples_per_combination=2,
)

with open("eval_dataset.json", "w", encoding="utf-8") as f:
    json.dump(dataset, f, ensure_ascii=False, indent=2)
```

Salida esperada `eval_dataset.json`:

```json
[
  {
    "id": 1,
    "input": "El cliente dice que no recibió su pedido pero el sistema muestra entregado. ¿Qué haces?",
    "expected_output": "Verificar la dirección de entrega, abrir una disputa con la transportista y ofrecer reenvío o reembolso al cliente.",
    "category": "razonamiento",
    "difficulty": "medium"
  },
  {
    "id": 2,
    "input": "¿Cuántos días tiene el cliente para solicitar una devolución?",
    "expected_output": "30 días naturales desde la fecha de entrega.",
    "category": "comprensión",
    "difficulty": "easy"
  }
]
```

### Diferencias clave respecto a la versión Ollama

| Aspecto | Ollama | Amazon Bedrock (boto3) |
|---------|--------|------------------------|
| Cliente | `ollama.chat()` | `bedrock.invoke_model()` |
| Modelo | `qwen2.5` (local) | `anthropic.claude-3-5-sonnet-...` |
| System prompt | dentro de `messages` con `role: "system"` | campo `"system"` separado en el body |
| Prefill | último mensaje `role: "assistant"` (depende del template) | último mensaje `role: "assistant"` (garantizado por la API) |
| Stop sequences | `options={"stop": [...]}` | `"stop_sequences": [...]` en el body |
| Stop reason | no expuesto directamente | `result["stop_reason"]` → `"stop_sequence"` o `"end_turn"` |
| Fiabilidad del JSON | depende del modelo local | alta — el prefill está garantizado |

> **Nota sobre `stop_reason`:** cuando la stop sequence `}\n` activa la parada, Bedrock
> devuelve `stop_reason: "stop_sequence"` y `stop_sequence: "}\n"`. Esto permite
> distinguir una parada controlada de un `end_turn` natural y detectar si el modelo
> no llegó a cerrar el objeto JSON.

---

## Instalación y ejecución

```bash
poetry install
poetry run generate
```

El script solicita dominio, categorías y número de muestras, genera el dataset y lo
escribe en `eval_dataset.json` en el directorio de trabajo.
