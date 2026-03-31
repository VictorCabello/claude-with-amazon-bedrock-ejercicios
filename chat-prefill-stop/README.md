# Chat Prefill + Stop Sequences

Ejercicio que demuestra dos técnicas de control de salida del modelo:
**prefilled assistant messages** y **stop sequences**.

## Descripción

### Prefilled assistant messages

El prefill consiste en inyectar texto parcial al inicio de la respuesta del asistente
**antes** de que el modelo genere nada. El modelo recibe ese texto como el comienzo
de su propia respuesta y continúa desde ahí, lo que garantiza que la salida empiece
exactamente con el formato deseado.

Caso de uso típico: forzar salida JSON, añadir etiquetas, evitar que el modelo
incluya disclaimers o rodeos al inicio de la respuesta.

### Stop sequences

Las stop sequences son cadenas de texto en las que el modelo **detiene la generación**
en cuanto las encuentra. El token de parada no se incluye en la respuesta final.

Caso de uso típico: limitar la respuesta a una sola línea, detener la generación
antes de un separador, extraer solo la primera parte de una respuesta estructurada.

---

### Demo de este ejercicio

La clase `ChatBot` actúa como clasificador de sentimientos que:

1. Fuerza la respuesta a empezar con `"Sentimiento: "` (prefill).
2. Detiene la generación al encontrar `"\n"` o `"."` (stop sequences).

El resultado es una respuesta de una sola palabra, formateada y sin variaciones:

```
> Hoy me ha tocado la lotería
*.- Sentimiento: positivo
```

---

## Implementación actual (Ollama)

> **Nota:** Este ejercicio no cuenta con acceso a Amazon Bedrock en este momento,
> por lo que `_process` fue implementado usando [Ollama](https://ollama.com/) como
> sustituto local del servicio.

```python
# src/claude_with_amazon_bedrock_ejercicio_chat_prefill_stop/main.py

import ollama

def _process(self) -> str:
    # Prefill: se añade como mensaje de asistente incompleto al final de messages.
    # El modelo continúa la generación a partir de ese punto.
    messages_with_prefill = self.messages + [
        {"role": "assistant", "content": self.prefill}
    ]

    response = ollama.chat(
        model='qwen2.5',
        messages=messages_with_prefill,
        options={"stop": self.stop_sequences},   # stop sequences
    )

    continuation: str = response['message']['content']
    full_answer: str = self.prefill + continuation   # reconstruimos la respuesta completa
    self._chat_as(ASSISTANT_ROLE, full_answer)
    return full_answer
```

> **Limitación con Ollama:** El comportamiento del prefill (continuar desde el mensaje
> de asistente incompleto) depende del *chat template* del modelo. Con `qwen2.5` funciona
> correctamente. Con otros modelos puede que el mensaje de asistente sea tratado como un
> turno completo previo en lugar de como inicio de la respuesta actual.

---

## Implementación equivalente con Amazon Bedrock (boto3)

En la API de Anthropic el prefill se pasa como el **último elemento de `messages`**
con `role: "assistant"` y contenido parcial. La API garantiza la continuación de forma
nativa, sin depender del template del modelo.

```python
import boto3
import json

MODEL_ID = "anthropic.claude-3-5-sonnet-20241022-v2:0"

bedrock = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1",
)

def _process(self) -> str:
    user_and_assistant_messages = [
        m for m in self.messages if m["role"] != "system"
    ]

    # Prefill: último mensaje con role "assistant" e contenido parcial
    user_and_assistant_messages.append(
        {"role": "assistant", "content": self.prefill}
    )

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024,
        "system": self.system_prompt,
        "messages": user_and_assistant_messages,
        "stop_sequences": self.stop_sequences,   # stop sequences
    }

    response = bedrock.invoke_model(
        modelId=MODEL_ID,
        body=json.dumps(body),
    )

    result = json.loads(response["body"].read())
    continuation: str = result["content"][0]["text"]

    # stop_reason puede ser "stop_sequence" o "end_turn"
    stop_reason: str = result["stop_reason"]
    print(f"[stop_reason: {stop_reason}]")

    full_answer: str = self.prefill + continuation
    self._chat_as(ASSISTANT_ROLE, full_answer)
    return full_answer
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
| Respuesta devuelta | continuación desde el prefill | continuación desde el prefill |
| Reconstrucción | `prefill + continuation` | `prefill + continuation` |

> **Importante:** Con la API de Anthropic, cuando la generación se detiene por una
> stop sequence, `stop_reason` vale `"stop_sequence"` y `stop_sequence` indica cuál
> fue la cadena que la activó. Esto permite distinguir si el modelo terminó de forma
> natural (`"end_turn"`) o fue interrumpido.

## Instalación y ejecución

```bash
poetry install
poetry run chat
```
