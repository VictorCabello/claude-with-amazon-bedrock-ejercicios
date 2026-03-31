# Chat Streaming

Ejercicio que demuestra cómo usar **streaming** para mostrar los tokens del
modelo en pantalla conforme se generan, en lugar de esperar a que la respuesta
completa esté lista.

## Descripción

Sin streaming, el flujo es bloqueante: el cliente espera hasta que el modelo
termina de generar la respuesta entera, y solo entonces la recibe de golpe.

Con streaming, el modelo envía cada token (o pequeño fragmento) en cuanto lo
produce. El cliente itera sobre el stream e imprime cada pieza de inmediato:

```
usuario:    "Explícame qué es el streaming"
asistente:  El  str  eam  ing  es  ...   (aparece token a token)
```

La clase `StreamingChatBot` activa el modo streaming pasando `stream=True` a
`ollama.chat()` y recorre el iterador para acumular y mostrar cada fragmento:

```python
stream = ollama.chat(model='qwen2.5', messages=self.messages, stream=True)

full_answer = ''
for chunk in stream:
    token = chunk['message']['content']
    print(token, end='', flush=True)   # imprime en tiempo real
    full_answer += token
```

## Implementación actual (Ollama)

> **Nota:** Este ejercicio no cuenta con acceso a Amazon Bedrock en este momento,
> por lo que `_process_streaming` fue implementado usando
> [Ollama](https://ollama.com/) como sustituto local del servicio.

```python
# src/claude_with_amazon_bedrock_ejercicio_chat_streaming/main.py

import ollama

def _process_streaming(self) -> str:
    stream = ollama.chat(
        model='qwen2.5',
        messages=self.messages,
        stream=True,
    )

    full_answer: str = ''
    for chunk in stream:
        token: str = chunk['message']['content']
        print(token, end='', flush=True)
        full_answer += token

    print()
    self._chat_as(ASSISTANT_ROLE, full_answer)
    return full_answer
```

Ollama expone una API compatible con OpenAI y devuelve un iterador de objetos
`ChatResponse` cuando `stream=True`.

## Implementación equivalente con Amazon Bedrock (boto3)

En Bedrock el endpoint de streaming es `invoke_model_with_response_stream`.
El body del request es idéntico al de `invoke_model`, pero la respuesta llega
como un `EventStream` con eventos `chunk` que contienen fragmentos JSON.

```python
import boto3
import json

MODEL_ID = "anthropic.claude-3-5-sonnet-20241022-v2:0"

bedrock = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1",
)

def _process_streaming(self) -> str:
    user_and_assistant_messages = [
        m for m in self.messages if m["role"] != "system"
    ]

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024,
        "system": self.system_prompt,
        "messages": user_and_assistant_messages,
    }

    response = bedrock.invoke_model_with_response_stream(
        modelId=MODEL_ID,
        body=json.dumps(body),
    )

    full_answer: str = ''
    for event in response["body"]:
        chunk_data = json.loads(event["chunk"]["bytes"])

        # Solo los eventos de tipo "content_block_delta" llevan texto
        if chunk_data.get("type") == "content_block_delta":
            token: str = chunk_data["delta"].get("text", "")
            print(token, end="", flush=True)
            full_answer += token

    print()
    self._chat_as(ASSISTANT_ROLE, full_answer)
    return full_answer
```

### Eventos del stream en Bedrock

La API de Anthropic vía Bedrock emite varios tipos de eventos durante el stream.
Los más relevantes son:

| Tipo de evento | Descripción |
|----------------|-------------|
| `message_start` | Inicio del mensaje; incluye metadatos del modelo |
| `content_block_start` | Inicio de un bloque de contenido |
| `content_block_delta` | **Fragmento de texto generado** (el que interesa imprimir) |
| `content_block_stop` | Fin de un bloque de contenido |
| `message_delta` | Metadatos del final (tokens usados, stop reason) |
| `message_stop` | Señal de fin de stream |

Solo hay que reaccionar a `content_block_delta` para reconstruir el texto.

### Diferencias clave respecto a la versión Ollama

| Aspecto | Ollama | Amazon Bedrock (boto3) |
|---------|--------|------------------------|
| Cliente | `ollama.chat(..., stream=True)` | `bedrock.invoke_model_with_response_stream()` |
| Iteración | `for chunk in stream` | `for event in response["body"]` |
| Acceso al token | `chunk['message']['content']` | `json.loads(event["chunk"]["bytes"])["delta"]["text"]` |
| Filtrado de eventos | no necesario | solo procesar `type == "content_block_delta"` |
| System prompt | dentro de `messages` con `role: "system"` | campo `"system"` separado en el body |
| Autenticación | ninguna | credenciales AWS (`~/.aws`) |

## Instalación y ejecución

```bash
poetry install
poetry run chat
```
