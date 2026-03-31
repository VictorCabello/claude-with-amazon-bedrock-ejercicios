# Chat System Prompt

Ejercicio que demuestra cómo usar un **system prompt** para definir la personalidad,
el rol y las restricciones del asistente antes de que comience la conversación.

## Descripción

El system prompt es un mensaje especial con `role: "system"` que se inserta al inicio
del historial. El modelo lo usa como instrucción de contexto permanente: define quién
es el asistente, qué puede y no puede hacer, y cómo debe responder.

La clase `ChatBot` recibe el system prompt en el constructor y lo prepopula en
`self.messages` antes de cualquier turno de usuario:

```python
self.messages: list[dict] = [
    {"role": "system", "content": self.system_prompt}
]
```

De este modo, en cada llamada al modelo el historial completo incluye siempre el
system prompt como primer elemento.

## Implementación actual (Ollama)

> **Nota:** Este ejercicio no cuenta con acceso a Amazon Bedrock en este momento,
> por lo que `_process` fue implementado usando [Ollama](https://ollama.com/) como
> sustituto local del servicio.

```python
# src/claude_with_amazon_bedrock_ejercicio_chat_system_prompt/main.py

import ollama

def _process(self) -> str:
    response = ollama.chat(
        model='qwen2.5',
        messages=self.messages   # incluye el system prompt + historial
    )
    answer: str = response['message']['content']
    self._chat_as(ASSISTANT_ROLE, answer)
    return answer
```

Ollama acepta el mensaje `role: "system"` dentro del array `messages`, igual que
lo haría la API de Anthropic a través de Bedrock.

## Implementación equivalente con Amazon Bedrock (boto3)

En la API de Anthropic/Bedrock el system prompt **no va dentro de `messages`**:
se pasa como un campo separado `"system"` en el body del request. Esto es la
diferencia más importante respecto a Ollama.

```python
import boto3
import json

MODEL_ID = "anthropic.claude-3-5-sonnet-20241022-v2:0"

bedrock = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1",   # ajusta la región según tu configuración
)

def _process(self) -> str:
    # Filtramos el system prompt fuera de messages antes de enviar
    user_and_assistant_messages = [
        m for m in self.messages if m["role"] != "system"
    ]

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024,
        "system": self.system_prompt,          # campo dedicado para el system prompt
        "messages": user_and_assistant_messages,
    }

    response = bedrock.invoke_model(
        modelId=MODEL_ID,
        body=json.dumps(body),
    )

    result = json.loads(response["body"].read())
    answer: str = result["content"][0]["text"]
    self._chat_as(ASSISTANT_ROLE, answer)
    return answer
```

### Diferencias clave respecto a la versión Ollama

| Aspecto | Ollama | Amazon Bedrock (boto3) |
|---------|--------|------------------------|
| Cliente | `ollama.chat()` | `bedrock.invoke_model()` |
| Modelo | `qwen2.5` (local) | `anthropic.claude-3-5-sonnet-...` |
| System prompt | dentro de `messages` con `role: "system"` | campo `"system"` separado en el body |
| Historial enviado | `self.messages` completo | solo mensajes `user` / `assistant` |
| Autenticación | ninguna | credenciales AWS (`~/.aws`) |
| Formato de respuesta | `response['message']['content']` | `result['content'][0]['text']` |
| Campo extra requerido | — | `anthropic_version` en el body |

> **Importante:** La API de Anthropic (y por tanto Bedrock) rechaza mensajes con
> `role: "system"` dentro del array `messages`. El system prompt **debe** pasarse
> en el campo `"system"` de primer nivel del body.

## Instalación y ejecución

```bash
poetry install
poetry run chat
```
