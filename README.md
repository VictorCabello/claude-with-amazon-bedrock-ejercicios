# Claude with Amazon Bedrock — Ejercicios

Este repositorio contiene ejercicios prácticos del curso
[Claude in Amazon Bedrock](https://anthropic-partners.skilljar.com/claude-in-amazon-bedrock).

## Estructura

Cada subdirectorio corresponde a un ejercicio independiente con su propio entorno
de dependencias gestionado con [Poetry](https://python-poetry.org/).

| Ejercicio | Descripción |
|-----------|-------------|
| [chat-multi-input](./chat-multi-input/) | Conversación multi-turn con manejo de historial |
| [chat-system-prompt](./chat-system-prompt/) | Uso de system prompt para definir rol y restricciones del asistente |
| [chat-streaming](./chat-streaming/) | Streaming de tokens en tiempo real en lugar de esperar la respuesta completa |
| [chat-prefill-stop](./chat-prefill-stop/) | Prefilled assistant messages y stop sequences para controlar la salida |

## Requisitos generales

- Python >= 3.14
- Poetry >= 2.0

## Cómo ejecutar un ejercicio

```bash
cd <nombre-ejercicio>
poetry install
poetry run <script>
```
