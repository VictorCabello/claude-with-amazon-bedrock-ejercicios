import ollama


USER_ROLE: str = 'user'
ASSISTANT_ROLE: str = 'assistant'
SYSTEM_ROLE: str = 'system'

DEFAULT_SYSTEM_PROMPT: str = (
    "Eres un clasificador de sentimientos. "
    "Analiza el texto del usuario y clasifica el sentimiento en: positivo, negativo o neutro. "
    "Responde únicamente con la etiqueta del sentimiento, sin explicaciones adicionales."
)

# Prefill: texto que se inyecta al inicio de la respuesta del asistente.
# El modelo continúa a partir de este punto, lo que fuerza un formato de salida concreto.
DEFAULT_PREFILL: str = "Sentimiento: "

# Stop sequences: el modelo detiene la generación en cuanto encuentra alguna de estas cadenas.
DEFAULT_STOP_SEQUENCES: list[str] = ["\n", "."]


class ChatBot:

    def __init__(
        self,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        prefill: str = DEFAULT_PREFILL,
        stop_sequences: list[str] | None = None,
    ):
        self.system_prompt: str = system_prompt
        self.prefill: str = prefill
        self.stop_sequences: list[str] = stop_sequences if stop_sequences is not None else DEFAULT_STOP_SEQUENCES
        self.messages: list[dict] = [
            {"role": SYSTEM_ROLE, "content": self.system_prompt}
        ]

    def send(self, new_msg: str) -> str:
        self._chat_as(USER_ROLE, new_msg)
        return self._process()

    def _process(self) -> str:
        # El prefill se añade como mensaje de asistente incompleto al final de la lista.
        # El modelo recibe esta señal y continúa la generación desde ese punto,
        # garantizando que la respuesta empiece exactamente con el texto del prefill.
        messages_with_prefill = self.messages + [
            {"role": ASSISTANT_ROLE, "content": self.prefill}
        ]

        response = ollama.chat(
            model='qwen2.5',
            messages=messages_with_prefill,
            options={"stop": self.stop_sequences},
        )

        # La API devuelve solo la continuación; concatenamos el prefill para tener
        # la respuesta completa tal y como la vería el usuario.
        continuation: str = response['message']['content']
        full_answer: str = self.prefill + continuation

        self._chat_as(ASSISTANT_ROLE, full_answer)
        return full_answer

    def _chat_as(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})


def main():
    print('Clasificador de sentimientos — Prefill + Stop Sequences')
    print('--------------------------------------------------------')
    print(f'Prefill activo   : "{DEFAULT_PREFILL}"')
    print(f'Stop sequences   : {DEFAULT_STOP_SEQUENCES}')
    print(f'System prompt    : "{DEFAULT_SYSTEM_PROMPT}"\n')
    print('Escribe una frase para clasificar su sentimiento.')
    print('Escribe "exit" para salir.\n')

    chatbot = ChatBot()
    is_working: bool = True

    while is_working:
        new_msg = input('> ')
        is_working = new_msg.strip().lower() != 'exit'
        if is_working:
            answer: str = chatbot.send(new_msg=new_msg)
            print(f'\n*.- {answer}\n')
