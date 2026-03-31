import ollama


USER_ROLE: str = 'user'
ASSISTANT_ROLE: str = 'assistant'
SYSTEM_ROLE: str = 'system'

DEFAULT_SYSTEM_PROMPT: str = (
    "Eres un asistente útil y conciso. "
    "Responde siempre en el mismo idioma que el usuario."
)


class StreamingChatBot:

    def __init__(self, system_prompt: str = DEFAULT_SYSTEM_PROMPT):
        self.system_prompt: str = system_prompt
        self.messages: list[dict] = [
            {"role": SYSTEM_ROLE, "content": self.system_prompt}
        ]

    def send_streaming(self, new_mesg: str) -> str:
        """Envía un mensaje y devuelve el texto completo tras procesar el stream."""
        self._chat_as(USER_ROLE, new_mesg)
        return self._process_streaming()

    def _process_streaming(self) -> str:
        stream = ollama.chat(
            model='qwen2.5',
            messages=self.messages,
            stream=True,        # activa el modo streaming
        )

        full_answer: str = ''
        for chunk in stream:
            token: str = chunk['message']['content']
            print(token, end='', flush=True)   # imprime en tiempo real
            full_answer += token

        print()  # salto de línea final
        self._chat_as(ASSISTANT_ROLE, full_answer)
        return full_answer

    def _chat_as(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})


def main():
    print('Chat con Streaming')
    print('------------------')
    print(f'System prompt activo:\n  "{DEFAULT_SYSTEM_PROMPT}"\n')
    print('Los tokens aparecen en pantalla conforme el modelo los genera.')
    print('Escribe "exit" para salir.\n')

    chatbot = StreamingChatBot()
    is_working: bool = True

    while is_working:
        new_msg = input('> ')
        is_working = new_msg.strip().lower() != 'exit'
        if is_working:
            print('\nAsistente: ', end='', flush=True)
            chatbot.send_streaming(new_mesg=new_msg)
            print()
