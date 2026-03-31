import ollama


USER_ROLE: str = 'user'
ASSISTANT_ROLE: str = 'assistant'
SYSTEM_ROLE: str = 'system'

DEFAULT_SYSTEM_PROMPT: str = (
    "Eres un asistente útil y conciso. "
    "Responde siempre en el mismo idioma que el usuario."
)


class ChatBot:

    def __init__(self, system_prompt: str = DEFAULT_SYSTEM_PROMPT):
        self.system_prompt: str = system_prompt
        self.messages: list[dict] = [
            {"role": SYSTEM_ROLE, "content": self.system_prompt}
        ]

    def send(self, new_mesg: str) -> str:
        self._chat_as(USER_ROLE, new_mesg)
        return self._process()

    def _process(self) -> str:
        response = ollama.chat(
            model='qwen2.5',
            messages=self.messages
        )
        answer: str = response['message']['content']
        self._chat_as(ASSISTANT_ROLE, answer)
        return answer

    def _chat_as(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})


def main():
    print('Sistema de chat con System Prompt')
    print('----------------------------------')
    print(f'System prompt activo:\n  "{DEFAULT_SYSTEM_PROMPT}"\n')
    print('Empieza a interactuar. Escribe "exit" para salir.\n')

    chatbot = ChatBot()
    is_working: bool = True

    while is_working:
        new_msg = input('> ')
        is_working = new_msg.strip().lower() != 'exit'
        if is_working:
            answer: str = chatbot.send(new_mesg=new_msg)
            print(f'\n*.- {answer}\n')
