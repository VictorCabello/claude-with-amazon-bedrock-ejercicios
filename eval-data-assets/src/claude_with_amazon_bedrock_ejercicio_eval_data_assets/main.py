import json
import ollama


USER_ROLE: str = 'user'
ASSISTANT_ROLE: str = 'assistant'
SYSTEM_ROLE: str = 'system'

# El sistema instruye al modelo a actuar exclusivamente como generador de registros JSON
# para datasets de evaluación de prompts.
DEFAULT_SYSTEM_PROMPT: str = (
    "Eres un generador de datos de evaluación para workflows de evaluación de prompts (prompt evaluation). "
    "Tu única tarea es producir registros JSON con la siguiente estructura exacta:\n"
    '{"input": "<pregunta o instrucción de prueba>", "expected_output": "<respuesta de referencia>", '
    '"category": "<categoría>", "difficulty": "<easy|medium|hard>"}\n'
    "Genera un único registro JSON por respuesta. "
    "No incluyas texto adicional, explicaciones ni markdown. "
    "Solo JSON puro."
)

# Prefill: fuerza que la respuesta empiece con '{' para garantizar salida JSON válida.
DEFAULT_PREFILL: str = '{"input": "'

# Stop sequence: detiene la generación tras el cierre del objeto JSON.
# El modelo debe cerrar el objeto con '}' y detenerse ahí.
DEFAULT_STOP_SEQUENCES: list[str] = ['}\n', '} ']


class EvalDataGenerator:
    """Genera registros JSON de evaluación para un dominio dado.

    Cada llamada a `generate` produce un EvalRecord: un par input/expected_output
    con metadatos de categoría y dificultad, listo para usarse como golden dataset
    en un pipeline de evaluación de prompts.
    """

    def __init__(
        self,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        prefill: str = DEFAULT_PREFILL,
        stop_sequences: list[str] | None = None,
    ):
        self.system_prompt: str = system_prompt
        self.prefill: str = prefill
        self.stop_sequences: list[str] = (
            stop_sequences if stop_sequences is not None else DEFAULT_STOP_SEQUENCES
        )

    def generate(self, domain: str, category: str, difficulty: str) -> dict:
        """Genera un único registro de evaluación para el dominio, categoría y dificultad dados.

        El prefill fuerza al modelo a abrir un objeto JSON directamente.
        La stop sequence corta la generación en cuanto el objeto queda cerrado.
        El resultado se reconstruye uniendo el prefill con la continuación del modelo
        y se parsea como JSON para devolver un dict estructurado.
        """
        user_message: str = (
            f"Genera un registro de evaluación para el dominio: '{domain}'. "
            f"Categoría: '{category}'. Dificultad: '{difficulty}'."
        )

        messages_with_prefill: list[dict] = [
            {"role": SYSTEM_ROLE, "content": self.system_prompt},
            {"role": USER_ROLE, "content": user_message},
            {"role": ASSISTANT_ROLE, "content": self.prefill},
        ]

        response = ollama.chat(
            model='qwen2.5',
            messages=messages_with_prefill,
            options={"stop": self.stop_sequences},
        )

        continuation: str = response['message']['content']

        raw_json: str = self.prefill + continuation

        # La API devuelve solo la continuación a partir del prefill.
        # Reconstruimos el JSON completo concatenando prefill + continuación + '}'.
        # El '}' lo añadimos nosotros porque la stop sequence lo consume sin incluirlo.
        if raw_json[-1] != '}':
            raw_json += "}"

        try:
            record: dict = json.loads(raw_json)
        except json.JSONDecodeError:
            # Si el modelo produce JSON malformado, devolvemos el texto crudo para inspección.
            record = {"raw": raw_json, "parse_error": True}

        return record

    def generate_dataset(
        self,
        domain: str,
        categories: list[str],
        difficulties: list[str],
        samples_per_combination: int = 1,
    ) -> list[dict]:
        """Genera un dataset completo combinando todas las categorías y dificultades.

        Cada combinación (category × difficulty) produce `samples_per_combination` registros.
        El resultado es una lista de dicts lista para serializar a JSON o cargar en un
        framework de evaluación como PromptFlow, LangSmith o RAGAS.
        """
        dataset: list[dict] = []
        record_id: int = 1

        for category in categories:
            for difficulty in difficulties:
                for _ in range(samples_per_combination):
                    record = self.generate(domain, category, difficulty)
                    record["id"] = record_id
                    dataset.append(record)
                    record_id += 1
                    print(f"  [{record_id - 1}] category={category!r} difficulty={difficulty!r} -> {record}")

        return dataset


def main():
    print("Generador de Data Assets para Evaluation Prompt Workflow")
    print("----------------------------------------------------------")
    print(f'Prefill activo   : "{DEFAULT_PREFILL}"')
    print(f'Stop sequences   : {DEFAULT_STOP_SEQUENCES}')
    print()

    domain: str = input("Dominio del dataset (ej. 'atención al cliente', 'SQL', 'medicina'): ").strip()
    if not domain:
        domain = "atención al cliente"

    categories_input: str = input(
        "Categorías separadas por coma (ej. 'comprensión,razonamiento,extracción'): "
    ).strip()
    categories: list[str] = (
        [c.strip() for c in categories_input.split(",") if c.strip()]
        if categories_input
        else ["comprensión", "razonamiento", "extracción"]
    )

    difficulties: list[str] = ["easy", "medium", "hard"]

    samples_input: str = input("Muestras por combinación [1]: ").strip()
    samples: int = int(samples_input) if samples_input.isdigit() else 1

    print(f"\nGenerando dataset: dominio={domain!r}, categorías={categories}, dificultades={difficulties}, muestras_por_combo={samples}\n")

    generator = EvalDataGenerator()
    dataset: list[dict] = generator.generate_dataset(
        domain=domain,
        categories=categories,
        difficulties=difficulties,
        samples_per_combination=samples,
    )

    output_path: str = "eval_dataset.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"\nDataset generado: {len(dataset)} registros -> {output_path}")
