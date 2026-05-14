from pathlib import Path
from functools import lru_cache


class PromptService:
    """
    Este servicio centraliza la carga de prompts.

    Funciones públicas:
     - Cargar un prompt por ruta relativa (load_prompt).
    """

    def __init__(self) -> None:
        self.base_path = Path(__file__).resolve().parent.parent / "prompts"


    @lru_cache(maxsize=None)
    def load_prompt(self, relative_path: str) -> str:
        prompt_path = self.base_path / relative_path
        return prompt_path.read_text(encoding="utf-8").strip()
