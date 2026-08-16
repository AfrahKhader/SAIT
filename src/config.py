"""Typed configuration loaded from the YAML files in SAIT/config.

All paths are anchored to the project root (the SAIT folder), computed from this
file's location — so the app works whether you launch it from SAIT/, from src/,
or from an IDE run button.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import yaml

# src/config.py -> src -> SAIT
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _resolve(path: str) -> str:
    """Make a config-relative path absolute, anchored at the project root."""
    p = Path(path)
    return str(p if p.is_absolute() else (PROJECT_ROOT / p))


@dataclass
class LLMConfig:
    tutor_model: str
    tutor_temperature: float
    guard_model: str
    guard_temperature: float
    embedding_model: str


@dataclass
class RetrieverConfig:
    k: int
    search_type: str


@dataclass
class DocumentsConfig:
    pdf_dir: str
    chunk_size: int
    chunk_overlap: int


@dataclass
class VectorStoreConfig:
    persist_directory: str
    collection_name: str
    rebuild: bool


@dataclass
class MisconceptionsConfig:
    enabled: bool
    catalog_path: str
    questions_path: str


@dataclass
class SessionConfig:
    thread_id: str


@dataclass
class GuardrailConfig:
    min_length: int


@dataclass
class Prompts:
    socratic_system_prompt: str
    guard_system_prompt: str
    redirects: Dict[str, str]


@dataclass
class Config:
    llm: LLMConfig
    retriever: RetrieverConfig
    documents: DocumentsConfig
    vectorstore: VectorStoreConfig
    misconceptions: MisconceptionsConfig
    session: SessionConfig
    guardrail: GuardrailConfig
    prompts: Prompts

    @classmethod
    def load(cls, config_path: str | None = None, prompts_path: str | None = None) -> "Config":
        """Read both YAML files and assemble a fully-typed Config.

        Defaults point at SAIT/config/*.yaml via the project root.
        """
        config_path = config_path or PROJECT_ROOT / "config" / "config.yaml"
        prompts_path = prompts_path or PROJECT_ROOT / "config" / "prompts.yaml"

        cfg = _read_yaml(config_path)
        prm = _read_yaml(prompts_path)

        documents = DocumentsConfig(**cfg["documents"])
        documents.pdf_dir = _resolve(documents.pdf_dir)

        vectorstore = VectorStoreConfig(**cfg["vectorstore"])
        vectorstore.persist_directory = _resolve(vectorstore.persist_directory)

        misconceptions = MisconceptionsConfig(**cfg["misconceptions"])
        misconceptions.catalog_path = _resolve(misconceptions.catalog_path)
        misconceptions.questions_path = _resolve(misconceptions.questions_path)

        return cls(
            llm=LLMConfig(**cfg["llm"]),
            retriever=RetrieverConfig(**cfg["retriever"]),
            documents=documents,
            vectorstore=vectorstore,
            misconceptions=misconceptions,
            session=SessionConfig(**cfg["session"]),
            guardrail=GuardrailConfig(**cfg["guardrail"]),
            prompts=Prompts(
                socratic_system_prompt=prm["socratic_system_prompt"],
                guard_system_prompt=prm["guard_system_prompt"],
                redirects=prm["redirects"],
            ),
        )


def _read_yaml(path) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p.resolve()}")
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)