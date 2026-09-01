"""OpenAI-compatible scene planning and visual review helpers.

The client deliberately depends only on the OpenAI Chat Completions protocol.
It can therefore target OpenAI itself or another server exposing a compatible
``/v1/chat/completions`` endpoint. Credentials are read from environment
variables and are never written to pipeline manifests.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPT_PATH = PROJECT_ROOT / "configs" / "scene_planner_prompts.json"
SCENE_PLAN_FIELDS = (
    "layout",
    "building",
    "tree",
    "lamp",
    "ground",
    "grass",
    "road",
    "water",
    "sky",
)
REVIEW_KINDS = ("building", "tree", "lamp")


@dataclass(frozen=True)
class ScenePlan:
    """Validated prompts produced by the Scene Design model."""

    layout: str
    building: str
    tree: str
    lamp: str
    ground: str
    grass: str
    road: str
    water: str
    sky: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ScenePlan":
        missing = [key for key in SCENE_PLAN_FIELDS if key not in value]
        extra = [key for key in value if key not in SCENE_PLAN_FIELDS]
        invalid = [
            key
            for key in SCENE_PLAN_FIELDS
            if key in value
            and (not isinstance(value[key], str) or not value[key].strip())
        ]
        if missing or extra or invalid:
            details = []
            if missing:
                details.append("missing=" + ",".join(missing))
            if extra:
                details.append("unexpected=" + ",".join(extra))
            if invalid:
                details.append("non_empty_string_required=" + ",".join(invalid))
            raise ValueError("Invalid Scene Plan: " + "; ".join(details))
        return cls(**{key: value[key].strip() for key in SCENE_PLAN_FIELDS})

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def parse_json_object(text: str) -> dict[str, Any]:
    """Extract one JSON object from a model response."""

    normalized = text.strip()
    normalized = re.sub(r"^```(?:json)?\s*", "", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\s*```$", "", normalized)
    normalized = normalized.replace("“", '"').replace("”", '"')
    start = normalized.find("{")
    end = normalized.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("The model response did not contain a JSON object.")
    parsed = json.loads(normalized[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("The model response must contain one JSON object.")
    return parsed


def load_scene_plan(path: str | Path) -> ScenePlan:
    plan_path = Path(path).expanduser().resolve()
    with plan_path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    if isinstance(value, dict) and "plan" in value:
        value = value["plan"]
    if not isinstance(value, dict):
        raise ValueError(f"Expected a Scene Plan object: {plan_path}")
    return ScenePlan.from_mapping(value)


def write_scene_plan(
    path: str | Path,
    plan: ScenePlan,
    *,
    source_prompt: str,
    style: str,
    provider: Mapping[str, Any] | None = None,
) -> Path:
    plan_path = Path(path).expanduser().resolve()
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {
        "version": 1,
        "source_prompt": source_prompt,
        "style": style,
        "plan": plan.to_dict(),
    }
    if provider is not None:
        record["provider"] = dict(provider)
    with plan_path.open("w", encoding="utf-8") as file:
        json.dump(record, file, ensure_ascii=False, indent=2)
        file.write("\n")
    return plan_path


class OpenAICompatibleVLM:
    """Scene planner and visual reviewer backed by an OpenAI-compatible API."""

    def __init__(
        self,
        *,
        planner_model: str,
        review_model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        prompt_path: str | Path = DEFAULT_PROMPT_PATH,
        timeout: float = 120.0,
        max_retries: int = 2,
        plan_attempts: int = 2,
        client: Any | None = None,
    ) -> None:
        if not planner_model.strip():
            raise ValueError("planner_model must not be empty.")
        if timeout <= 0:
            raise ValueError("timeout must be positive.")
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative.")
        if plan_attempts <= 0:
            raise ValueError("plan_attempts must be positive.")

        self.planner_model = planner_model.strip()
        self.review_model = (review_model or planner_model).strip()
        self.base_url = base_url.rstrip("/") if base_url else None
        self.prompt_path = Path(prompt_path).expanduser().resolve()
        self.plan_attempts = plan_attempts
        self.prompts = self._load_prompts(self.prompt_path)

        if client is None:
            try:
                from openai import OpenAI
            except ImportError as error:
                raise RuntimeError(
                    "The optional OpenAI API feature requires the 'openai' package."
                ) from error
            if not api_key:
                if self.base_url:
                    # Many local compatible servers ignore the key, while the SDK
                    # still requires a non-empty value.
                    api_key = "not-required"
                else:
                    raise RuntimeError(
                        "MAJUTSUCITY_LLM_API_KEY is required when no custom base URL is set."
                    )
            client_kwargs: dict[str, Any] = {
                "api_key": api_key,
                "timeout": timeout,
                "max_retries": max_retries,
            }
            if self.base_url:
                client_kwargs["base_url"] = self.base_url
            client = OpenAI(**client_kwargs)
        self.client = client

    @classmethod
    def from_environment(
        cls,
        *,
        prompt_path: str | Path = DEFAULT_PROMPT_PATH,
        planner_model: str | None = None,
        review_model: str | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        plan_attempts: int | None = None,
    ) -> "OpenAICompatibleVLM":
        resolved_planner = (
            planner_model
            or os.environ.get("MAJUTSUCITY_PLANNER_MODEL")
            or review_model
            or os.environ.get("MAJUTSUCITY_REVIEW_MODEL", "")
        )
        if not resolved_planner:
            raise RuntimeError(
                "Set MAJUTSUCITY_PLANNER_MODEL or MAJUTSUCITY_REVIEW_MODEL "
                "before enabling an API-backed language or vision feature."
            )
        return cls(
            planner_model=resolved_planner,
            review_model=(
                review_model
                or os.environ.get("MAJUTSUCITY_REVIEW_MODEL")
                or resolved_planner
            ),
            base_url=os.environ.get("MAJUTSUCITY_LLM_BASE_URL"),
            api_key=os.environ.get("MAJUTSUCITY_LLM_API_KEY"),
            prompt_path=prompt_path,
            timeout=float(
                timeout
                if timeout is not None
                else os.environ.get("MAJUTSUCITY_LLM_TIMEOUT", "120")
            ),
            max_retries=int(
                max_retries
                if max_retries is not None
                else os.environ.get("MAJUTSUCITY_LLM_MAX_RETRIES", "2")
            ),
            plan_attempts=int(
                plan_attempts
                if plan_attempts is not None
                else os.environ.get("MAJUTSUCITY_LLM_PLAN_ATTEMPTS", "2")
            ),
        )

    @staticmethod
    def _load_prompts(path: Path) -> dict[str, str]:
        with path.open("r", encoding="utf-8") as file:
            raw = json.load(file)
        if not isinstance(raw, dict):
            raise ValueError(f"Expected a prompt mapping: {path}")
        required = ("scene_prompt", *(f"{kind}_review" for kind in REVIEW_KINDS))
        result = {}
        for key in required:
            value = raw.get(key)
            if isinstance(value, list) and all(isinstance(item, str) for item in value):
                value = "\n".join(value)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Prompt {key!r} must be a string or string list: {path}")
            result[key] = value.strip()
        return result

    @staticmethod
    def _message_text(response: Any) -> str:
        try:
            content = response.choices[0].message.content
        except (AttributeError, IndexError, TypeError) as error:
            raise ValueError("The API response did not contain assistant content.") from error
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif hasattr(item, "text") and isinstance(item.text, str):
                    parts.append(item.text)
            if parts:
                return "\n".join(parts)
        raise ValueError("The API response contained unsupported assistant content.")

    def _chat(self, *, model: str, messages: list[dict[str, Any]], max_tokens: int) -> str:
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0,
            max_tokens=max_tokens,
        )
        return self._message_text(response)

    def scene_prompt(self, user_input: str, style: str | None = None) -> dict[str, str]:
        if not user_input.strip():
            raise ValueError("The Scene Design prompt must not be empty.")
        scene_input = user_input.strip()
        if style:
            scene_input = f"Selected city style: {style.strip()}.\n{scene_input}"
        request = self.prompts["scene_prompt"].replace("<USER_SCENE>", scene_input)

        last_error: Exception | None = None
        messages = [
            {
                "role": "system",
                "content": (
                    "You convert city descriptions into a strict JSON Scene Plan. "
                    "Return the JSON object only."
                ),
            },
            {"role": "user", "content": request},
        ]
        for attempt in range(1, self.plan_attempts + 1):
            if attempt > 1:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "The previous response was not a valid Scene Plan. Return "
                            "only one JSON object containing exactly these non-empty "
                            f"string fields: {', '.join(SCENE_PLAN_FIELDS)}."
                        ),
                    }
                )
            try:
                text = self._chat(
                    model=self.planner_model,
                    messages=messages,
                    max_tokens=2048,
                )
                return ScenePlan.from_mapping(parse_json_object(text)).to_dict()
            except (json.JSONDecodeError, TypeError, ValueError) as error:
                last_error = error
        raise ValueError(
            f"The planner failed to return a valid Scene Plan after "
            f"{self.plan_attempts} attempt(s): {last_error}"
        ) from last_error

    def image_review(self, image_path: str | Path, review_kind: str = "building") -> bool:
        if review_kind not in REVIEW_KINDS:
            raise ValueError(
                f"Unknown review kind {review_kind!r}; choose from {REVIEW_KINDS}."
            )
        path = Path(image_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Review image not found: {path}")
        media_type = mimetypes.guess_type(path.name)[0] or "image/png"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        text = self._chat(
            model=self.review_model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a strict visual quality-control reviewer.",
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{encoded}"
                            },
                        },
                        {
                            "type": "text",
                            "text": self.prompts[f"{review_kind}_review"],
                        },
                    ],
                },
            ],
            max_tokens=8,
        )
        normalized = text.strip().strip("`").splitlines()[-1].strip().lower()
        if normalized not in ("true", "false"):
            raise ValueError(
                "The visual reviewer must return exactly 'true' or 'false'; "
                f"received {text!r}."
            )
        return normalized == "true"

    def provider_record(self) -> dict[str, str]:
        """Return safe provenance metadata without credentials."""

        safe_base_url = self.base_url or "OpenAI default"
        safe_base_url = re.sub(r"(?<=//)[^/@]+@", "[redacted]@", safe_base_url)
        return {
            "protocol": "openai_chat_completions",
            "base_url": safe_base_url,
            "planner_model": self.planner_model,
            "review_model": self.review_model,
        }
