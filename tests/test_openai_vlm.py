from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from utils.openai_vlm import (
    OpenAICompatibleVLM,
    ScenePlan,
    load_scene_plan,
    write_scene_plan,
)


VALID_PLAN = {
    "layout": "A compact grid with a waterfront park.",
    "building": "One detailed cyberpunk building on a white background.",
    "tree": "One complete cyberpunk street tree on a white background.",
    "lamp": "One complete cyberpunk street lamp on a white background.",
    "ground": "A seamless dark urban ground base-color texture.",
    "grass": "A seamless dark teal grass base-color texture.",
    "road": "A seamless charcoal asphalt base-color texture.",
    "water": "A seamless deep-blue water base-color texture.",
    "sky": "A seamless 2:1 cyberpunk evening skybox.",
}


class FakeCompletions:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self.responses:
            raise AssertionError("No fake response remains.")
        message = SimpleNamespace(content=self.responses.pop(0))
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeClient:
    def __init__(self, responses: list[str]) -> None:
        self.completions = FakeCompletions(responses)
        self.chat = SimpleNamespace(completions=self.completions)


class OpenAICompatibleVLMTests(unittest.TestCase):
    def client(self, responses: list[str], **kwargs) -> OpenAICompatibleVLM:
        return OpenAICompatibleVLM(
            planner_model="planner-model",
            review_model="review-model",
            client=FakeClient(responses),
            **kwargs,
        )

    def test_scene_prompt_returns_validated_fields(self) -> None:
        planner = self.client([json.dumps(VALID_PLAN)])
        result = planner.scene_prompt("A compact city", style="cyberpunk")
        self.assertEqual(result, VALID_PLAN)
        call = planner.client.completions.calls[0]
        self.assertEqual(call["model"], "planner-model")
        self.assertIn("cyberpunk", call["messages"][1]["content"])

    def test_scene_prompt_retries_invalid_json(self) -> None:
        planner = self.client(
            ["not json", f"```json\n{json.dumps(VALID_PLAN)}\n```"],
            plan_attempts=2,
        )
        self.assertEqual(planner.scene_prompt("A city"), VALID_PLAN)
        self.assertEqual(len(planner.client.completions.calls), 2)

    def test_image_review_uses_openai_image_url_content(self) -> None:
        reviewer = self.client(["true"])
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "building.png"
            image_path.write_bytes(b"not-decoded-by-the-api-adapter")
            self.assertTrue(reviewer.image_review(image_path, "building"))
        call = reviewer.client.completions.calls[0]
        self.assertEqual(call["model"], "review-model")
        content = call["messages"][1]["content"]
        self.assertTrue(content[0]["image_url"]["url"].startswith("data:image/png;base64,"))

    def test_scene_plan_record_round_trip(self) -> None:
        plan = ScenePlan.from_mapping(VALID_PLAN)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scene_plan.json"
            write_scene_plan(
                path,
                plan,
                source_prompt="A city",
                style="cyberpunk",
                provider={"protocol": "openai_chat_completions"},
            )
            self.assertEqual(load_scene_plan(path), plan)


if __name__ == "__main__":
    unittest.main()
