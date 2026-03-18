import json
import os

import openai
from dotenv import load_dotenv
from rich.console import Console

from feedback_clustering.exceptions import ConfigurationError
from feedback_clustering.models import Cluster, FeedbackItem

load_dotenv()

_console = Console(stderr=True)

_SYSTEM_PROMPT = """You are a product analyst. Given a set of user feedback items that have been grouped together by semantic similarity, produce a concise label and analysis for this cluster.

Respond with a JSON object containing exactly these keys:
- "label": a short (3-6 word) descriptive label for the theme
- "description": a 1-2 sentence description of what the feedback in this cluster is about
- "suggested_action": a concrete, actionable recommendation for the product team

Example response:
{"label": "Mobile Login Issues", "description": "Users are experiencing login failures on mobile devices, particularly on iOS.", "suggested_action": "Audit the mobile authentication flow and add end-to-end tests for iOS login scenarios."}"""


def label_clusters(
    clusters_raw: list[tuple[int, list[FeedbackItem]]],
) -> list[Cluster]:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ConfigurationError(
            "OPENAI_API_KEY environment variable is not set. "
            "Add it to your .env file or export it before running."
        )

    client = openai.OpenAI(api_key=api_key)
    result: list[Cluster] = []

    for cluster_id, items in clusters_raw:
        representative_items = items[:3]
        representative_texts = [it.text for it in representative_items]

        feedback_block = "\n".join(
            f"- {text}" for text in [it.text for it in items]
        )
        user_message = (
            f"Here are {len(items)} feedback items grouped together:\n\n{feedback_block}\n\n"
            "Provide the label, description, and suggested action for this cluster."
        )

        label = f"Cluster {cluster_id}"
        description = "A group of related feedback items."
        suggested_action = "Review this cluster and identify common patterns."

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.3,
            )
            raw_content = response.choices[0].message.content or ""
            parsed = json.loads(raw_content)
            label = str(parsed.get("label", label)) or label
            description = str(parsed.get("description", description)) or description
            suggested_action = (
                str(parsed.get("suggested_action", suggested_action)) or suggested_action
            )
        except (json.JSONDecodeError, KeyError, Exception) as exc:
            _console.print(
                f"[yellow]Warning:[/yellow] Failed to parse label for cluster {cluster_id}: {exc}. "
                "Using fallback label."
            )

        result.append(
            Cluster(
                id=cluster_id,
                label=label,
                description=description,
                suggested_action=suggested_action,
                size=len(items),
                representative_examples=representative_texts,
            )
        )

    return result
