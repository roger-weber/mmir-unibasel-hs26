"""AWS Bedrock / Claude LLM integration."""

import json
import base64


# ─── Constants ───────────────────────────────────────────────────────────────

MODEL_ID = "us.anthropic.claude-sonnet-4-20250514-v1:0"
AWS_REGION = "us-west-2"


# ─── Exceptions ──────────────────────────────────────────────────────────────

class AWSConnectionError(Exception):
    """Raised when AWS credentials are missing or invalid."""
    pass


# ─── Client Management ───────────────────────────────────────────────────────

_bedrock_runtime_client = None


def _get_bedrock_runtime():
    """Lazy-initialize the Bedrock runtime client."""
    global _bedrock_runtime_client
    if _bedrock_runtime_client is None:
        try:
            import boto3
            session = boto3.Session(region_name=AWS_REGION)
            _bedrock_runtime_client = session.client("bedrock-runtime", region_name=AWS_REGION)
        except Exception:
            raise AWSConnectionError(
                "No valid AWS credentials found. LLM demos require AWS Bedrock access."
            ) from None
    return _bedrock_runtime_client


# ─── Public API ──────────────────────────────────────────────────────────────

def invoke_claude(
    prompt: str,
    *,
    image: bytes = None,
    system: str = None,
    max_tokens: int = 4096,
    history: list[str] = None,
    model_id: str = None,
) -> str:
    """
    Invoke Claude via AWS Bedrock.

    Args:
        prompt: The user message text.
        image: Optional JPEG image bytes to include.
        system: Optional system prompt.
        max_tokens: Maximum response tokens.
        history: Optional conversation history (alternating user/assistant strings).
        model_id: Override the default model ID.

    Returns:
        Claude's text response.

    Raises:
        AWSConnectionError: If AWS credentials are unavailable.
    """
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "messages": [],
    }

    if system:
        body["system"] = system

    # Add conversation history
    if history:
        roles = ["user", "assistant"]
        for i, message in enumerate(history):
            body["messages"].append({
                "role": roles[i % 2],
                "content": [{"type": "text", "text": message}],
            })

    # Build current user message
    content = [{"type": "text", "text": prompt}]
    if image:
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": base64.b64encode(image).decode("utf-8"),
            },
        })
    body["messages"].append({"role": "user", "content": content})

    client = _get_bedrock_runtime()
    response = client.invoke_model(
        body=json.dumps(body),
        modelId=model_id or MODEL_ID,
        accept="application/json",
        contentType="application/json",
    )
    result = json.loads(response["body"].read())
    return result["content"][0]["text"]
