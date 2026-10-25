"""AWS Bedrock / Claude LLM integration."""

import json
import base64


# ─── Constants ───────────────────────────────────────────────────────────────

MODEL_ID = "us.anthropic.claude-sonnet-5-5"           # Sonnet 4 reached end of life on Bedrock (2026-10)
JUDGE_MODEL_ID = "us.anthropic.claude-sonnet-5-5"   # offline relevance judging (eval harnesses)
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

    from botocore.exceptions import BotoCoreError, ClientError

    client = _get_bedrock_runtime()
    try:
        response = client.invoke_model(
            body=json.dumps(body),
            modelId=model_id or MODEL_ID,
            accept="application/json",
            contentType="application/json",
        )
    except (BotoCoreError, ClientError) as e:   # missing/expired credentials, no model access
        raise AWSConnectionError(f"Bedrock call failed: {e}") from e
    result = json.loads(response["body"].read())
    # Newer models may emit a "thinking" block before the answer; return the text blocks only.
    return "".join(block["text"] for block in result["content"] if block.get("type") == "text")


def invoke_claude_tools(
    messages: list[dict],
    tools: list[dict],
    *,
    system: str = None,
    max_tokens: int = 2048,
    model_id: str = None,
) -> dict:
    """
    One tool-calling turn with Claude via AWS Bedrock (Anthropic messages format).

    Args:
        messages: The conversation so far, as Anthropic message dicts. Assistant turns are
            passed back with their content blocks unchanged (text, tool_use).
        tools: Tool declarations: {"name", "description", "input_schema"} each; an
            empty list makes this a plain completion that still reports token usage.
        system: Optional system prompt.
        max_tokens: Maximum response tokens.
        model_id: Override the default model ID.

    Returns:
        The raw response dict: "content" (text and tool_use blocks), "stop_reason"
        ("tool_use" when the model requests tools, "end_turn" when it answers), "usage".

    Raises:
        AWSConnectionError: If AWS credentials are unavailable.
    """
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "messages": messages,
    }
    if tools:
        body["tools"] = tools
    if system:
        body["system"] = system

    from botocore.exceptions import BotoCoreError, ClientError

    client = _get_bedrock_runtime()
    try:
        response = client.invoke_model(
            body=json.dumps(body),
            modelId=model_id or MODEL_ID,
            accept="application/json",
            contentType="application/json",
        )
    except (BotoCoreError, ClientError) as e:
        raise AWSConnectionError(f"Bedrock call failed: {e}") from e
    return json.loads(response["body"].read())
