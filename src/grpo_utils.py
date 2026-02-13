from typing import Any


def messages_to_text(content: Any) -> str:
    """
    Convert conversational TRL prompt/completion objects to plain text.
    Works for string, list[message], and list[content blocks].
    """

    if content is None:
        return ""

    if isinstance(content, str):
        return content

    # TRL conversational format: list[dict(role, content)]
    if isinstance(content, list):
        pieces = []
        for item in content:
            if isinstance(item, str):
                pieces.append(item)
                continue

            if not isinstance(item, dict):
                continue

            # Message dict
            if "content" in item:
                msg_content = item["content"]
                if isinstance(msg_content, str):
                    pieces.append(msg_content)
                elif isinstance(msg_content, list):
                    for block in msg_content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            pieces.append(str(block.get("text", "")))
                continue

            # Content block dict
            if item.get("type") == "text":
                pieces.append(str(item.get("text", "")))

        return "\n".join(x for x in pieces if x).strip()

    if isinstance(content, dict):
        if "content" in content:
            return messages_to_text(content["content"])
        if content.get("type") == "text":
            return str(content.get("text", ""))

    return str(content)
