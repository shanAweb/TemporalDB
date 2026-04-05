import re
import unicodedata


def normalize(text: str) -> str:
    """Clean and normalize raw extracted text for NLP processing.

    Steps applied in order:
    1. Unicode NFC normalization
    2. Remove null bytes and non-printable control characters
    3. Normalize line endings to '\\n'
    4. Collapse runs of blank lines to a single blank line
    5. Strip leading/trailing whitespace from each line
    6. Collapse intra-line whitespace runs to a single space
    7. Strip overall leading/trailing whitespace

    Args:
        text: Raw text as returned by a connector.

    Returns:
        Cleaned, normalized text string.
    """
    # 1. Unicode NFC normalization
    text = unicodedata.normalize("NFC", text)

    # 2. Remove null bytes and non-printable control characters
    #    Keep: \n (0x0A), \t (0x09), standard printable chars
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # 3. Normalize line endings (CRLF and CR → LF)
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 4. Strip leading/trailing whitespace from each line
    lines = [line.strip() for line in text.split("\n")]

    # 5. Collapse intra-line whitespace runs to a single space
    lines = [re.sub(r"[ \t]+", " ", line) for line in lines]

    # 6. Collapse runs of more than two consecutive blank lines to two
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 7. Strip overall leading/trailing whitespace
    return text.strip()
