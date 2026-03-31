from __future__ import annotations


class JsonStringStreamDecoder:
    """Incrementally decode a streamed JSON *string literal* into plain text.

    Why this exists:
    - When you use DSPy with JSONAdapter and stream a string output field via
      `dspy.streaming.StreamListener(signature_field_name="answer")`, the streamed
      chunks frequently contain *raw JSON string literal tokens*, e.g.:

        '"Here are the\\n\\n**Primary Contacts:** ...'

      That is correct JSON, but it's not what you want to show in a chat bubble.
      You want the decoded text (no surrounding quotes, real newlines).

    This decoder is a small state machine that:
    - ignores everything before the opening quote
    - decodes common escapes (\\n, \\t, \\" etc.)
    - supports unicode escapes (\\uXXXX)
    - stops after the closing quote

    It is designed to work across arbitrary chunk boundaries.

    Note:
    - This is not a general JSON parser. It only decodes a single JSON string
      literal (as produced by model streaming).
    """

    def __init__(self) -> None:
        self._started = False
        self._ended = False
        self._escape = False
        self._unicode_remaining = 0
        self._unicode_buf: list[str] = []

    @property
    def ended(self) -> bool:
        return self._ended

    def feed(self, raw: str) -> str:
        if self._ended or not raw:
            return ""

        out: list[str] = []

        for ch in raw:
            if self._ended:
                break

            if self._unicode_remaining:
                if ch.lower() in "0123456789abcdef":
                    self._unicode_buf.append(ch)
                    self._unicode_remaining -= 1
                    if self._unicode_remaining == 0:
                        try:
                            out.append(chr(int("".join(self._unicode_buf), 16)))
                        except ValueError:
                            out.append("\\u" + "".join(self._unicode_buf))
                        self._unicode_buf = []
                else:
                    # Malformed unicode escape; emit best-effort.
                    out.append("\\u" + "".join(self._unicode_buf) + ch)
                    self._unicode_remaining = 0
                    self._unicode_buf = []
                continue

            if not self._started:
                if ch == '"':
                    self._started = True
                continue

            if self._escape:
                self._escape = False
                if ch == "n":
                    out.append("\n")
                elif ch == "t":
                    out.append("\t")
                elif ch == "r":
                    out.append("\r")
                elif ch == "b":
                    out.append("\b")
                elif ch == "f":
                    out.append("\f")
                elif ch == '"':
                    out.append('"')
                elif ch == "\\":
                    out.append("\\")
                elif ch == "/":
                    out.append("/")
                elif ch == "u":
                    self._unicode_remaining = 4
                    self._unicode_buf = []
                else:
                    # Unknown escape; pass-through.
                    out.append(ch)
                continue

            if ch == "\\":
                self._escape = True
                continue

            if ch == '"':
                self._ended = True
                break

            out.append(ch)

        return "".join(out)
