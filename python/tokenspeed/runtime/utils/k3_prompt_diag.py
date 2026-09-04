# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Diagnostic prompt-id rewrite for Kimi-K3 chat prompts.

Kimi-K3's reference renderer (``encoding_k3.py`` shipped with the model)
encodes template structure with control tokens allowed and message *text*
with them disallowed, so a marker string such as ``<|close|>`` inside an
assistant message becomes plain BPE pieces. A gateway that tokenizes the
rendered prompt as one string instead maps those strings to the control ids.
``rewrite_assistant_content`` converts a prompt of the second kind into the
first, so both renderings can be compared on identical engine input.

Only history assistant messages are rewritten: user text in the agentic
benchmark contains no marker strings, and messages whose body does not match
the expected wrapper are left untouched.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Sequence

logger = logging.getLogger(__name__)

_ASSISTANT_OPENER = '<|open|>message role="assistant"<|sep|>'
_OPEN_RESPONSE = "<|open|>response<|sep|>"
_CLOSE_RESPONSE_AND_MESSAGE = "<|close|>response<|sep|><|close|>message<|sep|>"
_END_OF_MSG = "<|end_of_msg|>"


@dataclass(frozen=True)
class K3Patterns:
    """Template fragments as token ids, derived from the serving tokenizer."""

    assistant_opener: tuple[int, ...]
    open_response: tuple[int, ...]
    close_response_and_message: tuple[int, ...]
    end_of_msg: int

    @classmethod
    def from_tokenizer(cls, tokenizer: Any) -> "K3Patterns":
        enc = _structural_encoder(tokenizer)
        eom = enc(_END_OF_MSG)
        if len(eom) != 1:
            raise ValueError(f"{_END_OF_MSG!r} is not a single token: {eom}")
        return cls(
            assistant_opener=tuple(enc(_ASSISTANT_OPENER)),
            open_response=tuple(enc(_OPEN_RESPONSE)),
            close_response_and_message=tuple(enc(_CLOSE_RESPONSE_AND_MESSAGE)),
            end_of_msg=eom[0],
        )


def _structural_encoder(tokenizer: Any) -> Callable[[str], list[int]]:
    def enc(text: str) -> list[int]:
        return list(tokenizer.encode(text, allow_special_tokens=True))

    return enc


def _plain_encoder(tokenizer: Any) -> Callable[[str], list[int]]:
    def enc(text: str) -> list[int]:
        return list(tokenizer.encode(text, allow_special_tokens=False))

    return enc


def supports_rewrite(tokenizer: Any) -> bool:
    """True if the tokenizer exposes Kimi's ``encode(..., allow_special_tokens)``."""
    try:
        tokenizer.encode("", allow_special_tokens=False)
    except TypeError:
        return False
    except Exception:  # noqa: BLE001 - any other failure means unsupported too
        return False
    return True


def _find(seq: Sequence[int], pattern: Sequence[int], start: int = 0) -> int:
    n, m = len(seq), len(pattern)
    if m == 0:
        return start
    first = pattern[0]
    i = start
    while i <= n - m:
        if seq[i] == first and list(seq[i : i + m]) == list(pattern):
            return i
        i += 1
    return -1


def rewrite_assistant_content(
    ids: Sequence[int],
    patterns: K3Patterns,
    decode: Callable[[Sequence[int]], str],
    encode_plain: Callable[[str], list[int]],
) -> list[int]:
    """Re-encode the text of every history assistant message with control
    tokens disallowed, leaving the template wrapper ids unchanged.

    Args:
        ids: prompt token ids as produced by the gateway.
        patterns: template fragments for this tokenizer.
        decode: ids -> text, must render control ids as their marker strings.
        encode_plain: text -> ids with control tokens disallowed.

    Returns:
        The rewritten id list. Messages whose body does not match
        ``<|open|>response<|sep|> ... <|close|>response<|sep|><|close|>message<|sep|>``
        followed by ``<|end_of_msg|>`` are copied verbatim.
    """
    ids = list(ids)
    opener = list(patterns.assistant_opener)
    open_resp = list(patterns.open_response)
    close_seq = list(patterns.close_response_and_message)
    out: list[int] = []
    i = 0
    n = len(ids)
    while i < n:
        j = _find(ids, opener, i)
        if j < 0:
            out.extend(ids[i:])
            break
        body_start = j + len(opener)
        try:
            eom = ids.index(patterns.end_of_msg, body_start)
        except ValueError:
            out.extend(ids[i:])
            break
        body = ids[body_start:eom]
        k = _find(body, open_resp)
        tail = len(body) - len(close_seq)
        if k < 0 or tail < k + len(open_resp) or body[tail:] != close_seq:
            out.extend(ids[i : eom + 1])
            i = eom + 1
            continue
        content_start = k + len(open_resp)
        content = body[content_start:tail]
        out.extend(ids[i:body_start])
        out.extend(body[:content_start])
        out.extend(encode_plain(decode(content)))
        out.extend(body[tail:])
        out.append(patterns.end_of_msg)
        i = eom + 1
    return out


def filter_special_token_ids(ids: Sequence[int], special_ids) -> list[int]:
    """Drop ids in ``special_ids`` from an output-id list.

    The smg gateway detokenizes from the token ids the engine hands over, so
    engine-side skip_special_tokens never reaches the streamed text. Removing
    the special ids here gives the gateway the same text vLLM's front end
    produces with skip_special_tokens=True.
    """
    return [t for t in ids if t not in special_ids]


class PromptRewriter:
    """Caches the tokenizer-derived patterns and applies the rewrite."""

    def __init__(self, tokenizer: Any):
        self._tokenizer = tokenizer
        self._patterns = K3Patterns.from_tokenizer(tokenizer)
        self._encode_plain = _plain_encoder(tokenizer)

    @property
    def patterns(self) -> K3Patterns:
        return self._patterns

    def __call__(self, ids: Sequence[int]) -> list[int]:
        out = rewrite_assistant_content(
            ids, self._patterns, self._tokenizer.decode, self._encode_plain
        )
        if len(out) != len(ids):
            logger.debug("k3 diag rewrite: %d -> %d prompt tokens", len(ids), len(out))
        return out
