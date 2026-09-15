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

"""Breakable-prefill graph state must not impersonate a full CUDA graph.

A breakable graph ends its current capture segment before running an eager
attention break.  Full-graph stream forks may remain open across that call and
therefore cannot be enabled by the breakable-prefill phase.  These CPU tests
pin both the standalone context-manager lifecycle and ``PrefillGraph.capture``'s
success/failure cleanup without exercising CUDA.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest
import torch

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

from tokenspeed.runtime.execution.forward_step import (
    get_is_capture_mode,
    get_is_cuda_graph_phase,
    get_is_prefill_graph_phase,
    prefill_graph_phase,
)
from tokenspeed.runtime.execution.prefill_graph import PrefillGraph

register_cuda_ci(est_time=5, suite="runtime-1gpu")


def _bare_prefill_graph(callback) -> PrefillGraph:
    graph = PrefillGraph.__new__(PrefillGraph)
    graph.disable = False
    graph.capture_buckets = [4]
    graph.attn_backend = SimpleNamespace(init_prefill_graph_state=lambda **_: None)
    graph.config = SimpleNamespace(max_num_seqs=4, data_parallel_size=1)
    graph._embed_tokens = SimpleNamespace(weight=torch.zeros(2, 8, dtype=torch.float32))
    graph._capture_all_buckets = callback
    return graph


def _assert_prefill_only_state() -> None:
    assert get_is_prefill_graph_phase()
    assert not get_is_cuda_graph_phase()
    assert not get_is_capture_mode()


def test_prefill_graph_phase_is_nested_and_exception_safe() -> None:
    assert not get_is_prefill_graph_phase()
    _assert_generic_graph_state_is_clear()

    with pytest.raises(RuntimeError, match="sentinel"):
        with prefill_graph_phase():
            _assert_prefill_only_state()
            with prefill_graph_phase():
                _assert_prefill_only_state()
            _assert_prefill_only_state()
            raise RuntimeError("sentinel")

    assert not get_is_prefill_graph_phase()
    _assert_generic_graph_state_is_clear()


def _assert_generic_graph_state_is_clear() -> None:
    assert not get_is_cuda_graph_phase()
    assert not get_is_capture_mode()


def test_prefill_capture_publishes_only_prefill_phase_and_restores_it() -> None:
    observations: list[tuple[bool, bool, bool]] = []

    def observe(_decode_wrapper) -> None:
        observations.append(
            (
                get_is_prefill_graph_phase(),
                get_is_cuda_graph_phase(),
                get_is_capture_mode(),
            )
        )

    _bare_prefill_graph(observe).capture(None)

    assert observations == [(True, False, False)]
    assert not get_is_prefill_graph_phase()
    _assert_generic_graph_state_is_clear()


def test_prefill_capture_failure_restores_prefill_phase() -> None:
    cause = RuntimeError("capture failed")

    def fail(_decode_wrapper) -> None:
        _assert_prefill_only_state()
        raise cause

    with pytest.raises(RuntimeError) as caught:
        _bare_prefill_graph(fail).capture(None)

    assert caught.value is cause
    assert not get_is_prefill_graph_phase()
    _assert_generic_graph_state_is_clear()
