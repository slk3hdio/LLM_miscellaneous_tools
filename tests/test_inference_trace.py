from __future__ import annotations

import torch

from model_visualizer.ui_components.inference.trace import (
    decode_token_ids,
    hidden_state_for_layer,
    next_inference_position,
    select_attention_head,
    top_token_predictions,
)


class FakeTokenizer:
    def __init__(self):
        self.vocab = {
            0: "<pad>",
            1: "Once",
            2: " upon",
            3: " time",
            4: ".",
            5: " ago",
        }

    def decode(self, token_ids, **_kwargs):
        return "".join(self.vocab[int(token_id)] for token_id in token_ids)


class FakeLMHead:
    def __init__(self, logits: torch.Tensor):
        self.logits = logits
        self.apply_final_norm_values = []

    def decode(self, _hidden_state, apply_final_norm=None):
        self.apply_final_norm_values.append(apply_final_norm)
        return self.logits


def test_decode_token_ids_preserves_order_and_ids():
    tokens = decode_token_ids(FakeTokenizer(), [1, 2, 3])

    assert [token.index for token in tokens] == [0, 1, 2]
    assert [token.token_id for token in tokens] == [1, 2, 3]
    assert [token.text for token in tokens] == ["Once", " upon", " time"]


def test_top_token_predictions_returns_top_n_per_position():
    logits = torch.tensor(
        [
            [0.1, 5.0, 2.0, 0.0, 4.0, 3.0],
            [0.1, 0.2, 7.0, 6.0, 1.0, 3.0],
        ]
    )
    lm_head = FakeLMHead(logits)

    predictions = top_token_predictions(
        torch.zeros(2, 4),
        lm_head,
        FakeTokenizer(),
        top_n=3,
        apply_final_norm=False,
    )

    assert len(predictions) == 2
    assert all(len(row) == 3 for row in predictions)
    assert [prediction.token_id for prediction in predictions[0]] == [1, 4, 5]
    assert [prediction.text for prediction in predictions[1]] == [" upon", " time", " ago"]
    expected_probability = torch.softmax(logits[0], dim=-1)[1].item()
    assert abs(predictions[0][0].probability - expected_probability) < 1e-6
    assert lm_head.apply_final_norm_values == [False]


def test_select_attention_head_returns_requested_head():
    attention = torch.arange(27, dtype=torch.float32).reshape(3, 3, 3)

    selected = select_attention_head(attention, 2)

    assert torch.equal(selected, attention[2])


def test_next_inference_position_generates_after_last_layer():
    first = next_inference_position(-1, 0, 3, has_trace=False)
    middle = next_inference_position(0, 0, 3, has_trace=True)
    rollover = next_inference_position(0, 2, 3, has_trace=True)

    assert first.generation_step == 0
    assert first.layer_index == 0
    assert first.should_generate is True
    assert middle.generation_step == 0
    assert middle.layer_index == 1
    assert middle.should_generate is False
    assert rollover.generation_step == 1
    assert rollover.layer_index == 0
    assert rollover.should_generate is True


def test_hidden_state_for_final_layer_skips_extra_final_norm():
    hidden_states = [
        torch.full((2, 3), 0.0),
        torch.full((2, 3), 1.0),
        torch.full((2, 3), 2.0),
        torch.full((2, 3), 3.0),
    ]

    middle_hidden, middle_apply_norm = hidden_state_for_layer(hidden_states, 1)
    final_hidden, final_apply_norm = hidden_state_for_layer(hidden_states, 2)

    assert torch.equal(middle_hidden, hidden_states[2])
    assert middle_apply_norm is True
    assert torch.equal(final_hidden, hidden_states[-1])
    assert final_apply_norm is False
