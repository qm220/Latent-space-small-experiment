"""Sequence-wide hidden-state capture from selected language decoder blocks."""

from __future__ import annotations

from typing import Any

import torch


def resolve_language_layers(model) -> torch.nn.ModuleList:
    language = getattr(getattr(model, "model", None), "language_model", None)
    layers = getattr(language, "layers", None)
    if layers is None:
        raise AttributeError(
            "Could not find model.model.language_model.layers. "
            f"Top-level children: {list(model._modules)}"
        )
    return layers


class SequenceActivationCapture:
    """Collect every token hidden state at selected layers during generate().

    The first forward is treated as prompt prefill (all prompt tokens). Later
    forwards append only the newest token so decode steps do not duplicate the
    prompt if the model recomputes a longer sequence.
    """

    def __init__(self, model, layer_indices: list[int]):
        self.model = model
        self.layer_indices = list(layer_indices)
        self.handles: list[Any] = []
        self.layer_names: dict[int, str] = {}
        self._prefill_done: dict[int, bool] = {idx: False for idx in self.layer_indices}
        self._chunks: dict[int, list[torch.Tensor]] = {idx: [] for idx in self.layer_indices}

    def _hook(self, layer_idx: int):
        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            cpu = hidden[0].detach().to("cpu").contiguous()
            if not self._prefill_done[layer_idx]:
                self._chunks[layer_idx].append(cpu)
                self._prefill_done[layer_idx] = True
            else:
                self._chunks[layer_idx].append(cpu[-1:])

        return hook

    def register(self) -> None:
        self.close()
        self._prefill_done = {idx: False for idx in self.layer_indices}
        self._chunks = {idx: [] for idx in self.layer_indices}
        layers = resolve_language_layers(self.model)
        n_layers = len(layers)
        for idx in self.layer_indices:
            if idx < 0 or idx >= n_layers:
                raise IndexError(f"Layer {idx} is outside 0..{n_layers - 1}")
            module = layers[idx]
            self.layer_names[idx] = f"model.model.language_model.layers.{idx}"
            self.handles.append(module.register_forward_hook(self._hook(idx)))

    def stacked(self) -> dict[int, torch.Tensor]:
        return {
            idx: torch.cat(chunks, dim=0)
            for idx, chunks in self._chunks.items()
            if chunks
        }

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def __enter__(self) -> "SequenceActivationCapture":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def pack_sequence_activations(
    stacked: dict[int, torch.Tensor],
    layer_names: dict[int, str],
    layer_indices: list[int],
    token_ids: list[int],
    token_strings: list[str],
    prompt_length: int,
) -> dict[str, Any]:
    layers = {f"layer_{idx}": stacked[idx] for idx in layer_indices if idx in stacked}
    packed = dict(layers)
    packed["token_ids"] = torch.tensor(token_ids, dtype=torch.long)
    tokens = [
        {
            "index": i,
            "id": token_id,
            "token": token_strings[i] if i < len(token_strings) else None,
            "split": "prompt" if i < prompt_length else "output",
        }
        for i, token_id in enumerate(token_ids)
    ]
    metadata = {
        "capture_location": "language_decoder_block_output",
        "capture_meaning": (
            "Hidden state of selected Qwen3-VL language decoder blocks for every "
            "prompt token (prefill) and every generated token (decode). "
            "Each layer tensor is [n_tokens, hidden_size]."
        ),
        "module_path_pattern": "model.model.language_model.layers.{index}",
        "layer_indices": layer_indices,
        "layer_names": layer_names,
        "prompt_length": prompt_length,
        "output_length": max(0, len(token_ids) - prompt_length),
        "sequence_length": len(token_ids),
        "activation_lengths": {key: int(tensor.shape[0]) for key, tensor in layers.items()},
        "shapes": {key: list(tensor.shape) for key, tensor in layers.items()},
        "dtypes": {key: str(tensor.dtype).replace("torch.", "") for key, tensor in layers.items()},
        "length_match": {
            key: int(tensor.shape[0]) == len(token_ids) for key, tensor in layers.items()
        },
        "alignment_note": (
            "Generation produces one hidden state per forward. Prefill covers all "
            "prompt tokens. Each decode step then adds one state used to sample the "
            "next token, so the final generated token usually has no extra hidden "
            "state. Align layer[i] with token_ids[i] for i < activation_length. "
            "The last output token id therefore has no matching row."
        ),
    }
    return {
        "tensors": packed,
        "metadata": metadata,
        "tokens": tokens,
    }
