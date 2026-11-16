"""Check archived panel prefixes and exercise the real decoder's first resumed step on CPU."""
import base64
import json
import sys
import tarfile
from pathlib import Path

import torch

from decoding import decode
from protocol_pilot import BLOCKS, PANEL


class BoundaryTokenizer:
    # A sentinel outside the archived vocabulary makes this CPU control independent of
    # the installed tokenizer. The cluster worker must use the pinned native tokenizer.
    eos_token_id = 0
    pad_token_id = 0

    def encode(self, text, **kwargs):
        assert text == '</think>'
        return [1_000_000]


class BoundaryModel:
    def get_input_embeddings(self):
        return torch.nn.Embedding(1, 1)

    def generate(self, input_ids, logits_processor, **kwargs):
        scores = torch.zeros(input_ids.shape[0], 1_000_001)
        assert (logits_processor(input_ids, scores).argmax(-1) == 1_000_000).all()


def verify(archive):
    torch.set_num_threads(1)
    with tarfile.open(archive) as source:
        for block in BLOCKS:
            rows = []
            for task in PANEL[block[1]]:
                state = json.load(source.extractfile(f'{block}/tasks/{task}.json'))
                assert state['task'] == task and state['condition'] == block[0]
                assert not state['done'] and not state['attempts'] and not state['probes']
                assert state['reasoning'] == state['total'] == len(state['prefix'])
                assert len(base64.b64decode(state['rng'], validate=True)) == 16
                # CUDA Philox state cannot be installed in a CPU generator. Use a
                # CPU state only in this in-memory test of the no-draw boundary.
                state['rng'] = base64.b64encode(bytes(torch.Generator().get_state().tolist())).decode()
                state['turn_reason_limit'] = state['reasoning']
                rows.append((state, [1]))
            prior = [(s['prefix'][:], s['rng']) for s, _ in rows]
            decode(BoundaryModel(), BoundaryTokenizer(), rows, lambda _: None, lambda: False, chunk=1)
            for (s, _), (prefix, rng) in zip(rows, prior):
                assert s['prefix'] == prefix + [1_000_000] and s['rng'] == rng
            print(block, len(rows), 'prefixes preserved and forced boundary appended')


if __name__ == '__main__':
    verify(Path(sys.argv[1]))
