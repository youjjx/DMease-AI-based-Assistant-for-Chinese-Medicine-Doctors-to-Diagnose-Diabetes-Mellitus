"""Publish only checksum-verified small planning checkpoints, without training data."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def main():
    folder = ROOT/'experiments/v4'
    protocol = json.loads((folder/'planning_protocol.json').read_text(encoding='utf8'))
    checked = []
    for variant in protocol['variants']:
        for seed in protocol['seeds']:
            name = f'{variant}_{seed}'
            meta = json.loads((folder/f'planning_training/{name}.json').read_text(encoding='utf8'))
            for suffix, key in (('', 'checkpoint_sha256'), ('_warm', 'warm_checkpoint_sha256')):
                source = ROOT/f'outputs/research_v4/planning/{name}{suffix}.pt'
                if hashlib.sha256(source.read_bytes()).hexdigest() != meta[key]:
                    raise ValueError('Planning checkpoint does not match frozen training selection: '+name+suffix)
                checked.append(source)
    destination = ROOT/'models/planning_v4'; destination.mkdir(parents=True, exist_ok=True)
    for source in checked:
        shutil.copyfile(source, destination/source.name)
    print(f'Exported {len(checked)} verified selected/warm checkpoints.')


if __name__ == '__main__':
    main()
