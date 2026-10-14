#!/usr/bin/env python
"""Export inference weights from a trusted local PyTorch checkpoint.

The output contains only ``state_dict``. Training configuration, optimizer
state, message history, and other top-level checkpoint fields are omitted.
Only load trusted checkpoints: the input uses PyTorch's pickle-based loader.
"""

import argparse
from collections.abc import Mapping
from pathlib import Path

import torch


class ExportError(ValueError):
    """A validation error whose message does not contain filesystem paths."""


def export_weights(checkpoint, output):
    """Save a new checkpoint without changing tensor names, values, or dtypes.

    Accept either an MMEngine checkpoint containing ``state_dict`` or a raw
    tensor state dict. Reject non-tensor entries instead of silently dropping
    model state. Existing output files are never overwritten.
    """
    checkpoint = Path(checkpoint)
    output = Path(output)
    if checkpoint.resolve() == output.resolve():
        raise ExportError('Input and output must be different files.')
    if output.exists() or output.is_symlink():
        raise FileExistsError('The output already exists; choose a new file.')
    if not checkpoint.is_file():
        raise ExportError('The input must be an existing checkpoint file.')
    if not output.parent.is_dir():
        raise ExportError('The output parent directory must already exist.')

    loaded = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if not isinstance(loaded, Mapping):
        raise ExportError('The checkpoint must be a mapping.')
    state_dict = loaded.get('state_dict', loaded)
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise ExportError('The checkpoint must contain a nonempty state dict.')
    if not all(isinstance(key, str) and isinstance(value, torch.Tensor)
               for key, value in state_dict.items()):
        raise ExportError('The state dict must contain string keys and tensors.')

    # Keep the state dict itself intact, including PyTorch module-version
    # metadata when present. Only checkpoint-level training state is removed.
    # Exclusive creation also prevents overwriting a file created meanwhile.
    with output.open('xb') as stream:
        try:
            torch.save({'state_dict': state_dict}, stream)
        except Exception:
            output.unlink()
            raise
    return len(state_dict), sum(value.numel() for value in state_dict.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', help='Trusted local input checkpoint')
    parser.add_argument('output', help='New output file; parent must exist')
    args = parser.parse_args()
    try:
        count, elements = export_weights(args.checkpoint, args.output)
    except ExportError as error:
        parser.exit(1, 'Export failed: {}\n'.format(error))
    except FileExistsError:
        parser.exit(1, 'Export failed: the output already exists.\n')
    except Exception as error:
        # Exception messages from file I/O or pickle may expose source paths.
        parser.exit(1, 'Export failed ({}). Check the input format and file '
                    'permissions.\n'.format(type(error).__name__))
    print('Exported {} tensor entries ({} elements).'.format(count, elements))
    print('Saved only state_dict; checkpoint training metadata was omitted.')


if __name__ == '__main__':
    main()
