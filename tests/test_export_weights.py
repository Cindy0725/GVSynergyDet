"""Check that release weights retain model state and omit training metadata."""

import subprocess
import sys
from pathlib import Path

import pytest
import torch

from projects.GVSynergyDet.tools.export_weights import export_weights


@pytest.mark.parametrize('wrapped', [True, False])
def test_export_retains_parameters_and_buffers(tmp_path, wrapped):
    model = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.BatchNorm1d(2))
    model.double()
    model[1].num_batches_tracked.fill_(17)
    state_dict = model.state_dict()
    source = tmp_path / 'training.pth'
    output = tmp_path / 'release.pth'
    payload = state_dict
    if wrapped:
        payload = {
            'state_dict': state_dict,
            'meta': {'config': 'fixture://training/data'},
            'optimizer': {'state': {'example': 'training-only metadata'}},
            'message_hub': {'best_ckpt': 'fixture://training/checkpoint'},
        }
    torch.save(payload, source)
    original_bytes = source.read_bytes()

    export_weights(source, output)

    exported = torch.load(output, map_location='cpu', weights_only=False)
    assert list(exported) == ['state_dict']
    assert list(exported['state_dict']) == list(state_dict)
    for key, expected in state_dict.items():
        actual = exported['state_dict'][key]
        assert actual.dtype == expected.dtype
        assert actual.device.type == 'cpu'
        assert torch.equal(actual, expected)
    assert exported['state_dict']._metadata == state_dict._metadata
    model.load_state_dict(exported['state_dict'], strict=True)
    assert b'fixture://training/' not in output.read_bytes()
    assert source.read_bytes() == original_bytes


def test_refuses_overwriting_source_or_existing_output(tmp_path):
    source = tmp_path / 'training.pth'
    output = tmp_path / 'release.pth'
    torch.save({'weight': torch.ones(1)}, source)
    original_bytes = source.read_bytes()
    with pytest.raises(ValueError, match='different files'):
        export_weights(source, source)
    output.write_bytes(b'existing output')
    with pytest.raises(FileExistsError):
        export_weights(source, output)
    assert source.read_bytes() == original_bytes
    assert output.read_bytes() == b'existing output'


def test_rejects_non_tensor_state_without_creating_output(tmp_path):
    source = tmp_path / 'training.pth'
    output = tmp_path / 'release.pth'
    torch.save({'config': 'fixture://training/data'}, source)
    with pytest.raises(ValueError, match='string keys and tensors'):
        export_weights(source, output)
    assert not output.exists()


def test_command_summary_does_not_expose_paths(tmp_path):
    source = tmp_path / 'private_training.pth'
    output = tmp_path / 'release.pth'
    torch.save({'state_dict': {'weight': torch.ones(3)}}, source)
    script = Path(__file__).parents[1] / 'tools' / 'export_weights.py'
    result = subprocess.run(
        [sys.executable, str(script), str(source), str(output)],
        check=True, capture_output=True, text=True)
    assert 'Exported 1 tensor entries (3 elements).' in result.stdout
    assert 'state_dict' in result.stdout
    assert str(source) not in result.stdout + result.stderr
    assert str(output) not in result.stdout + result.stderr
