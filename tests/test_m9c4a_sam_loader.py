"""Synthetic provenance and loading checks; no patient data or inference."""
import hashlib
import sys
from types import ModuleType
import pytest
import torch
from src.sam import loader

@pytest.fixture
def setup(tmp_path, monkeypatch):
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout())
    model.forward = lambda *a, **k: pytest.fail('inference forbidden')
    path = tmp_path / 'synthetic.pt'
    torch.save({'model_state_dict': model.state_dict()}, path)
    monkeypatch.setattr(loader, 'FROZEN_TURBO_SHA256', hashlib.sha256(path.read_bytes()).hexdigest())
    monkeypatch.setattr(loader, '_verify_source', lambda root: loader.FROZEN_SAM_COMMIT)
    calls = []
    def build(*, checkpoint):
        calls.append(checkpoint)
        return model
    monkeypatch.setattr(loader, '_import_registry', lambda root: {'vit_b_ori': build})
    return tmp_path, path, model, calls

def test_cpu_strict_frozen_no_inference(setup, monkeypatch):
    root, path, model, calls = setup
    original = torch.load
    def load(stream, **kwargs):
        assert kwargs == dict(map_location='cpu', weights_only=False)
        return original(stream, **kwargs)
    monkeypatch.setattr(torch, 'load', load)
    original_state = model.load_state_dict
    def state_load(state, *, strict):
        assert strict is True
        return original_state(state, strict=strict)
    monkeypatch.setattr(model, 'load_state_dict', state_load)
    result = loader.load_sam_checkpoint(root, path)
    assert result.model is model and calls == [None]
    assert not any(m.training for m in model.modules())
    assert all(not p.requires_grad and p.device.type == 'cpu' for p in model.parameters())

def test_hash_gate(setup, monkeypatch):
    root, path, _, _ = setup
    path.write_bytes(b'bad')
    monkeypatch.setattr(loader, '_import_registry', lambda root: pytest.fail('early import'))
    monkeypatch.setattr(torch, 'load', lambda *a, **k: pytest.fail('early load'))
    with pytest.raises(loader.SAMLoaderIntegrityError, match='SHA-256'):
        loader.load_sam_checkpoint(root, path)

@pytest.mark.parametrize('payload', [None, {}, {'model_state_dict': None}, {'state_dict': {}}])
def test_checkpoint_schema(setup, monkeypatch, payload):
    root, path, _, _ = setup
    monkeypatch.setattr(torch, 'load', lambda *a, **k: payload)
    with pytest.raises(loader.SAMLoaderIntegrityError, match='model_state_dict'):
        loader.load_sam_checkpoint(root, path)

@pytest.mark.parametrize('problem', ['missing', 'extra', 'shape'])
def test_strict_mismatch(setup, monkeypatch, problem):
    root, path, model, _ = setup
    state = dict(model.state_dict())
    if problem == 'missing': state.pop('0.weight')
    elif problem == 'extra': state['extra'] = torch.zeros(1)
    else: state['0.weight'] = torch.zeros(3, 3)
    monkeypatch.setattr(torch, 'load', lambda *a, **k: {'model_state_dict': state})
    with pytest.raises(RuntimeError): loader.load_sam_checkpoint(root, path)

@pytest.mark.parametrize('problem', ['commit', 'dirty', 'root', 'ignored'])
def test_source_gate(tmp_path, monkeypatch, problem):
    def git(root, *args):
        if args == ('rev-parse', '--show-toplevel'):
            return str(tmp_path / 'wrong' if problem == 'root' else tmp_path)
        if args == ('rev-parse', 'HEAD'):
            return '0' * 40 if problem == 'commit' else loader.FROZEN_SAM_COMMIT
        if args[0] == 'status': return ' M source.py' if problem == 'dirty' else ''
        return ''
    monkeypatch.setattr(loader, '_git', git)
    if problem == 'ignored':
        (tmp_path / 'segment_anything').mkdir()
        (tmp_path / 'segment_anything' / 'extra.py').write_text('')
    with pytest.raises(loader.SAMLoaderIntegrityError): loader._verify_source(tmp_path)

def test_import_collision(tmp_path, monkeypatch):
    module = ModuleType('segment_anything')
    module.__file__ = '/elsewhere/__init__.py'
    monkeypatch.setitem(sys.modules, 'segment_anything', module)
    with pytest.raises(loader.SAMLoaderIntegrityError, match='Conflicting'):
        loader._import_registry(tmp_path)

@pytest.mark.parametrize('registry', [{}, {'vit_b': None}, {'vit_b_ori': None}, None])
def test_registry_and_path_restoration(tmp_path, monkeypatch, registry):
    module = ModuleType('segment_anything')
    module.__file__ = str(tmp_path / 'segment_anything/__init__.py')
    module.sam_model_registry3D = registry
    monkeypatch.setattr(loader.importlib, 'import_module', lambda name: module)
    before = sys.path[:]
    with pytest.raises(loader.SAMLoaderIntegrityError, match='vit_b_ori'):
        loader._import_registry(tmp_path)
    assert sys.path == before
