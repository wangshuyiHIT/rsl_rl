"""Synthetic run fixtures: no simulator, published policy, or real training run."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest
import torch
from tensordict import TensorDict
import yaml

from rsl_rl.modules import ActorCritic, ActorCriticFiLMCritic

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "export_paper_policy.py"
SPEC = importlib.util.spec_from_file_location("scaler_paper_policy_export", SCRIPT)
exporter = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = exporter
SPEC.loader.exec_module(exporter)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def synthetic_run(tmp_path, baseline="FiLM-C", normalized=False, state_dependent_std=False):
    run = tmp_path / baseline
    params = run / "params"
    provenance = run / "paper_provenance"
    params.mkdir(parents=True)
    provenance.mkdir()
    frame = exporter.FRAME_DIMS[baseline]
    obs = TensorDict({"policy": torch.randn(3, frame * 10), "critic": torch.randn(3, 1490)}, batch_size=[3])
    policy_config = {
        "class_name": "rsl_rl.modules:ActorCriticFiLMCritic" if baseline == "FiLM-C" else "rsl_rl.modules:ActorCritic",
        "actor_hidden_dims": [16, 12, 8],
        "critic_hidden_dims": [16, 12, 8],
        "activation": "elu",
        "actor_obs_normalization": normalized,
        "critic_obs_normalization": normalized,
        "state_dependent_std": state_dependent_std,
        "noise_std_type": "log",
    }
    if baseline == "FiLM-C":
        policy_config["film_context_hidden_dims"] = [64]
    model_class = ActorCriticFiLMCritic if baseline == "FiLM-C" else ActorCritic
    kwargs = {key: value for key, value in policy_config.items() if key != "class_name"}
    policy = model_class(obs, exporter.GROUPS, 24, **kwargs)
    policy.update_normalization(obs)
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.002)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        (policy.evaluate(obs).square().mean() + policy.act_inference(obs).square().mean()).backward()
        optimizer.step()
    policy.eval()
    checkpoint = run / "model_7.pt"
    torch.save({"model_state_dict": policy.state_dict(), "iter": 7, "infos": None}, checkpoint)
    agent_path = params / "agent.yaml"
    agent_path.write_text(yaml.safe_dump({"policy": policy_config, "obs_groups": exporter.GROUPS}))
    sources = []
    for relative_file in [
        "rsl_rl/modules/actor_critic.py",
        "rsl_rl/modules/actor_critic_film.py",
        "rsl_rl/networks/mlp.py",
        "rsl_rl/networks/normalization.py",
        "rsl_rl/utils/utils.py",
    ]:
        active = exporter.TRAIN_ROOT / "rsl_rl" / relative_file
        relative = "modules/scaler_train/rsl_rl/" + relative_file
        archived = provenance / "source" / relative
        archived.parent.mkdir(parents=True, exist_ok=True)
        archived.write_bytes(active.read_bytes())
        sources.append({"path": relative, "sha256": digest(active)})
    manifest_path = provenance / "source_manifest.json"
    manifest_path.write_text(json.dumps({"protocol": exporter.PROTOCOL, "sources": sources}))
    schema = {
        "version": exporter.OBSERVATION_VERSION, "baseline": baseline,
        "actor_frame_dim": frame, "critic_frame_dim": 149,
        "actor_history_length": 10, "critic_history_length": 10,
        "critic_morphology_slice": [81, 86],
    }
    contract = {
        "protocol": exporter.PROTOCOL, "baseline": baseline, "observation_contract": schema,
        "configs": {"params/agent.yaml": digest(agent_path)},
        "source_manifest_sha256": digest(manifest_path),
        "parameters": {
            "actor": sum(p.numel() for p in policy.actor.parameters()),
            "critic": sum(p.numel() for p in policy.critic.parameters()),
            "total": sum(p.numel() for p in policy.parameters()),
        },
    }
    (provenance / "run_contract.json").write_text(json.dumps(contract))
    (provenance / "completion.json").write_text(json.dumps({
        "checkpoints": [{"path": checkpoint.name, "sha256": digest(checkpoint)}],
        "runner_learn_returned": True,
    }))
    return checkpoint, policy, obs


@pytest.mark.parametrize("baseline", ["Hist", "H", "U5", "FiLM-C", "Shuf"])
def test_export_all_five_real_architectures_from_synthetic_runs(tmp_path, baseline):
    checkpoint, policy, obs = synthetic_run(tmp_path, baseline)
    record = exporter.export_policy(checkpoint)
    output = Path(record["output"]["path"])
    assert output == checkpoint.parent / "exported_paper" / f"model_7_{baseline}.pt"
    exported = torch.jit.load(str(output), map_location="cpu")
    torch.testing.assert_close(exported(obs["policy"]), policy.act_inference(obs), rtol=0, atol=0)
    assert all(not key.startswith("critic") for key in exported.state_dict())
    assert record["input"]["shape"] == ["batch", exporter.FRAME_DIMS[baseline] * 10]
    assert record["output"]["shape"] == ["batch", 24]
    assert record["input"]["layout"] == "frame-major"
    assert record["input"]["history_order"].startswith("oldest_to_newest")
    assert record["input"]["empirical_normalization_included"] is False
    assert record["input"]["normalizer"] == "Identity"
    assert record["checkpoint"]["sha256"] == digest(checkpoint)
    assert record["output"]["sha256"] == digest(output)
    assert record["checkpoint_completion_binding"] == "verified"
    assert "No closed-loop evaluation" in record["limitations"]
    assert json.loads(output.with_suffix(".json").read_text()) == record
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        exporter.export_policy(checkpoint)


@pytest.mark.parametrize("baseline", ["U5", "FiLM-C"])
def test_export_honors_actual_normalization_and_state_dependent_actor(tmp_path, baseline):
    checkpoint, policy, obs = synthetic_run(tmp_path, baseline, normalized=True, state_dependent_std=True)
    record = exporter.export_policy(checkpoint)
    exported = torch.jit.load(record["output"]["path"], map_location="cpu")
    assert record["input"]["empirical_normalization_included"] is True
    assert record["input"]["normalizer"] == "EmpiricalNormalization"
    torch.testing.assert_close(exported(obs["policy"]), policy.act_inference(obs), rtol=0, atol=0)
    with pytest.raises(torch.jit.Error, match="declared input width"):
        exported(torch.zeros(1, 810))


def test_export_requires_same_run_contract(tmp_path):
    checkpoint, _, _ = synthetic_run(tmp_path)
    (checkpoint.parent / "paper_provenance" / "run_contract.json").unlink()
    with pytest.raises(FileNotFoundError):
        exporter.export_policy(checkpoint)
    assert not (checkpoint.parent / "exported_paper").exists()


@pytest.mark.parametrize("tamper", ["agent", "source", "completion", "schema", "class", "parameters"])
def test_export_rejects_inconsistent_provenance(tmp_path, tamper):
    checkpoint, _, _ = synthetic_run(tmp_path)
    provenance = checkpoint.parent / "paper_provenance"
    contract_path = provenance / "run_contract.json"
    contract = json.loads(contract_path.read_text())
    if tamper == "agent":
        with (checkpoint.parent / "params" / "agent.yaml").open("a") as stream:
            stream.write("\n# changed after training\n")
    elif tamper == "source":
        archived = provenance / "source/modules/scaler_train/rsl_rl/rsl_rl/modules/actor_critic_film.py"
        with archived.open("a") as stream:
            stream.write("\n# changed snapshot\n")
    elif tamper == "completion":
        (provenance / "completion.json").write_text(json.dumps({"checkpoints": []}))
    elif tamper == "schema":
        contract["observation_contract"]["actor_frame_dim"] = 81
        contract_path.write_text(json.dumps(contract))
    elif tamper == "class":
        path = checkpoint.parent / "params" / "agent.yaml"
        config = yaml.safe_load(path.read_text())
        config["policy"]["class_name"] = "ActorCritic"
        path.write_text(yaml.safe_dump(config))
        contract["configs"]["params/agent.yaml"] = digest(path)
        contract_path.write_text(json.dumps(contract))
    else:
        contract["parameters"]["total"] += 1
        contract_path.write_text(json.dumps(contract))
    with pytest.raises(ValueError):
        exporter.export_policy(checkpoint)
    assert not (checkpoint.parent / "exported_paper").exists()


def test_film_export_does_not_accept_plain_critic_state(tmp_path):
    checkpoint, _, obs = synthetic_run(tmp_path)
    plain = ActorCritic(obs, exporter.GROUPS, 24, actor_hidden_dims=[16, 12, 8], critic_hidden_dims=[16, 12, 8],
                        noise_std_type="log")
    torch.save({"model_state_dict": plain.state_dict(), "iter": 7}, checkpoint)
    completion = checkpoint.parent / "paper_provenance" / "completion.json"
    completion.write_text(json.dumps({"checkpoints": [{"path": checkpoint.name, "sha256": digest(checkpoint)}]}))
    with pytest.raises(RuntimeError, match="Missing key"):
        exporter.export_policy(checkpoint)
    assert not (checkpoint.parent / "exported_paper").exists()


def test_absent_completion_is_recorded_without_claiming_finished_training(tmp_path):
    checkpoint, _, _ = synthetic_run(tmp_path, "Hist")
    (checkpoint.parent / "paper_provenance" / "completion.json").unlink()
    record = exporter.export_policy(checkpoint)
    assert record["completion_manifest_sha256"] is None
    assert "no completed-run claim" in record["checkpoint_completion_binding"]


def test_publication_directory_is_never_written(tmp_path):
    checkpoint, _, _ = synthetic_run(tmp_path / "policies", "U5")
    with pytest.raises(ValueError, match="publication directory"):
        exporter.export_policy(checkpoint)
    assert not (checkpoint.parent / "exported_paper").exists()


def test_cli_has_no_automatic_checkpoint_selection(monkeypatch):
    monkeypatch.setattr(sys, "argv", [str(SCRIPT)])
    with pytest.raises(SystemExit) as raised:
        exporter.main()
    assert raised.value.code == 2


def make_baked_fixture(checkpoint):
    provenance = checkpoint.parent / 'paper_provenance'
    snapshot = provenance / 'assets/robot.usd'
    snapshot.parent.mkdir()
    snapshot.write_bytes(b'frozen USD fixture')
    asset_path = provenance / 'asset_manifest.json'
    asset_path.write_text(json.dumps({'unresolved': [], 'files': [
        {'snapshot': 'assets/robot.usd', 'sha256': digest(snapshot)}]}))
    source_path = provenance / 'source_manifest.json'
    source = json.loads(source_path.read_text())
    source.update(protocol=exporter.BAKED_PROTOCOL, asset_manifest_sha256=digest(asset_path))
    source_path.write_text(json.dumps(source))
    contract_path = provenance / 'run_contract.json'
    contract = json.loads(contract_path.read_text())
    contract.update(protocol=exporter.BAKED_PROTOCOL, physical_model='baked',
                    baked_assignment={'rule': 'verified_round_robin'},
                    asset_manifest_sha256=digest(asset_path), source_manifest_sha256=digest(source_path))
    contract_path.write_text(json.dumps(contract))
    return provenance, contract


def test_baked_export_preserves_protocol_and_bound_assets(tmp_path):
    checkpoint, policy, obs = synthetic_run(tmp_path, 'H')
    provenance, contract = make_baked_fixture(checkpoint)
    record = exporter.export_policy(checkpoint)
    assert record['protocol'] == exporter.BAKED_PROTOCOL and record['physical_model'] == 'baked'
    assert record['asset_manifest_sha256'] == digest(provenance/'asset_manifest.json')
    output = torch.jit.load(record['output']['path'])
    torch.testing.assert_close(output(obs['policy']), policy.act_inference(obs), rtol=0, atol=0)


@pytest.mark.parametrize('confound', ['asset_bytes', 'source_protocol', 'physical_model', 'assignment'])
def test_baked_export_rejects_tampered_or_mixed_physics_provenance(tmp_path, confound):
    checkpoint, _, _ = synthetic_run(tmp_path, 'U5')
    provenance, contract = make_baked_fixture(checkpoint)
    if confound == 'asset_bytes':
        (provenance/'assets/robot.usd').write_bytes(b'changed geometry')
    elif confound == 'source_protocol':
        source_path = provenance/'source_manifest.json'
        source = json.loads(source_path.read_text()); source['protocol'] = exporter.PROTOCOL
        source_path.write_text(json.dumps(source)); contract['source_manifest_sha256'] = digest(source_path)
    elif confound == 'physical_model':
        contract['physical_model'] = 'carrier'
    else:
        contract['baked_assignment'] = None
    (provenance/'run_contract.json').write_text(json.dumps(contract))
    with pytest.raises(ValueError):
        exporter.export_policy(checkpoint)
    assert not (checkpoint.parent/'exported_paper').exists()
