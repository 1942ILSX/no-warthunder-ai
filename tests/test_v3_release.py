"""Release provenance and gates; fixtures do not make gameplay quality claims."""
from copy import deepcopy
import json

import pytest

from nowarthunder_ai import cli, evaluation, qualification
from nowarthunder_ai.checkpoint import save_checkpoint
from nowarthunder_ai.export import export_checkpoint
from nowarthunder_ai.progress import progress_config
from test_v2_workflow import source_checkpoint, report_for


def v3_checkpoint(path):
    state = source_checkpoint(path)
    state['config'].update(calibration_version=3, progress_recovery=progress_config())
    save_checkpoint(path, state)
    return state


def v3_report(path):
    report = report_for(path)
    for row in report['records']:
        row.update(enemy_loop_observed_steps=100, enemy_loop_steps=0)
    return {**report, **evaluation.summarize_records(report['records'], version=3),
            'progress_recovery': progress_config()}


def test_v3_export_requires_fresh_matching_report_even_with_allow_untrained(tmp_path):
    checkpoint = tmp_path / 'actor.pt'
    v3_checkpoint(checkpoint)
    destination = tmp_path / 'release'
    with pytest.raises(ValueError, match='Trained v3 export requires --evaluation'):
        export_checkpoint(checkpoint, destination, policy_version='test-v3', allow_untrained=True)
    report_path = tmp_path / 'evaluation.json'
    variants = []
    wrong_version = v3_report(checkpoint)
    wrong_version['metrics_version'] = 2
    variants.append((wrong_version, 'fresh v3'))
    retrospective = v3_report(checkpoint)
    retrospective['retrospective_assessment'] = True
    variants.append((retrospective, 'fresh v3'))
    wrong_detector = v3_report(checkpoint)
    wrong_detector['progress_recovery']['window'] = 121
    variants.append((wrong_detector, 'mismatched_detector_config'))
    missing = v3_report(checkpoint)
    del missing['records'][0]['enemy_loop_steps']
    variants.append((missing, 'missing_moving_no_progress_telemetry'))
    bad_loops = v3_report(checkpoint)
    for row in bad_loops['records']:
        row['enemy_loop_steps'] = 50
    bad_loops['calibration'] = {'eligible': True}
    variants.append((bad_loops, 'quality_limit_exceeded'))
    duplicates = v3_report(checkpoint)
    duplicates['records'][1]['seed'] = duplicates['records'][0]['seed']
    variants.append((duplicates, 'exactly once'))
    for report, error in variants:
        report_path.write_text(json.dumps(report))
        with pytest.raises(ValueError, match=error):
            export_checkpoint(checkpoint, destination, policy_version='test-v3', evaluation=report_path,
                              allow_untrained=True)
        assert not destination.exists()
    report_path.write_text(json.dumps(v3_report(checkpoint)))
    result = export_checkpoint(checkpoint, destination, policy_version='test-v3', evaluation=report_path)
    assert result['weights_bytes'] == 91816
    notes = json.loads((destination / 'deployment-notes.json').read_text())
    assert notes['calibrationVerified']
    assert notes['calibration']['calibration_version'] == 3
    assert notes['checkpointSHA256'] == cli.file_sha256(checkpoint)
    assert notes['evaluationSHA256'] == cli.file_sha256(report_path)


def test_unknown_saved_contract_cannot_silently_bypass_release_gate(tmp_path):
    checkpoint = tmp_path / 'actor.pt'
    state = source_checkpoint(checkpoint)
    state['config']['calibration_version'] = 4
    save_checkpoint(checkpoint, state)
    with pytest.raises(ValueError, match='Unknown checkpoint calibration version'):
        export_checkpoint(checkpoint, tmp_path / 'release', policy_version='invalid')


def test_cli_v3_uses_independent_default_seed_and_saved_detector(tmp_path, monkeypatch):
    checkpoint = tmp_path / 'actor.pt'
    v3_checkpoint(checkpoint)
    seen, reports = [], []
    monkeypatch.setattr(cli, '_factory', lambda args: None)
    monkeypatch.setattr(cli, 'write_report', lambda report, output=None: reports.append(report))
    def fake_evaluate(actor, factory, device, config):
        seen.append(deepcopy(config))
        return {'deterministic_policy': True}
    monkeypatch.setattr(evaluation, 'evaluate', fake_evaluate)
    assert cli.main(['evaluate', '--checkpoint', str(checkpoint), '--calibration-version', '3',
                     '--device', 'cpu']) == 0
    assert seen[0]['calibration_version'] == 3
    assert seen[0]['eval_seed'] == 4_000_000
    assert seen[0]['progress_recovery'] == progress_config()
    assert reports[0]['campaign']['purpose'] == 'final_test'


def test_cli_qualification_calls_explicit_immutable_helper(monkeypatch):
    calls = []
    monkeypatch.setattr(qualification, 'qualify_checkpoint', lambda *args: calls.append(args) or {})
    monkeypatch.setattr(cli, 'write_report', lambda *args: None)
    assert cli.main(['qualify', '--checkpoint', 'source.pt', '--evaluation', 'calibration.json',
                     '--profile', 'expert', '--output', 'new.pt']) == 0
    assert calls == [('source.pt', 'calibration.json', 'new.pt', 'expert')]
