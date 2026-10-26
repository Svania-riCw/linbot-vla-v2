import json

import yaml

from tools.render_robotwin_inference_config import render


def _checkpoint_fixture(tmp_path, image_size):
    checkpoint = tmp_path / 'experiment' / 'checkpoints' / 'global_step_10000' / 'hf_ckpt'
    checkpoint.mkdir(parents=True)
    (checkpoint / 'config.json').write_text(json.dumps({
        'post_training': True,
        'max_action_dim': 55,
        'max_state_dim': 55,
        'token_moe_layers': [],
    }))
    (checkpoint / 'model.safetensors').touch()

    training_config = {
        'data': {'img_size': image_size},
    }
    (tmp_path / 'experiment' / 'lingbotvla_cli.yaml').write_text(yaml.safe_dump(training_config))

    qwen = tmp_path / 'qwen'
    qwen.mkdir()
    (qwen / 'config.json').write_text('{}')
    norm_stats = tmp_path / 'norm_stats.json'
    norm_stats.write_text('{}')
    return checkpoint, qwen, norm_stats


def test_render_inherits_checkpoint_image_size(tmp_path, capsys):
    checkpoint, qwen, norm_stats = _checkpoint_fixture(tmp_path, 384)
    output = tmp_path / 'inference.yaml'

    render(checkpoint, qwen, output, norm_stats=norm_stats)

    rendered = yaml.safe_load(output.read_text())
    assert rendered['data']['img_size'] == 384
    assert rendered['data']['cameras'] == [
        'camera_top', 'camera_wrist_left', 'camera_wrist_right'
    ]
    assert 'Evaluation image input resolution: 384x384' in capsys.readouterr().err


def test_explicit_data_config_overrides_checkpoint_image_size(tmp_path):
    checkpoint, qwen, norm_stats = _checkpoint_fixture(tmp_path, 384)
    data_config = tmp_path / 'custom_data.yaml'
    data_config.write_text(yaml.safe_dump({
        'data': {'img_size': 320, 'cameras': ['camera_top']},
    }))
    output = tmp_path / 'inference.yaml'

    render(checkpoint, qwen, output, norm_stats=norm_stats, data_config=data_config)

    rendered = yaml.safe_load(output.read_text())
    assert rendered['data']['img_size'] == 320
    assert rendered['data']['cameras'] == ['camera_top']


def test_explicit_image_size_overrides_all_config_sources(tmp_path, capsys):
    checkpoint, qwen, norm_stats = _checkpoint_fixture(tmp_path, 256)
    data_config = tmp_path / 'custom_data.yaml'
    data_config.write_text(yaml.safe_dump({
        'data': {'img_size': 320, 'cameras': ['camera_top']},
    }))
    output = tmp_path / 'inference.yaml'

    render(
        checkpoint, qwen, output, norm_stats=norm_stats,
        data_config=data_config, image_size=384,
    )

    rendered = yaml.safe_load(output.read_text())
    assert rendered['data']['img_size'] == 384
    assert 'Evaluation image input resolution: 384x384' in capsys.readouterr().err
