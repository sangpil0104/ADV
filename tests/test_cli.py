"""Exercise the user-facing count argument without loading GPU backends."""

import signal

import pytest
import yaml

from advv import cli
from advv.reporting import records_for
from advv.storage import read_json
from conftest import FakeBackend


@pytest.fixture(autouse=True)
def restore_sigterm_handler():
    original = signal.getsignal(signal.SIGTERM)
    yield
    signal.signal(signal.SIGTERM, original)


@pytest.mark.parametrize("quantity, target", [(["3"], 3), (["--target-count", "2"], 2), ([], 2)])
def test_cli_total_count_and_resume(cfg, tmp_path, monkeypatch, quantity, target):
    cfg["run"]["target_count"] = 2  # Positional and flag forms override the YAML value.
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg))
    instances = []

    class Backend(FakeBackend):
        def __init__(self, resolved, root):
            count = resolved["run"]["target_count"]
            # One NO and one pixel-identical source must not consume the accepted quota.
            super().__init__(
                resolved,
                [False] + [True] * (count + 1),
                [(10, 20, 30), "source"] + [(180, i + 1, 60) for i in range(count)],
            )
            instances.append(self)

    monkeypatch.setattr(cli, "LocalBackend", Backend)
    monkeypatch.setattr(cli, "preflight", lambda *args, **kwargs: {"status": "ready"})
    assert (
        cli.main(
            [
                "run",
                *quantity,
                "--config",
                str(config_path),
                "--gpus",
                "0,1",
                "--run-id",
                "cli_count",
                "--prompt",
                "Keep the accident and vary the vehicle pose.",
            ]
        )
        == 0
    )
    root = tmp_path / "runs/cli_count"
    state = read_json(root / "state.json")
    assert state["status"] == "completed"
    assert state["accepted_count"] == target
    assert state["attempts"] == target + 2
    assert read_json(root / "config.json")["run"]["target_count"] == target
    assert len(list((root / "accepted").glob("*.png"))) == target
    assert all(r["backend"] == "fake" for r in records_for(root))
    assert "vary the vehicle pose" in instances[0].generations[0]["target_prompt"]
    assert (
        read_json(root / "config.json")["generation"]["augmentation_prompt"]
        == "Keep the accident and vary the vehicle pose."
    )

    assert cli.main(["run", "--resume", "--run-dir", str(root), "--gpus", "2,3"]) == 0
    assert read_json(root / "state.json")["accepted_count"] == target
    assert instances[-1].generations == [] and instances[-1].calls == []


@pytest.mark.parametrize(
    "quantity, message",
    [
        (["0"], "positive integer"),
        (["-1"], "positive integer"),
        (["1.5"], "positive integer"),
        (["abc"], "positive integer"),
        (["--target-count", "0"], "positive integer"),
        (["--target-count", "-1"], "positive integer"),
        (["--target-count", "1.5"], "positive integer"),
        (["2", "--target-count", "3"], "not both"),
        (["2", "--target-count", "2"], "not both"),
        (["2", "--resume"], "saved target count"),
        (["--target-count", "2", "--resume"], "saved target count"),
        (["--prompt", " "], "nonempty"),
        (["--resume", "--prompt", "new instructions"], "saved prompt"),
        (["--resume", "--image", "image1.png"], "saved image selection"),
    ],
)
def test_bad_count_fails_before_loading_resources(monkeypatch, capsys, quantity, message):
    def forbidden(*args, **kwargs):
        pytest.fail("Invalid input must be rejected before config/model/run access")

    for name in ("load_config", "read_json", "preflight", "create_run", "LocalBackend"):
        monkeypatch.setattr(cli, name, forbidden)
    with pytest.raises(SystemExit) as exc:
        cli.main(["run", *quantity, "--gpus", "0,1"])
    assert exc.value.code == 2
    assert message in capsys.readouterr().err


def test_default_config_from_other_directory_and_automatic_run_id(cfg, tmp_path, monkeypatch):
    cfg["run"]["output_root"] = str(tmp_path / "assets/outputs")
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    monkeypatch.setattr(cli, "DEFAULT_CONFIG_PATH", path)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    instances = []

    class Backend(FakeBackend):
        def __init__(self, resolved, root):
            super().__init__(resolved)
            instances.append(self)

        def generate(self, source, plan, root):
            self.colors = [(plan.attempt_index + 1, 30, 50)]
            return super().generate(source, plan, root)

    monkeypatch.setattr(cli, "LocalBackend", Backend)
    monkeypatch.setattr(cli, "preflight", lambda *a, **kw: {"status": "ready"})
    assert cli.main(["run", "4", "--gpus", "4,5,6,7"]) == 0
    roots = list((tmp_path / "assets/outputs").iterdir())
    assert len(roots) == 1 and roots[0].name.startswith("run_")
    assert read_json(roots[0] / "state.json")["accepted_count"] == 4
    assert [b.cfg["execution"]["selected_gpu_ids"] for b in instances] == [["4", "5"], ["6", "7"]]
    assert read_json(roots[0] / "config.json")["dataset"]["input_dir"] == str(tmp_path / "input")


def test_input_dir_cli_option_has_been_removed(monkeypatch, capsys):
    monkeypatch.setattr(cli, "require_advv", lambda: pytest.fail("Parsed obsolete input-dir"))
    with pytest.raises(SystemExit) as exc:
        cli.main(["run", "10", "--gpus", "4,5,6,7", "--input-dir", "assets"])
    assert exc.value.code == 2
    assert "unrecognized arguments: --input-dir" in capsys.readouterr().err
