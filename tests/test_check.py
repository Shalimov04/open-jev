"""`openjev check` — the newcomer dry run. No teacher, no network (the /models ping is allowed to fail)."""
import json

import pytest

from openjev import check

YAML = """name: t
type: choice
question: "Which is it?"
options: [good, bad]
data:
  source: {{csv: {csv}}}
  text: "{{text}}"
  max_chars: 10
  gold: label
  train: {{split: train, n: 4}}
  calib: {{split: train, n: 4}}
  eval: {{split: train, n: 4}}
student: {{model: no/such-model}}
"""


@pytest.fixture
def task_yaml(tmp_path):
    csv = tmp_path / "d.csv"
    csv.write_text("text,label\n" + "".join(f"row {i} is quite long,{'good' if i % 2 else 'bad'}\n"
                                            for i in range(12)))
    p = tmp_path / "t.yaml"
    p.write_text(YAML.format(csv=csv))
    return p


def test_check_reports_splits_prompt_and_truncation(task_yaml, tmp_path, capsys):
    check.report(task_yaml, runs=str(tmp_path / "runs"))
    out = capsys.readouterr().out
    assert out.count("    4 rows") == 3                                   # train / calib / eval sizes
    assert "good 50.0%" in out
    assert "A: good\nB: bad\nAnswer with a single letter." in out          # the prompt, verbatim
    assert out.count("[train:") == 3                                      # 3 formatted examples
    assert "max_chars=10 binds on 100% of rows" in out
    assert "WARNING max_chars=10 truncates 100%" in out
    assert "teacher cost: 12 examples x 1 call" in out


def test_check_bad_source_is_one_line_and_exit_1(tmp_path):
    p = tmp_path / "t.yaml"
    p.write_text(YAML.format(csv=tmp_path / "missing.csv"))
    with pytest.raises(SystemExit) as e:
        check.main(p, runs=str(tmp_path / "runs"))
    assert "missing.csv" in str(e.value) and "hint:" in str(e.value)


def test_probe_warns_when_a_class_is_under_predicted(task_yaml, tmp_path, capsys):
    """The kinopoisk case: gold is 50/50, the teacher answers `good` almost every time."""
    from openjev.data import examples
    from openjev.spec import load_task
    task = load_task(task_yaml)
    run_dir = tmp_path / "runs" / "t"
    run_dir.mkdir(parents=True)
    exs = examples(task)
    with open(run_dir / "teacher.jsonl", "w") as f:   # pre-cached: probe makes no teacher call
        for i, e in enumerate(e for e in exs if e["split"] == "calib"):
            f.write(json.dumps({**e, "probs": [0.9, 0.1] if i else [0.2, 0.8], "raw": {}}) + "\n")
    check.probe(task, run_dir, 4, exs)
    out = capsys.readouterr().out
    assert "predicted marginal: good 75.0%  bad 25.0%" in out
    assert "mean max-p: 0.8" in out
    assert "WARNING the teacher under-predicts 'bad' (25% vs 50%)" in out
