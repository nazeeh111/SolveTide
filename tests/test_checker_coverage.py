"""Every reported sensitivity must be covered by the detached checker."""
import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from solvetide import Runtime, SolverConfig, assemble
from solvetide.artifacts import bundle
from solvetide.checker import digest, verify


@pytest.fixture
def real_artifact():
    system = assemble("heat", 6, {"contrast": 2., "boundary_value": .1})
    runtime = Runtime(SolverConfig(strategy="direct"))
    forward = runtime.solve(system)
    gradient = runtime.gradient(system, forward, target=.2)
    assert forward.accepted and gradient.accepted
    return bundle(system, forward, gradient, runtime.config, target=.2)


def seal(document):
    document["content_sha256"] = digest(document)
    return document


@pytest.mark.parametrize("mutation", [
    "extra-gradient", "missing-gradient", "extra-derivative",
    "empty-gradient", "empty-derivatives",
])
def test_reported_and_checked_derivatives_must_match_exactly(real_artifact, mutation):
    document = copy.deepcopy(real_artifact)
    name = next(iter(document["gradient"]))
    if mutation == "extra-gradient":
        document["gradient"]["never_checked"] = 123456789.
    elif mutation == "missing-gradient":
        document["gradient"].pop(name)
    elif mutation == "extra-derivative":
        document["equations"]["derivatives"]["never_reported"] = copy.deepcopy(
            document["equations"]["derivatives"][name])
    elif mutation == "empty-gradient":
        document["gradient"] = {}
    else:
        document["equations"]["derivatives"] = {}
    with pytest.raises(ValueError, match="keys"):
        verify(seal(document))


@pytest.mark.parametrize("field", ["gradient", "derivatives"])
@pytest.mark.parametrize("value", [None, [], "not a map"])
def test_derivative_maps_are_required(real_artifact, field, value):
    if field == "gradient":
        real_artifact[field] = value
    else:
        real_artifact["equations"][field] = value
    with pytest.raises(ValueError, match="map"):
        verify(seal(real_artifact))


@pytest.mark.parametrize("value", [True, "1", [1], None])
def test_each_reported_gradient_is_a_finite_real_number(real_artifact, value):
    name = next(iter(real_artifact["gradient"]))
    real_artifact["gradient"][name] = value
    with pytest.raises(ValueError, match="finite"):
        verify(seal(real_artifact))


def test_matching_real_forward_adjoint_and_known_negative_control(real_artifact):
    assert verify(real_artifact)["status"] == "checked"
    name = next(iter(real_artifact["gradient"]))
    real_artifact["gradient"][name] += 1.
    assert verify(seal(real_artifact))["status"] == "rejected"


def test_matching_empty_roster_does_not_invent_a_sensitivity(real_artifact):
    real_artifact["gradient"] = {}
    real_artifact["equations"]["derivatives"] = {}
    assert verify(seal(real_artifact))["status"] == "checked"


def test_standalone_checker_needs_no_scientific_runtime(real_artifact, tmp_path):
    checker = Path(__file__).resolve().parents[1] / "solvetide/checker.py"
    artifact = tmp_path / "real.json"
    artifact.write_text(json.dumps(real_artifact, allow_nan=False))
    command = [sys.executable, "-B", "-S", str(checker), str(artifact)]
    valid = subprocess.run(command, capture_output=True, text=True, timeout=15)
    assert valid.returncode == 0, valid.stderr
    assert json.loads(valid.stdout)["status"] == "checked"
    real_artifact["gradient"]["unverified"] = 77.
    artifact.write_text(json.dumps(seal(real_artifact), allow_nan=False))
    invalid = subprocess.run(command, capture_output=True, text=True, timeout=15)
    assert invalid.returncode == 2
    assert json.loads(invalid.stdout)["status"] == "invalid"
    assert "keys" in invalid.stdout and "Traceback" not in invalid.stderr
