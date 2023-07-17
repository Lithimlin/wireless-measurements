import pytest

from wifi_info import requirements


@pytest.fixture()
def modify_required(monkeypatch, request):
    required = request.param
    if not required:
        raise ValueError(f"No parameter for '{__name__}'")

    monkeypatch.setattr(requirements.Applications, "REQUIRED", required)


@pytest.fixture()
def modify_recommended(monkeypatch, request):
    recommended = request.param
    if not recommended:
        raise ValueError(f"No parameter for '{__name__}'")

    monkeypatch.setattr(requirements.Applications, "RECOMMENDED", recommended)


@pytest.mark.parametrize(
    "app_name, expected",
    [
        ("python3", True),
        ("__ownwapvenaweit__", False),
    ],
)
def test_check_application(app_name, expected):
    assert requirements.check_application(app_name) == expected


@pytest.mark.parametrize("modify_required", [["python3"]], indirect=True)
def test_check_required_applications(modify_required):
    assert requirements.check_required_applications() == True


@pytest.mark.parametrize("modify_required", [["__ownwapvenaweit__"]], indirect=True)
def test_check_required_applications_nonexistent(modify_required):
    with pytest.raises(requirements.ApplicationNotInstalledException):
        requirements.check_required_applications()


@pytest.mark.parametrize(
    "modify_recommended", [{"python3": "I'm using it to run this"}], indirect=True
)
def test_check_recommended_applications(modify_recommended):
    assert requirements.check_recommended_applications() == True


@pytest.mark.parametrize(
    "modify_recommended", [{"__ownwapvenaweit__": "I'm testing"}], indirect=True
)
def test_check_recommended_applications_nonexistent(modify_recommended):
    with pytest.warns(requirements.ApplicationNotInstalledWarning):
        assert requirements.check_recommended_applications() == False


@pytest.mark.parametrize(
    "modify_required, modify_recommended",
    [(["python3"], {"__ownwapvenaweit__": "I said so"})],
    indirect=True,
)
def test_run_checks(modify_required, modify_recommended):
    with pytest.warns(requirements.ApplicationNotInstalledWarning):
        requirements.run_checks()


@pytest.mark.parametrize("modify_required", [["__ownwapvenaweit__"]], indirect=True)
def test_run_checks_exception(modify_required):
    with pytest.raises(requirements.ApplicationNotInstalledException):
        requirements.run_checks()
