"""Settings: defaults, environment overrides, and the on/off switches."""

import pytest

from src.guardrails import Action, Settings, Source, get_pipeline, reset_pipeline
from src.guardrails.detectors import ClassifierLayer, PatternLayer


class TestDefaults:
    def test_every_source_has_a_threshold_and_an_action(self):
        settings = Settings()
        for source in Source:
            assert 0 < settings.threshold(source) <= 1
            assert isinstance(settings.action(source), Action)

    def test_user_text_is_blocked_and_documents_are_dropped(self):
        settings = Settings()
        assert settings.action(Source.USER_TEXT) is Action.BLOCK
        assert settings.action(Source.ATTACHMENT) is Action.DROP

    def test_user_text_has_the_lowest_threshold(self):
        settings = Settings()
        assert settings.threshold(Source.USER_TEXT) == min(settings.thresholds.values())

    def test_only_user_collections_are_untrusted(self):
        assert Settings().untrusted_collections == {"userdocs", "outline"}

    def test_settings_are_immutable(self):
        with pytest.raises(Exception):
            Settings().enabled = False  # type: ignore[misc]


class TestFromEnv:
    def test_defaults_when_nothing_is_set(self, monkeypatch):
        for name in ("USE_GUARDRAILS", "GUARDRAIL_CLASSIFIER", "GUARDRAIL_MODEL", "GUARDRAIL_DEVICE"):
            monkeypatch.delenv(name, raising=False)
        settings = Settings.from_env()
        assert settings.enabled and settings.classifier_enabled
        assert settings.device is None
        assert settings.model == Settings().model

    @pytest.mark.parametrize("value, expected", [("false", False), ("FALSE", False), ("true", True), ("", True)])
    def test_use_guardrails_switch(self, monkeypatch, value, expected):
        monkeypatch.setenv("USE_GUARDRAILS", value)
        assert Settings.from_env().enabled is expected

    def test_classifier_switch(self, monkeypatch):
        monkeypatch.setenv("GUARDRAIL_CLASSIFIER", "false")
        assert Settings.from_env().classifier_enabled is False

    def test_model_and_device(self, monkeypatch):
        monkeypatch.setenv("GUARDRAIL_MODEL", "someone/else")
        monkeypatch.setenv("GUARDRAIL_DEVICE", "mps")
        settings = Settings.from_env()
        assert (settings.model, settings.device) == ("someone/else", "mps")

    def test_threshold_override_per_source(self, monkeypatch):
        monkeypatch.setenv("GUARDRAIL_THRESHOLD_ATTACHMENT", "0.8")
        settings = Settings.from_env()
        assert settings.threshold(Source.ATTACHMENT) == 0.8
        assert settings.threshold(Source.USER_TEXT) == Settings().threshold(Source.USER_TEXT)

    def test_bad_threshold_is_an_error_not_silently_ignored(self, monkeypatch):
        monkeypatch.setenv("GUARDRAIL_THRESHOLD_USER_TEXT", "high")
        with pytest.raises(ValueError):
            Settings.from_env()


class TestSharedPipeline:
    def test_classifier_layer_follows_the_switch(self, monkeypatch):
        monkeypatch.setenv("GUARDRAIL_CLASSIFIER", "false")
        reset_pipeline()
        assert [type(layer) for layer in get_pipeline().layers] == [PatternLayer]
        monkeypatch.setenv("GUARDRAIL_CLASSIFIER", "true")
        reset_pipeline()
        assert [type(layer) for layer in get_pipeline().layers] == [PatternLayer, ClassifierLayer]

    def test_toggling_the_switch_keeps_the_same_pipeline(self, monkeypatch):
        first = get_pipeline()
        monkeypatch.setenv("USE_GUARDRAILS", "false")
        second = get_pipeline()
        assert second is first and second.enabled is False   # the loaded model survives
        monkeypatch.setenv("USE_GUARDRAILS", "true")
        assert get_pipeline() is first and first.enabled is True

    def test_changing_a_threshold_rebuilds_it(self, monkeypatch):
        first = get_pipeline()
        monkeypatch.setenv("GUARDRAIL_THRESHOLD_USER_TEXT", "0.5")
        second = get_pipeline()
        assert second is not first
        assert second.settings.threshold(Source.USER_TEXT) == 0.5
