from config.fec_defaults import FEC_DEFAULT_ENTITY_TYPES
from config.settings import FecDomainSettings


def test_fec_default_entity_types_count() -> None:
    assert len(FEC_DEFAULT_ENTITY_TYPES) == 12
    assert "coding_scheme" in FEC_DEFAULT_ENTITY_TYPES


def test_fec_domain_resolve_defaults() -> None:
    s = FecDomainSettings()
    types = s.resolve_entity_types()
    assert types == list(FEC_DEFAULT_ENTITY_TYPES)


def test_fec_domain_json_override() -> None:
    s = FecDomainSettings(entity_types_json='["a","b"]')
    assert s.resolve_entity_types() == ["a", "b"]
