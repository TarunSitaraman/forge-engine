"""Concept identity: user decisions about naming, persisted as configuration."""

from .config import (
    DEFAULT_CONFIG_PATH,
    CollisionResolution,
    ConceptIdentity,
    IdentityConfig,
    IdentityConfigError,
)
from .service import IdentityResolution, IdentityService

__all__ = [
    "IdentityConfig",
    "IdentityConfigError",
    "ConceptIdentity",
    "CollisionResolution",
    "IdentityService",
    "IdentityResolution",
    "DEFAULT_CONFIG_PATH",
]
