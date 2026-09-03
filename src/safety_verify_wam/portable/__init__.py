"""Host-neutral video/state/action safety sidecar.

This package is deliberately separate from the Efficient-WAM feature-tap
reader. It can be imported by any policy that can expose RGB observations,
time-aligned robot state, and a candidate action chunk.
"""

from .adapters import CanonicalSafetyAdapter, RoboTwinPolicyAdapter
from .bridge import RoboTwinSafetySidecar
from .checkpoint import (
    LoadedPortableSafety,
    load_portable_checkpoint,
    save_portable_checkpoint,
    sha256_file,
)
from .contracts import (
    PORTABLE_CLASS_NAMES,
    PORTABLE_INPUT_SCHEMA,
    RISK_CLASS_INDEX,
    SAFE_CLASS_INDEX,
    AdaptedSafetyInput,
    RobotProfile,
    SafetyBatch,
)
from .data import (
    IGNORE_STEP_LABEL,
    PortableSafetyManifestDataset,
    portable_safety_collate,
)
from .guard import (
    ActionGuardConfig,
    DeterministicActionGuard,
    GuardResult,
)
from .losses import portable_safety_loss
from .model import (
    PortableSafetyConfig,
    PortableSafetyCore,
    trainable_parameter_count,
)
from .multidomain import (
    MULTIDOMAIN_CHECKPOINT_SCHEMA,
    LoadedMultiProfileCheckpoint,
    MultiProfilePortableSafetyCore,
    MultiProfileSafetyConfig,
    ProfileAdapterConfig,
    config_fingerprint,
    initialize_from_single_profile,
    load_multidomain_checkpoint,
    save_multidomain_checkpoint,
    trainable_parameter_count as multidomain_trainable_parameter_count,
)
from .runtime import (
    PortableSafetyRuntime,
    SafetyAssessment,
    SafetyThresholds,
    choose_action,
)

__all__ = [
    "PORTABLE_CLASS_NAMES",
    "PORTABLE_INPUT_SCHEMA",
    "SAFE_CLASS_INDEX",
    "RISK_CLASS_INDEX",
    "RobotProfile",
    "SafetyBatch",
    "AdaptedSafetyInput",
    "IGNORE_STEP_LABEL",
    "PortableSafetyManifestDataset",
    "portable_safety_collate",
    "CanonicalSafetyAdapter",
    "RoboTwinPolicyAdapter",
    "RoboTwinSafetySidecar",
    "PortableSafetyConfig",
    "PortableSafetyCore",
    "trainable_parameter_count",
    "MULTIDOMAIN_CHECKPOINT_SCHEMA",
    "ProfileAdapterConfig",
    "MultiProfileSafetyConfig",
    "MultiProfilePortableSafetyCore",
    "LoadedMultiProfileCheckpoint",
    "initialize_from_single_profile",
    "save_multidomain_checkpoint",
    "load_multidomain_checkpoint",
    "multidomain_trainable_parameter_count",
    "config_fingerprint",
    "ActionGuardConfig",
    "DeterministicActionGuard",
    "GuardResult",
    "SafetyThresholds",
    "SafetyAssessment",
    "PortableSafetyRuntime",
    "choose_action",
    "portable_safety_loss",
    "LoadedPortableSafety",
    "save_portable_checkpoint",
    "load_portable_checkpoint",
    "sha256_file",
]
