#!/usr/bin/env python3
# Author: Andrew Marshall-Lee
"""Named expert-network feature sets used by the ANN/MOE workflow."""

from __future__ import annotations

import argparse
import ast
import hashlib
import re


BASE_FEATURES = [
    "packed_mpc_name",
    "PROPER_SEMIMAJOR_AXIS",
    "PROPER_ECCENTRICITY",
    "SINE_OF_PROPER_INCLINATION",
    "family_id",
]

EXPERT_FEATURES = {
    "base": [],
    "H": ["ABSOLUTE_MAGNITUDE"],
    "pV": ["V_albedo"],
    "H_pV": ["ABSOLUTE_MAGNITUDE", "V_albedo"],
    "A_iz": ["A_MAG", "I_MAG_minus_Z_MAG"],
    "gaia_color": ["gaia_slope", "gaia_z_minus_i"],
    "A_iz_gaia_color": [
        "A_MAG",
        "I_MAG_minus_Z_MAG",
        "gaia_slope",
        "gaia_z_minus_i",
    ],
    "A_iz_pV_gaia_color": [
        "V_albedo",
        "A_MAG",
        "I_MAG_minus_Z_MAG",
        "gaia_slope",
        "gaia_z_minus_i",
    ],
    "H_A_slope": ["ABSOLUTE_MAGNITUDE", "A_MAG", "gaia_slope"],
}


def feature_list_for(expert: str) -> list[str]:
    """Return the full column list for a named expert."""
    if expert not in EXPERT_FEATURES:
        valid = ", ".join(sorted(EXPERT_FEATURES))
        raise ValueError(f"Unknown expert '{expert}'. Valid names: {valid}")
    return BASE_FEATURES + EXPERT_FEATURES[expert]


def expert_for_feature_list(feature_list: list[str]) -> str | None:
    """Return the named expert matching a full feature list, if one exists."""
    feature_tuple = tuple(feature_list)
    for expert in EXPERT_FEATURES:
        if tuple(feature_list_for(expert)) == feature_tuple:
            return expert
    return None


def artifact_tag_for(feature_list: list[str], expert: str | None = None) -> str:
    """Return a stable artifact tag for a named or custom feature set."""
    if expert:
        return expert

    matched = expert_for_feature_list(feature_list)
    if matched:
        return matched

    extras = [col for col in feature_list if col not in BASE_FEATURES]
    label = "_".join(extras) if extras else "base"
    label = re.sub(r"[^A-Za-z0-9_]+", "_", label).strip("_")
    digest = hashlib.sha1(str(feature_list).encode("utf-8")).hexdigest()[:8]
    return f"custom_{label}_{digest}" if label else f"custom_{digest}"


def resolve_feature_list(args: argparse.Namespace) -> list[str]:
    """Resolve either --expert or --features CLI input to a feature list."""
    expert = getattr(args, "expert", None)
    features = getattr(args, "features", None)

    if bool(expert) == bool(features):
        raise ValueError("Provide exactly one of --expert or --features.")
    if expert:
        return feature_list_for(expert)
    return ast.literal_eval(features)


def resolve_feature_selection(args: argparse.Namespace) -> tuple[list[str], str]:
    """Resolve CLI input to a feature list and artifact tag."""
    feature_list = resolve_feature_list(args)
    return feature_list, artifact_tag_for(feature_list, getattr(args, "expert", None))


def add_feature_args(parser: argparse.ArgumentParser) -> None:
    """Add shared --expert/--features CLI arguments."""
    parser.add_argument(
        "--expert",
        choices=sorted(EXPERT_FEATURES),
        help="Named expert feature set.",
    )
    parser.add_argument(
        "--features",
        help="Python literal list of feature columns for a custom expert.",
    )


def main() -> None:
    print("Available experts:\n")
    for name, extras in EXPERT_FEATURES.items():
        shown = ", ".join(extras) if extras else "proper elements only"
        print(f"{name:20s} {shown}")


if __name__ == "__main__":
    main()
