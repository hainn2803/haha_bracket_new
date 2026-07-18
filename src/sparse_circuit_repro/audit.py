from __future__ import annotations

from typing import Any, Mapping

from .common import candidate_ids, load_json, release_root, sha256_file, verify_manifest


QUOTE_CSV_SHA256 = "c38db7b63313960577c6b214f3bdb8979d126532b7afe5ee1f853f6cdf2ae01a"
BRACKET_CSV_SHA256 = "4379a582f1d57051e5e8ebbf7e84252c738bd6708da4646e08f7e23d967be547"


def _row(payload: Mapping[str, Any], config_id: str) -> Mapping[str, Any]:
    matches = [row for row in payload["frozen_handle_rows"] if row["config_id"] == config_id]
    if len(matches) != 1:
        raise AssertionError(f"expected exactly one frozen-handle row for {config_id}")
    return matches[0]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def audit() -> dict[str, Any]:
    root = release_root()
    expected = root / "artifacts" / "expected"
    integrity = verify_manifest(root)

    quote_csv = root / "data" / "quote_circuit_nodes.csv"
    bracket_csv = root / "data" / "bracket_circuit_nodes.csv"
    quote_sites = candidate_ids(quote_csv)
    bracket_sites = candidate_ids(bracket_csv)
    _require(len(quote_sites) == 64, "quote candidate universe is not the full 64-site export")
    _require(len(bracket_sites) == 133, "bracket candidate universe is not the full 133-site export")
    _require(sha256_file(quote_csv) == QUOTE_CSV_SHA256, "quote candidate CSV checksum changed")
    _require(sha256_file(bracket_csv) == BRACKET_CSV_SHA256, "bracket candidate CSV checksum changed")

    quote_necessity = load_json(expected / "necessity_quote.json")
    bracket_necessity = load_json(expected / "necessity_bracket.json")
    quote_u = _row(quote_necessity, "handle::quote_U::global")
    bracket_r = _row(bracket_necessity, "handle::R_mid::global")
    _require(float(quote_u["ablated_contrast_accuracy"]) < 0.60, "quote necessity damage no longer reproduces")
    _require(quote_u["singleton_ranks"]["accuracy"] == 1, "quote handle is no longer accuracy-damage rank 1")
    _require(quote_u["singleton_ranks"]["margin"] == 1, "quote handle is no longer margin-damage rank 1")
    _require(float(bracket_r["ablated_contrast_accuracy"]) < 0.65, "bracket R necessity damage no longer reproduces")
    _require(bracket_r["singleton_ranks"]["margin"] == 1, "bracket R is no longer margin-damage rank 1")
    _require(
        quote_necessity["circuit_only_ablation_status"] == "invalid_baseline_ablation_skipped",
        "quote circuit-only baseline unexpectedly marked valid",
    )
    _require(
        bracket_necessity["circuit_only_ablation_status"] == "invalid_baseline_ablation_skipped",
        "bracket circuit-only baseline unexpectedly marked valid",
    )

    quote_rediscovery = load_json(expected / "rediscover_quote.json")
    bracket_rediscovery = load_json(expected / "rediscover_bracket.json")
    _require(not bool(quote_rediscovery["rounds"][0]["redundancy_certified"]), "quote redundancy status changed")
    _require(not bool(bracket_rediscovery["rounds"][0]["redundancy_certified"]), "bracket redundancy status changed")

    failed_augmented = load_json(expected / "failed_progressive_augmented.json")
    failed_raw = load_json(expected / "failed_progressive_raw_rmid.json")
    surface_density = load_json(expected / "rejected_surface_density.json")
    active_density = load_json(expected / "rejected_active_density.json")
    _require(not bool(failed_augmented["R_early_accepted"]), "historical augmented signature is no longer a negative result")
    _require(not bool(failed_raw["R_early_accepted"]), "historical raw-Rmid signature is no longer a negative result")
    _require(float(surface_density["decoder_metrics"]["Dte"]["pearson"]) < 0.70, "surface-density pilot status changed")
    _require(float(active_density["decoder_metrics"]["Dte"]["pearson"]) < 0.90, "active-density pilot status changed")

    progressive = load_json(expected / "progressive_rmid.json")
    progressive_manifest = load_json(expected / "progressive_rmid_manifest.json")
    progressive_best = progressive["calibration"]["best"]
    _require(progressive_best["site_ids"] == ["4.attn.act_in:1249"], "progressive downstream-depth handle changed")
    _require(bool(progressive["R_early_accepted"]), "progressive upstream handle no longer passes")
    _require(bool(progressive["heldout"]["summary"]["all_required_rates_at_least_0_90"]), "progressive heldout gates changed")
    _require(bool(progressive["heldout"]["mediation"]["passes"]), "progressive mediation changed")
    _require(progressive_manifest["candidate_count_audit"] == 133, "progressive audit does not contain all 133 sites")
    _require(progressive_manifest["candidate_count_primary"] == 132, "progressive primary universe changed")
    _require(bool(progressive_manifest["bank"]["content_splits_disjoint"]), "progressive content splits overlap")
    _require(not bool(progressive_manifest["known_sites_used_for_selection"]), "known sites leaked into progressive selection")
    _require(progressive_manifest["Dte_used_for"] == "final heldout and mediation only", "progressive Dte policy changed")

    graded = load_json(expected / "graded_depth.json")
    graded_manifest = load_json(expected / "graded_depth_manifest.json")
    upstream_best = graded["upstream"]["calibration"]["best"]
    expected_pair = ["2.attn.resid_delta:1249", "3.attn.resid_delta:1249"]
    _require(bool(graded["model_accepted"]), "X -> D -> R -> Y is no longer accepted")
    _require(graded_manifest["e_definition"] == "active_depth", "graded variable is no longer active depth")
    _require(graded_manifest["candidate_count_audit"] == 133, "graded audit does not contain all 133 sites")
    _require(graded_manifest["candidate_count_primary"] == 132, "graded primary universe changed")
    _require(bool(graded_manifest["bank"]["content_splits_disjoint"]), "graded content splits overlap")
    _require(not bool(graded_manifest["known_published_site_used_for_selection"]), "published site leaked into graded selection")
    _require(graded_manifest["Dte_used_for"] == "heldout validation and mediation only", "graded Dte policy changed")
    _require(upstream_best["site_ids"] == expected_pair, "strict upstream depth handle changed")
    _require(bool(graded["direct_E_handle"]["summary"]["passes"]), "direct depth handle no longer passes")
    _require(bool(graded["direct_E_handle"]["mediation"]["passes"]), "direct depth mediation no longer passes")
    _require(bool(graded["upstream"]["heldout"]["summary"]["passes"]), "upstream depth handle no longer passes")
    _require(bool(graded["upstream"]["heldout"]["mediation"]["passes"]), "upstream depth mediation no longer passes")
    _require(float(graded["decoder_metrics"]["Dte"]["pearson"]) >= 0.99, "heldout depth decoding degraded")

    return {
        "integrity": integrity,
        "candidate_universes": {
            "quote": {"count": len(quote_sites), "sha256": sha256_file(quote_csv)},
            "bracket": {"count": len(bracket_sites), "sha256": sha256_file(bracket_csv)},
        },
        "necessity": {
            "quote_U": {
                "site": "0.mlp.resid_delta:460",
                "accuracy_after_ablation": quote_u["ablated_contrast_accuracy"],
                "accuracy_damage_rank": quote_u["singleton_ranks"]["accuracy"],
            },
            "bracket_R": {
                "site": "4.attn.resid_delta:1079",
                "accuracy_after_ablation": bracket_r["ablated_contrast_accuracy"],
                "margin_damage_rank": bracket_r["singleton_ranks"]["margin"],
            },
        },
        "redundancy": {
            "quote": "not_recovered",
            "bracket": "not_recovered",
        },
        "notable_negative_results": {
            "augmented_progressive_signature": "not_accepted",
            "raw_Rmid_signature": "not_accepted",
            "surface_density_Dte_pearson": surface_density["decoder_metrics"]["Dte"]["pearson"],
            "active_density_Dte_pearson": active_density["decoder_metrics"]["Dte"]["pearson"],
        },
        "progressive_depth": {
            "downstream_depth_read": progressive_best["site_ids"],
            "upstream_depth_handle": upstream_best["site_ids"],
            "Dte_pearson": graded["decoder_metrics"]["Dte"]["pearson"],
            "all_heldout_gates_pass": True,
            "accepted_model": "X -> D -> R -> Y",
        },
    }
