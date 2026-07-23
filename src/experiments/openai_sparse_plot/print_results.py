def short_result(row, handle_order):
    # Keep all useful evaluation results.
    result = {
        "sites": row["site_ids"],
        "strength": row["strength"],
        "score": row["summary"]["score"],
        "sensitivity": row["sensitivity_score"],
        "invariance": row["invariance_score"],
        "order": handle_order(row),
        "recovery_passed": row["recovery_passed"],
    }

    if "is_D" in row:
        result["is_D"] = row["is_D"]

    if "r_accuracy" in row:
        result["r_accuracy"] = row["r_accuracy"]

    if row["restoration"] is not None:
        result["restoration"] = row["restoration"]
        result["restoration_passed"] = row["restoration_passed"]
        result["edge_certified"] = row["edge_certified"]

    return result


def print_best_handles(variable, handles, handle_order, selection_key):
    best_by_sites = {}
    for row in handles:
        sites = tuple(sorted(row["site_ids"]))
        if sites not in best_by_sites or selection_key(row) > selection_key(best_by_sites[sites]):
            best_by_sites[sites] = row

    print(f"\nBest {variable} handles:")
    for i, row in enumerate(sorted(best_by_sites.values(), key=selection_key, reverse=True), 1):
        print(i, short_result(row, handle_order))


def print_refinement(variable, chain, rounds, handle_order, limit=None):
    print(f"\n{variable} chain:")
    for row in chain:
        print(row["name"], short_result(row, handle_order))

    print(f"{variable} refinement rounds:", len(rounds))
    for i, round_result in enumerate(rounds, 1):
        results = round_result["results"]
        recovered = [row for row in results if row["recovery_passed"]]
        certified = [row for row in results if row["edge_certified"]]
        print(f"Round {i}: downstream={round_result['downstream_handle']['site_ids']}, tested={len(results)}, recovered={len(recovered)}, certified={len(certified)}")
        ranked = sorted(results, key=lambda row: (row["sensitivity_score"], row["invariance_score"], row["summary"]["score"]), reverse=True)
        for row in ranked[:limit]:
            print(short_result(row, handle_order))


def print_calibration(variable, results, handle_order, limit=None):
    print(f"\n{variable} calibration results:")
    ranked = sorted(results, key=lambda row: row["summary"]["score"], reverse=True)
    for row in ranked[:limit]:
        print(short_result(row, handle_order))



def print_frozen_handle(variable, handle, handle_order):
    result = short_result(handle, handle_order)
    result["name"] = handle["name"]
    print(f"\nFrozen {variable}:", result)

def print_discovery_results(variable, candidate_handles, cal_results, valid_handles, handle_order, selection_key):
    print(f"\n{variable} candidate handles:", len(candidate_handles))
    print(f"{variable} calibration results:", len(cal_results))
    print(f"{variable} valid handles:", len(valid_handles))
    print_calibration(variable, cal_results, handle_order)
    print_best_handles(variable, valid_handles, handle_order, selection_key)