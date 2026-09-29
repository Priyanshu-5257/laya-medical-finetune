"""Label-preserving permutations of already tokenized Laya choice items."""
from __future__ import annotations


def permute_choice_item(item: dict, order: list[int], sep_token_id: int) -> dict:
    """Return an item whose option at new position j came from old position order[j]."""
    ids = item["ids"]
    markers = item["markers"]
    k = len(markers)
    if sorted(order) != list(range(k)):
        raise ValueError("order must be a permutation of option positions")
    option_end = ids.index(sep_token_id, markers[-1])
    spans = [ids[markers[i]:markers[i + 1] if i + 1 < k else option_end]
             for i in range(k)]
    new_ids = list(ids[:markers[0]])
    new_markers = []
    for old_position in order:
        new_markers.append(len(new_ids))
        new_ids.extend(spans[old_position])
    new_ids.extend(ids[option_end:])
    result = dict(item)
    result["ids"] = new_ids
    result["markers"] = new_markers
    result["target"] = [item["target"][i] for i in order]
    result["label"] = order.index(item["label"])
    return result
