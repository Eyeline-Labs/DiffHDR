import re
import torch
from typing import Iterable, List, Sequence, Set

# keywords you care about; you can pass your own sequence too
DEFAULT_KEYWORDS = ("q", "k", "v", "o", "ffn.0", "ffn.2")

# map simple keys -> regexes that catch common naming variants
KEYWORD_REGEX = {
    "q":      r"(?:^|[.])(?:q|q_proj)(?:$|[.])",
    "k":      r"(?:^|[.])(?:k|k_proj)(?:$|[.])",
    "v":      r"(?:^|[.])(?:v|v_proj)(?:$|[.])",
    "o":      r"(?:^|[.])(?:o|o_proj|out_proj)(?:$|[.])",
    "ffn.0":  r"(?:^|[.])(?:ffn|mlp)[.]0(?:$|[.])",
    "ffn.2":  r"(?:^|[.])(?:ffn|mlp)[.]2(?:$|[.])",
}

def _compile_patterns(keywords: Iterable[str]) -> List[re.Pattern]:
    pats = []
    for k in keywords:
        regex = KEYWORD_REGEX.get(k, re.escape(k))
        pats.append(re.compile(regex))
    return pats

def _matches_any(name: str, patterns: Sequence[re.Pattern]) -> bool:
    return any(p.search(name) for p in patterns)

def make_lora_targets_by_keywords(
    vace_model: torch.nn.Module,
    block_indices: Iterable[int],
    keywords: Iterable[str] = DEFAULT_KEYWORDS,
) -> List[str]:
    """
    Returns fully-qualified module names under vace_blocks.{i}.*
    whose submodule is nn.Linear and whose name matches any keyword.
    """
    patterns = _compile_patterns(keywords)
    want_prefixes: Set[str] = {f"vace_blocks.{i}." for i in block_indices}
    targets: List[str] = []

    for full_name, module in vace_model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if not any(full_name.startswith(pref) for pref in want_prefixes):
            continue
        # match keywords like q/k/v/o/ffn.0/ffn.2 anywhere in the qualified name
        if _matches_any(full_name, patterns):
            targets.append(full_name)

    # De-dupe but keep order stable
    seen = set(); uniq = []
    for n in targets:
        if n not in seen:
            uniq.append(n); seen.add(n)
    return uniq

def parse_int_list(s): return {int(x) for x in s.split(",") if x.strip().isdigit()}

def resolve_vace_indices(vace, block_ids_set, block_indices_set):
    idx = set(block_indices_set or [])
    if block_ids_set:
        bid2idx = {bid: i for i, bid in enumerate(vace.vace_layers)}
        idx |= {bid2idx[bid] for bid in block_ids_set if bid in bid2idx}
    return sorted(idx)
