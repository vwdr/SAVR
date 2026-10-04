"""Read-only inventory of configuration state references, never outcome files.

This is not an exposure certificate: configurations include unexecuted plans,
and some historical launch formats do not encode state IDs explicitly.
Run locally or via stdin over ssh titan from the scoped repository root.
"""
import hashlib
import json
from pathlib import Path


def state_references(value, path="$", output=None):
    output = [] if output is None else output
    if isinstance(value, dict):
        for key, child in value.items():
            location = f"{path}.{key}"
            if key in {"initial_state_id", "initial_state_ids", "state_ids",
                       "final_state_ids", "all_four_suites_state_ids"}:
                numbers = child if isinstance(child, list) else [child]
                if all(type(n) is int for n in numbers):
                    output.append((location, numbers))
            state_references(child, location, output)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            state_references(child, f"{path}[{index}]", output)
    return output


def audit(root):
    root = root.resolve()
    allowed = {Path("/Users/veddwivedi/Documents/VLA/SAVR"), Path("/home/ved/SAVR")}
    if root not in allowed:
        raise ValueError("Run only from the approved SAVR repository")
    rows = []
    for path in sorted((root / "configs").rglob("*.json")):
        if not path.resolve().is_relative_to(root) or path.is_symlink():
            raise ValueError("Configuration path escapes scope or is a symlink")
        raw = path.read_bytes()
        references = state_references(json.loads(raw))
        ids = sorted({n for _, ns in references for n in ns})
        rows.append(dict(path=str(path.relative_to(root)), sha256=hashlib.sha256(raw).hexdigest(),
                         explicit_state_ids=ids,
                         reserved_references=[dict(key=p, ids=ns) for p, ns in references
                                              if any(n >= 10 for n in ns)]))
    # Names only: no records, summary contents, predictions or locked data opened.
    result_directories = sorted(p.name for p in (root / "results").iterdir() if p.is_dir())
    return dict(schema="confirmation-metadata-inventory-v1", root=str(root),
                config_count=len(rows), configs=rows, result_directories=result_directories,
                outcome_files_opened=[], exposure_certified=False,
                limitation="Configuration references are not proof of executed exposure or its absence.")


if __name__ == "__main__":
    print(json.dumps(audit(Path.cwd()), indent=2, sort_keys=True))
