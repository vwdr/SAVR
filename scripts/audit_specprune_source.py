#!/usr/bin/env python3
"""CPU-only source contract audit; never loads weights or changes upstream code."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

REFERENCE_SHA256 = "f40ee7883e16aab1a2d89b6e8f31cc81f6b8055120b1fefe169e05c7031098fa"
UPSTREAM_BLOBS = {
    "Readme.md": "a5a6b8f4a57d40b444b5e2aa3cf73bda3d1b2ad0",
    "LICENSE": "b2c22d519c9b30ad4cc809cb686074f658520044",
    "SETUP.md": "bed19d2975234f9ee279bec2b1acb510e4872b83",
    "modeling_llama.py": "12473d0af453e1f1d9be92432f73b4fd2dd816ef",
    "modeling_prismatic.py": "0eaf8fc4bbd1cc6e50c28e7c0d2d7e81d6b8a678",
    "run_libero_eval.py": "981e36be805c9a64fde490e06f3b8ce6792fa036",
    "openvla_utils.py": "be5b5ade3d786612bb0a4285d1c9bc29f0133eb4",
    "spec_prune_constants.py": "5807d3fdc7d50ca00f670bc6e115e88ab423dc5d",
}


def verify_source_identity(reference_hash, upstream_blobs):
    if reference_hash != REFERENCE_SHA256 or upstream_blobs != UPSTREAM_BLOBS:
        raise ValueError("source identity differs from the pinned checkpoint or upstream revision")


def method(tree, owner, name):
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == owner)
    return next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)


def head_slice(function):
    assignments = [n for n in ast.walk(function) if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == "actions_hidden_states" for t in n.targets)]
    assert len(assignments) == 1, "ambiguous action readout"
    value = assignments[0].value
    assert isinstance(value, ast.Subscript) and isinstance(value.slice, ast.Tuple)
    return value.slice.elts[1]


def integer_expression(node, names):
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return node.value
    if isinstance(node, ast.Name):
        return names[node.id]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -integer_expression(node.operand, names)
    if isinstance(node, ast.BinOp):
        left, right = integer_expression(node.left, names), integer_expression(node.right, names)
        if isinstance(node.op, ast.Add): return left + right
        if isinstance(node.op, ast.Sub): return left - right
        if isinstance(node.op, ast.Mult): return left * right
    raise ValueError("unsupported source expression")


def audit(upstream, reference):
    reference_hash = hashlib.sha256(reference.read_bytes()).hexdigest()
    blobs = {p.name: hashlib.sha1(b"blob " + str(p.stat().st_size).encode() + b"\0" + p.read_bytes()).hexdigest()
             for p in sorted(upstream.iterdir()) if p.is_file()}
    verify_source_identity(reference_hash, blobs)
    source = upstream / "modeling_prismatic.py"
    st, rt = ast.parse(source.read_text()), ast.parse(reference.read_text())
    owner = "OpenVLAForActionPrediction"
    sm = method(st, owner, "_regression_or_discrete_prediction")
    rm = method(rt, owner, "_regression_or_discrete_prediction")
    ss, rs = head_slice(sm), head_slice(rm)
    examples = []
    for prompt in (12, 34, 70):
        names = dict(ACTION_DIM=7, NUM_ACTIONS_CHUNK=8, NUM_PATCHES=513, NUM_PROMPT_TOKENS=prompt)
        # BOS + prompt (including final spacer) + 56 placeholders + stop + 513 inserted tokens.
        length = 1 + prompt + 56 + 1 + 513
        upstream_positions = list(range(length))[slice(integer_expression(ss.lower, names), integer_expression(ss.upper, names))]
        reference_positions = list(range(length))[slice(integer_expression(rs.lower, names), integer_expression(rs.upper, names))]
        assert len(upstream_positions) == len(reference_positions) == 56
        examples.append(dict(prompt_tokens=prompt, sequence_tokens=length,
                             upstream_first=upstream_positions[0], reference_first=reference_positions[0],
                             upstream_last=upstream_positions[-1], reference_last=reference_positions[-1],
                             equal=upstream_positions == reference_positions))
    def attention_flag(fn):
        calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Attribute) and n.func.attr == "language_model"]
        assert len(calls) == 1
        return next(ast.literal_eval(k.value) for k in calls[0].keywords if k.arg == "output_attentions")
    # Compare token-extension operations after deleting docstrings.
    def normalized_prepare(tree):
        fn = method(tree, owner, "_prepare_input_for_action_prediction")
        return [ast.dump(n, include_attributes=False) for n in fn.body
                if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str))]
    same_prepare = normalized_prepare(st) == normalized_prepare(rt)
    assert same_prepare, "input-extension difference requires a separate interpretation"
    return dict(
        kind="source_contract_audit_not_policy_result",
        upstream_revision="8091adc4b574ce9008d49a1dc9a210f4eec314c1",
        identical_input_extension_ast=same_prepare,
        upstream_readout_slice=ast.unparse(ss),
        reference_readout_slice=ast.unparse(rs),
        synthetic_layouts=examples,
        direct_same_checkpoint_readout_compatible=all(r["equal"] for r in examples),
        upstream_output_attentions=attention_flag(sm),
        reference_output_attentions=attention_flag(rm),
        gpu_or_model_execution=False,
        reference_sha256=reference_hash,
        upstream_git_blob_identity_verified=True,
        upstream_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(upstream.iterdir()) if p.is_file()},
        upstream_git_blob_sha1=blobs,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.upstream, args.reference), indent=2, sort_keys=True))
