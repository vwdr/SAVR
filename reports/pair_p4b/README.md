# P4B design-audit lineage

`design_audit_v4.json` is the authoritative P4B proposal audit.

- `design_audit_v1.json` is a preserved pre-freeze draft that considered
  spending the locked-test population. That design was rejected before any
  unused label, model, or GPU access.
- `design_audit_v2.json` is a preserved intermediate audit that corrected the
  population boundary but projected the original horizon-only matching rule.
  It was superseded before execution because P4B uses stricter
  suite-by-horizon matching.
- `design_audit_v3.json` authenticates the final independent population,
  locked-test preservation, and design-aligned statistics before the execution
  artifact paths were added to the frozen configuration.
- `design_audit_v4.json` reproduces the same scientific design and calculations
  against the implementation-complete configuration. It used metadata and
  completed P4 evidence only, with CUDA hidden and no model access.

Neither superseded audit authorizes or reports a P4B experiment.
