# ACR Phase A0 Split and Resource Audit

**Audit date:** 2026-08-02

**Pinned LIBERO revision:** `8f1084e3132a39270c3a13ebe37270a43ece2a01`

**Scope:** source, initialization metadata, historical population identifiers,
and static storage only

## 1. Exit conclusion

**Population ledger: PASS.**

- All four required LIBERO suites map to exactly 10 ordered tasks.
- Every mapped task has exactly 50 pinned initial states, IDs `0-49`.
- Historical SAVR work consumed only LIBERO-Spatial, tasks `0-9`, states
  `0-9`, seed `0`, plus correctness queries that did not create rollouts.
- No result filename or manifest indicates any state ID `10-49` was executed.
- No historical run uses LIBERO-Object, LIBERO-Goal, or LIBERO-10.
- No ACR result, run, configuration, report, or manifest exists.
- The ACR development, confirmation, transfer, primary-final, and reserve
  populations therefore remain unopened.

No simulator was started, no model/checkpoint was loaded, no GPU was queried
or used, and no data was downloaded during this audit. Historical result
documents were consulted only to reconcile population identifiers; no new
outcome was generated and no protected final outcome was accessed.

## 2. Source verification

| Source | SHA-256 |
|---|---|
| `third_party/LIBERO/libero/libero/benchmark/libero_suite_task_map.py` | `0c950df0a785aa55de968bb38ccd865d2017f71ddbe6f48cfd05ac0742b6d62d` |
| `third_party/LIBERO/libero/libero/benchmark/__init__.py` | `4ce6edbb00b3280692f2e2b2e8a14bf6e2d5878965b0484a386c0355a867ecc5` |

The benchmark implementation uses task order index `0` by default and loads
each task's `.pruned_init` file. The ordered suite map and actual initialization
files were checked directly at the pinned revision.

## 3. Verified suite mappings and counts

### LIBERO-Spatial

| ID | Ordered task | States |
|---:|---|---:|
| 0 | `pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate` | 50 |
| 1 | `pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate` | 50 |
| 2 | `pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate` | 50 |
| 3 | `pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate` | 50 |
| 4 | `pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate` | 50 |
| 5 | `pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate` | 50 |
| 6 | `pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate` | 50 |
| 7 | `pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate` | 50 |
| 8 | `pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate` | 50 |
| 9 | `pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate` | 50 |

Aggregate verification hash over the 10 ordered file hashes:
`be28d0f5f515e11bc1c11084b55622e2971c23f0f3fe8aaed28bf7e722c88205`.

### LIBERO-Object

| ID | Ordered task | States |
|---:|---|---:|
| 0 | `pick_up_the_alphabet_soup_and_place_it_in_the_basket` | 50 |
| 1 | `pick_up_the_cream_cheese_and_place_it_in_the_basket` | 50 |
| 2 | `pick_up_the_salad_dressing_and_place_it_in_the_basket` | 50 |
| 3 | `pick_up_the_bbq_sauce_and_place_it_in_the_basket` | 50 |
| 4 | `pick_up_the_ketchup_and_place_it_in_the_basket` | 50 |
| 5 | `pick_up_the_tomato_sauce_and_place_it_in_the_basket` | 50 |
| 6 | `pick_up_the_butter_and_place_it_in_the_basket` | 50 |
| 7 | `pick_up_the_milk_and_place_it_in_the_basket` | 50 |
| 8 | `pick_up_the_chocolate_pudding_and_place_it_in_the_basket` | 50 |
| 9 | `pick_up_the_orange_juice_and_place_it_in_the_basket` | 50 |

Aggregate verification hash over the 10 ordered file hashes:
`1d932dba80fddf2649b9778b3e4cccd681169e1954297077e578afcda92c2ff8`.

### LIBERO-Goal

| ID | Ordered task | States |
|---:|---|---:|
| 0 | `open_the_middle_drawer_of_the_cabinet` | 50 |
| 1 | `put_the_bowl_on_the_stove` | 50 |
| 2 | `put_the_wine_bottle_on_top_of_the_cabinet` | 50 |
| 3 | `open_the_top_drawer_and_put_the_bowl_inside` | 50 |
| 4 | `put_the_bowl_on_top_of_the_cabinet` | 50 |
| 5 | `push_the_plate_to_the_front_of_the_stove` | 50 |
| 6 | `put_the_cream_cheese_in_the_bowl` | 50 |
| 7 | `turn_on_the_stove` | 50 |
| 8 | `put_the_bowl_on_the_plate` | 50 |
| 9 | `put_the_wine_bottle_on_the_rack` | 50 |

Aggregate verification hash over the 10 ordered file hashes:
`70a38c1a25b21157cc8e857e0f83e51137db991fac6ff3ef88aa3c905793a207`.

### LIBERO-10

| ID | Ordered task | States |
|---:|---|---:|
| 0 | `LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket` | 50 |
| 1 | `LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket` | 50 |
| 2 | `KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it` | 50 |
| 3 | `KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it` | 50 |
| 4 | `LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate` | 50 |
| 5 | `STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy` | 50 |
| 6 | `LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate` | 50 |
| 7 | `LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket` | 50 |
| 8 | `KITCHEN_SCENE8_put_both_moka_pots_on_the_stove` | 50 |
| 9 | `KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it` | 50 |

Aggregate verification hash over the 10 ordered file hashes:
`aa133df0f5481f06140fd3f5a1ce356571497bbfdc98d265dc390338dcdcd036`.

The aggregate values above are audit digests computed by hashing the
concatenation of the 10 individual SHA-256 strings in task order. They are not
substitutes for the pinned Git revision or per-file verification during runs.

## 4. Historical population audit

| Historical stage | Suite | Tasks | State IDs | Seed | Population effect |
|---|---|---:|---:|---:|---|
| Phase 1 import/environment smoke | none | none | none | none | No rollout population |
| Phase 2A FR smoke | Spatial | 0 | 0 | 0 | Consumed |
| Phase 2B FR pilot | Spatial | 0-9 | 0-4 | 0 | Consumed |
| Phase 4 real-model correctness | Spatial metadata | 0 | 0 | 0 | Queries only; reinforces design exposure |
| Phase 5 structural policy smoke | Spatial | 0 | 0-2 | 0 | Consumed |
| Phase 6 FR trace collection | Spatial | 0-9 | 0-9 | 0 | Consumed for signal/calibration design |
| Phase 6 SAVR grid | Spatial | 0-9 | 0-9 | 0 | Consumed for method selection |
| Phase 6R-C correctness/recovery | Spatial metadata | 0 | 0 | 0 | Queries only; no rollout |
| Phase 6R-D Stage 1 | Spatial | 0-9 | 0-2 | 0 | Consumed across candidate methods |
| Phase 6S-D validation | Spatial | 0-9 | 3-9 | 0 | Consumed by SAVR3 |

Manifest reconciliation covered every manifest currently under TITAN
`results/*/manifest.json`. Reports and frozen configs resolved fields omitted
from older manifest schemas. A filename-level scan found only task IDs `00-09`
and state IDs `00-09`; no path encoded state `10` or greater.

The conservative ledger result is therefore:

```text
LIBERO-Spatial tasks 0-9, states 0-9, seed 0: consumed for design/calibration.
All other suite/state/seed combinations in the ACR protocol: untouched.
```

## 5. Frozen ACR population ledger

| Role | Suite | Tasks | State IDs | Seed | Status after A0 |
|---|---|---:|---:|---:|---|
| Development Stage 1 | Object | 0-9 | 0-2 | 0 | UNTOUCHED |
| Development Stage 2 | Object | 0-9 | 3-9 | 0 | UNTOUCHED |
| Confirmation | Goal | 0-9 | 0-9 | 0 | UNTOUCHED |
| Transfer | LIBERO-10 | 0-9 | 0-9 | 0 | UNTOUCHED |
| Primary final | Spatial | 0-9 | 10-49 | 7 | LOCKED / UNTOUCHED |
| Primary final | Object | 0-9 | 10-49 | 7 | LOCKED / UNTOUCHED |
| Primary final | Goal | 0-9 | 10-49 | 7 | LOCKED / UNTOUCHED |
| Primary final | LIBERO-10 | 0-9 | 10-49 | 7 | LOCKED / UNTOUCHED |
| Reserve | all four | 0-9 | 10-49 | 17, 27 | LOCKED / UNTOUCHED |

## 6. Confirmation that no ACR outcome exists

Checks performed:

- no result, configuration, or report filename contains `ACR`;
- no result manifest contains `ACR`, `asymmetric camera`, or
  `scene camera refresh`;
- tracked source, scripts, tests, configurations, results, and reports contain
  no ACR implementation or result reference;
- the only ACR material before this checkpoint was proposal/protocol/status
  documentation.

**Decision:** no ACR outcome exists. No performance statement can be made.

## 7. Phase A0 resource audit

| Item | Observed |
|---|---|
| GPU use | None; no GPU availability command was issued |
| Model/checkpoint loading | None |
| Simulator | Not imported or started |
| Network download | None |
| Initialization inspection | CPU-only `torch.load(..., map_location="cpu")` on 40 pinned init files to count states |
| TITAN project footprint before A0 synchronization | approximately `31G` |
| Filesystem availability at audit | `420,963,584 KiB` available on the filesystem containing `/home/ved/SAVR` |
| A0 artifact cap | 512 MiB |
| A0 new artifacts | Markdown audit/design/report files only; far below cap |

All remote read-only commands began in `/home/ved/SAVR` and targeted only
files inside that project. Nothing outside `/home/ved/SAVR` was inspected for
project content or modified.

## 8. Phase A0 gate

**PASS.** The population ledger is verified, protected populations remain
unopened, no ACR outcome exists, and A0 remained within its CPU-only/no-download
resource boundary.
