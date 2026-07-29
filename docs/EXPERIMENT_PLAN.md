# SAVR Experiment Plan

Status: design only; no experiment is authorized by this document.

## 1. Research questions

1. Can visual features be reused while maintaining task success within a predeclared non-inferiority tolerance relative to Full Refresh?
2. Do robot-state and recent-action signals improve refresh decisions beyond visual change alone?
3. How do refresh rate, latency, and success trade off across thresholds and reuse horizons?
4. Which task dynamics and failure modes make feature reuse unsafe?

## 2. Initial stack candidate

- Base policy: OpenVLA-OFT LIBERO checkpoint
- Benchmark: LIBERO, beginning with a small smoke-test subset
- Hardware: one explicitly user-approved TITAN RTX GPU

This stack is provisional until the environment and one FR episode pass a controlled smoke test.

## 3. Common policy interface

All policies must share the same observation preprocessing, base checkpoint, action decoding, simulator settings, task set, initial states, and episode limits. Only the refresh decision may differ.

Required policies:

- FR: refresh every step
- PR(k): refresh every `k` steps
- VOR: refresh on image threshold or maximum horizon
- SAVR: refresh on image, robot-state, action-history, or maximum-horizon signal

## 4. Staged execution

### Stage A — compatibility smoke test

- create a project-local environment
- install pinned dependencies without system changes
- verify headless LIBERO rendering
- load one approved checkpoint
- run one FR episode
- record peak GPU memory and any compatibility errors

Exit condition: deterministic observation/action loop and complete run manifest.

### Stage B — instrumentation correctness

- identify and document the exact visual-feature tensor and cache boundary
- confirm FR wrapper matches the unwrapped base policy
- test cache invalidation, horizon enforcement, and reset behavior
- verify that timings use synchronized GPU measurements where applicable

Exit condition: unit/integration tests plus numerical or behavioral equivalence evidence for FR.

### Stage C — pilot threshold calibration

- use calibration tasks/episodes disjoint from final evaluation
- examine normalized image, state, and recent-action change distributions
- choose a small predeclared threshold grid
- select PR periods and maximum horizons with comparable refresh budgets where possible

Exit condition: frozen thresholds and evaluation protocol.

### Stage D — final evaluation

- run FR, PR, VOR, and SAVR on identical tasks and initial states
- use paired seeds/episodes
- record task success, timing, refresh decisions, and failures
- report confidence intervals and paired comparisons

### Stage E — ablations

- remove state signal
- remove action signal
- remove image signal
- vary maximum reuse horizon
- compare individual and combined triggers

## 5. Primary metrics

- episode task success
- success difference versus FR with confidence interval
- median and tail end-to-end step latency
- median and tail policy latency
- visual refresh rate and skipped-refresh percentage
- episode wall-clock time
- peak GPU memory

Secondary:

- trigger contribution and overlap
- cache age distribution
- task-level failure taxonomy

## 6. Experimental controls

- pin code revisions, checkpoint IDs, dependency versions, and benchmark revision
- freeze task lists, initial states, seeds, camera configuration, control frequency, and episode horizon
- warm up before timing
- separate model load time from steady-state inference
- synchronize CUDA around device timing
- retain raw per-step records and aggregate from them
- never tune thresholds on final evaluation episodes

## 7. Statistical plan

Define the task-success non-inferiority tolerance before final evaluation. Use paired episode outcomes across policies where initial states are shared. Report uncertainty for success and latency, not only point estimates. Treat task-level outcomes as clustered when aggregating across tasks.

## 8. Stop conditions

Stop and request user direction if:

- a required action affects anything outside `/home/ved/SAVR`
- an environment requires system-wide changes or `sudo`
- storage estimates exceed the approved project budget
- all GPUs appear potentially shared and no allocation has been coordinated
- FR correctness cannot be established
- the selected base model does not expose a safe reusable visual-feature boundary
- the manuscript’s formal definition conflicts with the implementation plan
