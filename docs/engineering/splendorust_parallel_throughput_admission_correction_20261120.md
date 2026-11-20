# Parallel-throughput pilot: launch-gate correction

20 November 2026. This is a technical launch-gate correction only. The user asked
for it in this session ("Do it here"). It authorizes preparation and freeze. It
does **not** authorize a launch: launching still needs an explicit user go-ahead
with quiet-machine confirmation.

## What was wrong

The 15 November pilot (`tools/splendorust_parallel_throughput_20261115`, freeze
`445ddb6a…`) was frozen while the 4,096-game collection was still running. Two
defects would make its launch gate refuse before any model or game started:

1. **It binds receipts that will never exist.** The gate requires completion
   receipts from the original collection attempt
   (`outputs/splendorust_collection4096_20261115_01/observed_owner_exit.json` and
   `final_audit/worker/result.json`). That attempt ended incomplete on
   16 November. The collection was completed and accepted by the amended recovery
   `outputs/splendorust_collection4096_resume_20261118_02` instead.
2. **It depends on an external `rg` binary.** The registry collision scan calls
   `rg` (ripgrep), which is not installed on this machine. It was presumably
   available to the agent that prepared the pilot.

3. **Its pinned Windows console host is out of date.** The plan inherits the
   six-world freeze's hash of `C:\Windows\System32\conhost.exe` (`c8614618…`).
   Windows servicing has since replaced that file (`dd5c3514…`, same size),
   most likely in the update that caused the 16 November restart. The pin is
   re-verified at launch and by every worker at startup. Live ownership also
   requires every console child's hash to equal it. So even with defects 1 and
   2 fixed, the first live service would have failed. This was found when this
   version's first freeze attempt refused on the stale hash, before writing
   anything. It was the only one of 2,548 bound inputs that had changed.

The original version stays frozen, unchanged and unlaunched. Its gate would
refuse in any case.

## What this version changes

`tools/splendorust_parallel_throughput_20261120` is a copy. Its output folder is
`outputs/splendorust_parallel_throughput_20261120_01` and its preparation folder
is `outputs/splendorust_parallel_throughput_preparation_20261120_01`. The changes
are limited to the following.

- **Launch gate (`owner.accepted_predecessors`).** It binds the accepted
  recovery's independent acceptance, actual root exit with all 3,831 owned
  identities absent, and launcher exit (freeze `842a86bb…`). It also binds the
  completed training4096 study's owner, root, launcher and observer exits
  (freeze `c1a14396…`). The original collection attempt's frozen reservation and
  process-absence checks are retained. The gate refuses if:
  - any other study package's code is running (a safe, repeatable refusal made
    before any claim is written);
  - the superseded 15 November version was ever launched or changed.
- **Registry discovery (`schedule.registered_files`).** A Python directory walk
  replaces `rg --files --no-ignore outputs -g …`. It uses the same seven basename
  globs, matched case-sensitively; skips dot-prefixed entries; does not follow
  links or junctions; and fails closed on unreadable folders. On the real
  registry it found every one of the 9,342 files the 15 November `rg` scan
  checked, plus later files. An independent reviewer ran the original `rg`
  command with a bundled ripgrep 15.0.0: both methods return exactly 9,824 files
  today. The walk takes about 1–4 minutes over about 5.2 million files under
  `outputs/`, so the 10-minute reservation bound's margin will shrink as
  `outputs/` grows.
- **Process checks.** The running-study check is repeated just before launch,
  after the multi-minute scan. The package exemption matches the exact package
  name. The collection's observer must also be absent. Limitation: command lines
  that cannot be read without administrator rights are not seen, so the
  quiet-machine confirmation is still required.
- **Final reservation validation (`schedule.final_validation`).** This step was
  an ad-hoc command outside the package in the original. It is now bound source.
- **Fresh engineering setups.** The original 50 setups are registered to the
  frozen original, so reusing them would collide. The renamed domain
  (`splendorust_parallel_throughput_20261120_v1`) yields 50 fresh setups, again
  checked against the whole registry. Setups are engineering-only and never
  enter a corpus.
- **Console host pin (`preparation.rebind_console`).** The pin is rebound to
  the current System32 `conhost.exe`, and any other path is refused. This
  applies both to `console_executable` and to the same entry in the inherited
  worker inputs. The superseded hash is recorded as plain fields
  (`console_rebinding`). `prepared.json` was written before this helper existed
  and keeps the inherited pin; the freeze applies and records the rebinding.
  **A further Windows update before launch would make the launch refuse
  safely.** Pause Windows Update until the pilot has run.
- **Version names.** Package, study and output paths are renamed. The plan binds
  this note, the original brief and protocol, and the original evidence review.

The following are unchanged, with source identical apart from renaming:
- the design, arms and order;
- the players: EF8A, float32, 512 simulations, three worlds, cpuct 0.4, depth 64;
- the deadlines (119/120 s live, 60/75 s collection);
- the identity and adoption bars, actual-model qualification, checks, caps,
  ownership and reporting.

## Checks before freeze

The following must pass before the freeze:
- the source-bound model-free suite, including new counterexamples:
  - the original gate's refusal, reproduced on the real files;
  - the corrected gate, passing on the real accepted receipts;
  - every bound receipt value enforced;
  - surviving predecessor processes and other running studies refusing launch;
  - the superseded original staying unlaunched;
  - the walk's equivalence with the original scan;
- the final reservation validation;
- the detached ownership fixture;
- one independent review.

The freeze records all of them. At launch, `rg` is no longer needed. The machine
should be quiet: close browsers and chat apps, and keep it plugged in. Process
Lasso is installed and could override the pilot's CPU pinning, so it should be
paused or set not to manage Python during the pilot.
