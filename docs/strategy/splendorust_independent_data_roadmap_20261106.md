# Broader self play data after the one pass comparison

6 November2026. The current400-game one-pass strength comparison continues
unchanged. This next study is prepared only; no larger collection, fitting,
evaluation or cloud rental has launched.

Recommend1,024 fresh training games plus128 untouched held-out games. From the
same new corpus compare192games×16positions with1,024games×3positions. Both arms
receive3,072examples and96updates, with two fixed repetitions each. This tests
broader experience without simultaneously lengthening training. The same fixed
original network supplies512-simulation teacher decisions and initializes allfits.

Commit allfour fits before held-out evaluation. Both broad fits are designated
in advance for a400-game matched128 comparison against the original network,
with12 excluded operational games and full verification. Narrow fits are offline
controls; no broad-versus-narrow live-strength claim follows from this design.
No selection of the best repetition or offline value-loss admission gate.

The estimated local cost is48-60hours including all checks, with75-90GiB of new
evidence under the current detailed format. Use a suitable destination and the
separately prepared actual bounds; no RAM or forecast gates. The full proposal is
in `docs/engineering/splendorust_independent_data_preparation_20261106.md`.

A positive current result would strengthen the case for scaling this short
recipe. An inconclusive result does not rule out data diversity as a bottleneck.
Clear deterioration should prompt a saved-evidence diagnosis before this recipe
is used for the larger collection. Preserve every outcome; do not enlarge the
current comparison or alter its fixed candidates.

For speed, CPU8 remains the measured practical default. Independent game shards
on additional CPUs or a separately qualified cloud backend are plausible future
options. A small2-4instance-hour/$10compute rental pilot could establish actual
games per dollar; published hourly prices alone do not establish a speedup.
Cloud porting, lossless recording changes and any paid capacity remain proposals.

The wider aim is a network that learns stronger play from search-guided self-play.
This study addresses independent experience rather than more search per move or
more repetitions over the same games. It is a controlled next step, not a claim
that1,024 games will be enough for superhuman strength. OriginalEF8A remains the
reference; external testing and six-world work remain deferred.
