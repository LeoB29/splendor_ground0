# AlphaZero Generation-11 External Benchmark 106 Result

Date completed: 2026-08-18

Experiment 106 completed all 400 scheduled games normally with empty stderr.
Searched gen11 versus raw legal `035` finished 53-0-147: 26.5% score share,
mean margin -4.055, seat shares 34% and 19%, paired CI95 [20%, 33%]. Raw gen11
finished 25-1-174: 12.75%, mean margin -7.225, seat shares 13.5% and 12%, CI95
[8.25%, 17.5%].

Against the exact gen8 anchors from 103, searched transfer changed by 0.0
percentage points (CI95 [-9, +9]) and raw transfer by +0.5 points (CI95
[-6, +7]). Gen11 search nevertheless added 13.75 points over its raw policy
against `035` (paired CI95 [6.75, 21]). Thus search remains useful, but the
third scaling block produced no detected external progress and remains far
below parity with `035`.

Final decision:
`generation_11_no_external_progress_detected_below_035_no_install`. No
training, self-play, promotion, or installation occurred. Exact gen2 remains
installed. Raw result SHA-256:
`7e4bb4c7d53f13bcf29258f5856ab0b5d6fbe56844c6c1d9d742426e780c0ca1`.
