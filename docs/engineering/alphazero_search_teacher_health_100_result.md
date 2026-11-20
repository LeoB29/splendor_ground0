# AlphaZero search-teacher health audit 100 result

100 completed on 2026-08-14. Exact generation-3 search beat its own raw network 66-2-32 over 50 paired seeds / 100 games: 67.0% score share, mean margin +2.25, and 67.0% from both seats. All games completed normally. Frozen decision: `search_teacher_healthy_clean_resume_permitted`.

The larger panel shows that the earlier 8-8 diagnostic was noise: the generation-3 search teacher remains materially stronger than the raw network.

The offline audit nevertheless found a specific reserve-search pathology. From generation 2 to generation 3, raw-policy `RESERVE_DECK` top rate fell 5.22 percentage points and raw prior mass was essentially unchanged (-0.036 points), while search visit-top reserve rate rose 5.22 points and visit mass rose 1.45 points. Generation-3 search promoted reserve 1,059 times and demoted it 748 times. Selected reserve rows averaged terminal outcome -0.065. This implicates search/Q-value behavior rather than a raw-prior collapse, but it did not prevent a large aggregate search lift.

No replay was modified, no checkpoint was trained, and exact generation 2 remained installed. The evidence permits a separately frozen continuation that removes the inadmissible forced-pass game by an outcome-blind deterministic replacement policy.
