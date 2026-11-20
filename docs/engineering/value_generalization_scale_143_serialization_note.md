# Experiment 143 reporting recovery

2026-09-08. All 5,120 games, both value fits, baseline selection and all 512 test
prediction files were saved before the original process exited. The exception
occurred in final JSON writing: NumPy Boolean scalar flags are not accepted by
the standard encoder. This is a reporting type error, not a failed data, model,
selection, numerical or scientific gate. Original wall time was 302.438 seconds.

Preserve the frozen source and protocol. The recovery script reproduces the
frozen evaluator with two reporting adapters: NumPy scalars become their exact
Python scalar values when serialized, and every attempted prediction-file write
instead requires exact equality with the already saved file. It generates no
games, fits no model, selects no checkpoint and changes no statistic or gate.
All 512 saved prediction tensors must match exactly before the report is accepted.

The original incomplete JSON, failure record and run log are copied into
`outputs/value_generalization_scale_143/serialization_failure`. Source/data/model
identities are checked before and after recovery. The independent frozen verifier
then reconstructs the complete experiment. Do not rerun the main pipeline or
interpret this exception as evidence about value learning.
