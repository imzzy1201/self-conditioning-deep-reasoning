"""Reusable tooling for the FLOP-matched experiment pipeline.

Modules:
    flops     -- per-micro-batch FLOP persistence (flops.json), cumulative-FLOPs
                 readout from training_state.pt, FLOP-matched pair selection,
                 and audits.
    evaluate  -- geometry sanity check, sampler-correct inference, and scoring.
"""