# Study labels

All three use `data/rank_full_track05` (55 features stored, `track_min_pt = 0.5`,
corrections on, analytic min-bias backend, 256 grid cells, vertex likelihood on).
Only the feature list differs.  Seed 12345 throughout.

| result directory | channel | feature list | why |
|---|---|---|---|
| `mva/Hbb/results/study_bbrank20` | H(bb) | H(bb)-ranked 20 | the list stage 3 selected for this channel |
| `mva/Hcc/results/study_ccrank20` | H(cc) | H(cc)-ranked 20 | the list the H(cc) RFE produced |
| `mva/Hcc/results/study_bbrank20` | H(cc) | **H(bb)**-ranked 20 | does the H(bb) list transfer to charm? |

The two H(cc) rows differ only in which channel's ranking supplied the features,
so they isolate whether the ranking is channel-specific.

Stage 3 comparisons (3 seeds each) live in `stage3_<arm>_s<seed>/` and are
separate from these.
