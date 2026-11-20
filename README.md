[![make.fun bounty](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/3215242167-create/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/3215242167-create/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

# GTA VI trophy rarity forecast

A forecaster for make.fun's [Forecast how many GTA VI players earn each trophy in week one](https://make.fun/bounty/gta6-trophy-rarity-forecast-01) bounty ($250 in USDC). Given a case file with every read the house made of Exophase's GTA VI PS5 trophy page before the checkpoint, it writes five quantiles (5/25/50/75/95%) of each of the 55 trophies' earn rate at the case's target.

## Method

Per trophy, on the logit scale: the history is reduced to distinct snapshots (Exophase refreshes on its own schedule, so long stretches of reads are identical), then a recency-weighted line is fitted to logit(rate) against log time since release (half-life 24 h, last 24 h of snapshots, slope clamped to [0, 1.2] so a refresh step-drop never extrapolates into a decline) and read at the target; the line's level is anchored at the last read, so a refresh step inside the window cannot drag the median below where the trophy stands. When the history has never moved — the practice cases' situation, where every read is the same unrefreshed page — no trend exists, so the median sits 0.8 logit below the last read: the next refresh is expected to dilute, since players-tracked only grows and newly tracked players have earned little yet (launch night: 638 → 3,868 tracked, every rate but the 100% one lower).

The spread is honest about the week's uncertainty: a random-walk term (0.5 logit per √day, the house baseline's scale) plus the fit's residual scatter plus a 1.0-logit unseen-refresh term — Exophase can refresh between checkpoint and target and reprice every rate at once (launch night: 638 → 3,868 players tracked, every rate but the 100% one lower). The house baseline's narrow 90-minute band scores ~1.51 on the practice for exactly that reason; this entry's band is built to survive it. Tier priors (platinum 0.5%, gold 2%, silver 5%, bronze 10%) cover trophies with no usable history, and any per-case error still writes a valid persistence forecast.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Python 3.13, standard library only, no network. Reads every `<id>.in` in the input directory, writes `<id>.out` to the output directory. Deterministic: two runs on the same case write the same bytes.

## License

MIT — see `LICENSE`.
