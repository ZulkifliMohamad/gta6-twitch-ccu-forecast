# gta6-twitch-ccu-forecast

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/ZulkifliMohamad/gta6-twitch-ccu-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/ZulkifliMohamad/gta6-twitch-ccu-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Forecasts GTA VI launch-day Twitch viewers hour by hour from hour 1's print, learning the launch-day curve shape from past big launches.

## Run it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

## Model

Launch days have a shape, not just a level: hour 1 is the small hours of
the night, viewers dip, then climb as the US unlocks and peak in the US
afternoon before easing off. `main.py` learns that shape from the case's
own comparables — the trimmed mean of each hour's log-ratio to its
launch's hour-1 print, with an interquartile-range spread, both smoothed
over three hours — and anchors it on the case's hour-1 print. Launches
whose hour 1 was nearly empty are left out of the training pool, as is
the game's own pre-release history. Standard library only, no network,
deterministic.

---

<sub>Entered on [make.fun](https://make.fun/?ref=6ksyay6h&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
