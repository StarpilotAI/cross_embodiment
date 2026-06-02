# Cosmos background-generation report

Live (permanent, on hisham.bedri@gmail.com's here.now account):

**https://wintry-tulip-tj52.here.now/**  ·  slug `wintry-tulip-tj52`

A single-page report of the cross-embodiment → NVIDIA Cosmos Transfer2.5
experiment: taking the bimanual-YAM composite video and regenerating its
background with edge-controlled video-to-video, running locally on the RTX 5090.

## Files

- `index.html` — the report source (tracked here).
- `publish.sh` — re-publish to the **same** URL above after editing `index.html`
  or regenerating the videos.

## Videos

The embedded `.mp4`s are **not** committed (large, derived, and already hosted on
here.now). They live in the gitignored asset dir:

    data_overlay_industrial/cosmos/report/*.mp4

Regenerate them with the cosmos pipeline (`data_overlay_industrial/cosmos/`,
see that folder's README) — winning config is edge `control_weight 0.5`,
`resolution "480"`, `--disable-guardrails`, on the 5090 box (`jchir@192.168.1.157`,
WSL2 Ubuntu, repo `~/cosmos-transfer2.5`).

## Update the live site

```sh
# edit cosmos_report/index.html, ensure the asset videos exist, then:
bash cosmos_report/publish.sh
```

Requires the here.now API key at `~/.herenow/credentials` (already saved on this
machine). `publish.sh` always targets slug `wintry-tulip-tj52`, so the URL never
changes.
