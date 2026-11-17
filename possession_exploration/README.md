# Possession exploration

DFL changes possession on every touch: a deflection or a clearance that hits a rival already
counts as a new possession. The challenge only counts a *controlled* gain
([CHALLENGE.md](../CHALLENGE.md)), so many of DFL's short spells are not possessions for us.

We built this app to look at those short transitions in DFL and to check whether they fit the
challenge's definition and look like the public GT. We also used it to relabel the ones that were too
forced. We only reviewed the very short spells:

- if it was clearly a touch without control (a clearance, a deflection), we merged it away;
- if we had doubts, we left it as it was. An occasional short possession is fine, since we don't know
  exactly how the GT treats them.

The annotations live in [labels/pokexavi.jsonl](labels/pokexavi.jsonl) and the possession model
trains on them ([possession_model/README.md](../possession_model/README.md)).

## Run

From the `Hacktrick` root:

```powershell
python external/fetch_dfl.py                       # once, if external/dfl/processed is empty
python possession_exploration/prepare_dfl.py       # copy the 7 DFL matches for the app
python -m streamlit run possession_exploration/app.py
```

Type your name in the sidebar. Every save adds lines to `labels/<you>.jsonl`; commit and push that
file to share it.
