# Derived data

No R1Hz measurements are committed here. Recreate the bounded pilot window from
the canonical Harvard Dataverse release (DOI
[10.7910/DVN/RCB5VJ](https://doi.org/10.7910/DVN/RCB5VJ)) with:

~~~bash
python3 scripts/extract_r1hz_window.py \
  --start "2018-01-28 12:00:00" \
  --seconds 3600 \
  --output data/derived/r1hz_2018-01-28_1200.csv
~~~

The extractor reads the source file without modifying it and excludes rows
whose marker is s, because their recovered values are synthetic.
