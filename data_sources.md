# Public Data Sources

This repository does not require private or login-protected data. The external
checks use official USGS/ScienceBase data releases:

| Dataset | DOI | Role in this package |
|---|---|---|
| Oso ring-shear strength testing | https://doi.org/10.5066/F7KH0KSD | Laboratory pressure-dissipation consistency check |
| Mount Kaba-san field-scale landslide initiation experiment | https://doi.org/10.5066/P18XMZPC | Field-scale pore-pressure dissipation consistency check |
| USGS debris-flow flume sensor data, June 2016 | https://doi.org/10.5066/F7N58JKH | Observed pore-pressure regime inventory |

The large raw Mount Kaba-san and flume files are kept outside the release ZIP.
The reproducible release includes scripts, normalized outputs and source
metadata. This avoids inflating the journal upload package while preserving
traceability to the official USGS records.
