---
title: "Open F1 Data Acquisition"
---

# F1 Lap Time vs Finished Position

F1 has been gaining its popluarity this past few years. There are always a driver who dominates the championship. 

## Can we predict who will be the dominator? 

Set of variables that may be interesting to look at can be ```position```(finished position) and ```best_lap```(best lap time), and artificial variable called ```consistency```. 

## Ethical Question

Data is available at [OpenF1](https://openf1.org) for free for personal use. "The entire project is open source, community-driven, and built with transparency", so we can rely on generosity and accuracy. 

## Basic API Practice

Here are steps to fetch the data with API to season results. 

```python
import pandas as pd
import requests

# Basic OpenF1 API interaction to get season results
BASE_URL = "https://api.openf1.org/v1"
def get_season_results(season: int) -> pd.DataFrame:
    url = f"{BASE_URL}/sessions"
    params = {
        "year": season
        }
    sessions = requests.get(url, params=params)
    sessions.raise_for_status()
    session_data = sessions.json()
    return pd.DataFrame(session_data)
```

This will give you,

```python
# Example usage
df = get_season_results(2023)
df.head()
```

|          | session_key | session_name | ... |
| -------- | -------- | -------- | -------- |
|       0  |   9222   | Practice | ... |
|       1  |   7763   | Practice | ... |

For our EDA, we merge and stack Sessions result and Laps datasets across 15 sessions in 2023 season. We will also add artificial features called ```best_lap```, ```consistency``` and ```avg_lap``` which are just the best lap time, standard deviation and the average lap time for the drivers. 

## Overview on Laps datasets and Session result datasets

Here's the quick overview for our Laps and Session result datasets.

### Driver Stats

- **Total sample size**: (1802, 6)
- **Main features**: ```session_key``` (References races), ```driver_number``` (References drivers), ```avg_lap```(Average lap time grouped by ```driver_number``` and ```session_key```), ```best_lap```(Average lap time grouped by ```driver_number``` and ```session_key```), ```consistency```(Standard deviation for the lap time grouped by ```driver_number``` and ```session_key```), ```position```(Finished position grouped by ```driver_number``` and ```session_key```)
- **Transformation**: Stacked data was grouped by ```driver_number``` and ```session_key```, and new artificial featrues, ```best_lap```, ```consistency``` and ```avg_lap```, are created.
- **Causions**: Data is clean, well organized, however, ```lap_time``` and ```position``` variables are heavily based on the ```session_key``` becuase each racing track has difference track distance, layout, and orientation. This EDA did not take that an account for simplicity.

### More information about the data

You can perform your own EDA on OpenF1 datasets! Check out [OpenF1 API endpoints](https://openf1.org/docs/#api-endpoints) for more datasets to explore!

Check out my (Github repo)[https://github.com/Vaoikun/Data_Acquisition_Blog] for full code.

Thank you for reading!
