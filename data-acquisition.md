---
title: "Open F1 Data Acquisition"
---

# F1 Lap Time vs Finished Position

Formula 1 has grown dramatically in popularity in recent years, attracting a new generation of fans and analysts. Each season typically features one or two dominant drivers who consistently finish at the top of the standings.
This raises an interesting analytical question:

## Can race results be partially explained by lap time performance metrics?

Using publicly available data from the OpenF1 API, we perform a simple exploratory data analysis (EDA) to investigate the relationship between:
- Best lap time
- Lap time consistency
- Final race position

Dominant drivers often appear to have both:
- strong raw pace
- strong consistency
We can attempt to quantify these traits using lap time statistics derived from race telemetry.

| Variable      | Description                                     |
| ------------- | ----------------------------------------------- |
| ```position```    | Final finishing position in a race              |
| ```best_lap```    | Fastest lap time achieved by a driver in a race |
| ```avg_lap```     | Average lap time for a driver during a race     |
| ```consistency``` | Standard deviation of lap times for a driver    |


## Ethical Question

The dataset used in this analysis is obtained from [OpenF1](https://openf1.org). OpenF1 provides free access to Formula 1 telemetry and race data through an open API.
### “The entire project is open source, community-driven, and built with transparency.” 
Because the data is openly available and intended for public use, it is appropriate for educational analysis and experimentation. However, users should be aware of API usage limits and the importance of responsible data access.


## Basic API Practice

The OpenF1 API provides several endpoints for retrieving race telemetry, session data, weather information, and more. We begin by retrieving all sessions for a given season.

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

Example usage:

```python
# Example usage
df = get_season_results(2023)
df.head()
```
This produces a table containing metadata about each session in the season.

|          | session_key | session_name | ... |
| -------- | -------- | -------- | -------- |
|       0  |   9222   | Practice | ... |
|       1  |   7763   | Practice | ... |

Each session_key uniquely identifies a race session and allows us to retrieve lap-level telemetry data.

For this exploratory analysis we combine two datasets:
- Session Results – finishing positions for each driver
- Lap Data – individual lap times recorded during the race
These datasets are merged using the keys:
- ```session_key```
- ```driver_number```

## Overview on Laps datasets and Session result datasets

Here's the quick overview for our Laps and Session result datasets.

### Driver Stats

- **Total sample size**: (1802, 6)
- **Main features and transormations**:

| Variable        | Description                               |
| --------------- | ----------------------------------------- |
| `session_key`   | Identifies the race session               |
| `driver_number` | Unique identifier for each driver         |
| `avg_lap`       | Average lap time for the driver in a race |
| `best_lap`      | Fastest lap recorded by the driver        |
| `consistency`   | Standard deviation of lap times           |
| `position`      | Final finishing position                  |

- **Causions**: A key limitation of this dataset is that lap times vary significantly between circuits.
Tracks differ in:
- circuit length
- number of corners
- track layout
- elevation changes
- accidents
Because of these differences, raw lap times are not directly comparable across races. In more advanced analyses, lap times are often normalized relative to the fastest lap in each race.
For simplicity, this exploratory analysis uses raw lap times while acknowledging this limitation.

### Exploratory Visualization

After a few more lines of code, we can examine whether lap time consistency correlates with finishing position.

<img src="consist_vs_pos.png" alt="Consistency vs Position" width="100">

Similarly, we can examine the relationship between best lap time and finishing position.

<img src="best_vs_position.png" alt="Best Lap vs Position" width="100">

These plots provide a simple way to explore whether raw pace or lap consistency better predicts race performance.

### Future Work

This analysis represents a simple exploratory study. Several improvements could be made in future work, such as,
incorporating tire strategy and pit stops, investigating weather effects, and compare driver performance across teams. OpenF1 provides many additional endpoints that could support more sophisticated analyses.

### More information about the data

You can perform your own analysis on OpenF1 datasets! Check out [OpenF1 API endpoints](https://openf1.org/docs/#api-endpoints) for documentation.

The complete code used in this analysis can be found on (Github repo)[https://github.com/Vaoikun/Data_Acquisition_Blog].

## Conclusion
Open telemetry datasets such as OpenF1 provide a valuable opportunity to explore motorsport performance data using modern data science tools. Even simple statistical measures such as lap consistency and best lap time can reveal meaningful insights about driver performance.

As Formula 1 continues to expand its data availability, opportunities for deeper analytics will only grow.
