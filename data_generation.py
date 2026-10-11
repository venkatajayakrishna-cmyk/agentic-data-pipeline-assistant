import pandas as pd, numpy as np

df = pd.read_csv("data/covid_19_data.csv").sample(5000, random_state=1)
df.loc[df.sample(300).index, "Confirmed"] = np.nan                      # missing values
df.loc[df.sample(200).index, "ObservationDate"] = "not a date"          # bad dates
df.loc[df.sample(150).index, "Country/Region"] = "  " + df["Country/Region"] + "  "  # stray spaces
df = pd.concat([df, df.sample(100)])                                    # duplicates
df.to_csv("data/messy_covid.csv", index=False)