import pandas as pd

tags = pd.read_excel("corr.xlsx", sheet_name='corr', index_col=0)
id = pd.read_excel("DWH1.xlsx", sheet_name='ren_tags2', index_col=0)

uniq_values = tags["value2"].unique()

tags["tag_id"] = 0

for idx in range(len(uniq_values)):
    df_bool = id.index[id["value"] == uniq_values[idx]]

    ind_bool = df_bool[0]
    tags.loc[tags['value2'] == uniq_values[idx], "tag_id"] = id["tag_id"][ind_bool]

path_to_file = "DWH1.xlsx"
tags_ = pd.read_excel(path_to_file, sheet_name='ren_tags2', index_col=0)
news_by_tag1 = pd.read_excel(path_to_file, sheet_name='ren_news_by_tag', index_col=0)
news_by_tag2 = pd.read_excel(path_to_file, sheet_name='ren_news_by_tag2', index_col=0)

news_by_tag = pd.concat([news_by_tag1, news_by_tag2])
news_by_tag.reset_index(drop=True, inplace=True)

tags_['popularity'] = 0
