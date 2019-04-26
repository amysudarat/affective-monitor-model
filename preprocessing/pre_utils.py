# -*- coding: utf-8 -*-

def match_with_sample(original_df,filtered_idx):
    # check if index of arousal_df is in filtered_idx list or not
    original_df = original_df.reset_index(drop=True)
    output = original_df[original_df.index.isin(filtered_idx.tolist())]
    return output.reset_index(drop=True)
