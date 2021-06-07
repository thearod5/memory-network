import os

import pandas as pd
from sklearn.model_selection import train_test_split

from constants import PATH_TO_CONVERSATION_DATA

convos_cols = ["hash_value", "subreddit_name", "c_id", "response score", "turn", "context", "response"]
facts_cols = ["hash_value", "subreddit_name", "c_id", "domain", "fact"]


def get_grouped_data(path_to_conversation_folder, file_name):
    c_file_name = f"{file_name}.convos.txt"
    f_file_name = f"{file_name}.facts.txt"

    path_to_c = os.path.join(path_to_conversation_folder, c_file_name)
    path_to_f = os.path.join(path_to_conversation_folder, f_file_name)

    convos_df = pd.read_csv(path_to_c, sep="\t", error_bad_lines=False).dropna()
    convos_df.columns = convos_cols

    facts_df = pd.read_csv(path_to_f, sep="\t", error_bad_lines=False).dropna()
    facts_df.columns = facts_cols

    facts_df["a_facts"] = facts_df.groupby("c_id")["fact"].transform(lambda x: ' '.join(x))
    grouped_facts_df = facts_df[["c_id", "a_facts"]].drop_duplicates().reset_index(drop=True)

    data_df = convos_df.merge(grouped_facts_df, left_on="c_id", right_on="c_id")[
        ["c_id", "a_facts", "context", "response"]]
    data_df.columns = ["id", "story", "query", "response"]
    return data_df


def get_conversation_data():
    train_df = read_conversation_df(os.path.join(PATH_TO_CONVERSATION_DATA, "train.csv"))
    test_df = read_conversation_df(os.path.join(PATH_TO_CONVERSATION_DATA, "test.csv"))

    return train_df, test_df


def get_sample_conversation_data():
    path_to_train = os.path.join(PATH_TO_CONVERSATION_DATA, "sample_train.csv")
    path_to_test = os.path.join(PATH_TO_CONVERSATION_DATA, "sample_test.csv")
    train_df = read_conversation_df(path_to_train)
    test_df = read_conversation_df(path_to_test)
    return train_df, test_df


def read_conversation_df(path_to_df):
    data_df = pd.read_csv(path_to_df)
    stories = data_df["story"]
    queries = data_df['query']
    responses = data_df["response"]
    return zip(stories, queries, responses)


def create_aggregate_conversation():
    files = os.listdir(PATH_TO_CONVERSATION_DATA)
    excluded_files = [".DS_Store", "train.csv", "test.csv", "sample_test.csv", "sample_train.csv", ".create"]
    names = pd.Series([f.split(".")[0] for f in files if f not in excluded_files])

    agg_df = None
    for n in names.unique():
        try:
            grouped_df = get_grouped_data(PATH_TO_CONVERSATION_DATA, n)
            agg_df = grouped_df if agg_df is None else pd.concat([grouped_df, agg_df])
        except Exception as e:
            print(f"Failed:{n}")
            raise e

        agg_df.to_csv(os.path.join(PATH_TO_CONVERSATION_DATA, "conversations.csv"), index=False)
    print("done")


if __name__ == "__main__":
    create_aggregate_conversation()
    df = pd.read_csv(os.path.join(PATH_TO_CONVERSATION_DATA, "conversations.csv"))
    train_df, test_df = train_test_split(df, shuffle=True)
    train_df.to_csv(os.path.join(PATH_TO_CONVERSATION_DATA, "train.csv"), index=False)
    test_df.to_csv(os.path.join(PATH_TO_CONVERSATION_DATA, "test.csv"), index=False)
