import pandas as pd

df = pd.read_parquet("train-00001.parquet")

for label in sorted(df["label"].unique()):
    print("\n" + "=" * 80)
    print("LABEL:", label)
    print("=" * 80)

    samples = df[df["label"] == label].head(5)

    for _, row in samples.iterrows():
        print("\nQuestion:")
        print(row["question"])

        print("\nReference:")
        print(row["reference_answer"])

        print("\nStudent:")
        print(row["student_answer"])