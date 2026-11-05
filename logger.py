import pandas as pd
from datetime import datetime
import os


def log_interaction(question, context, answer):

    log_entry = {
        "timestamp": datetime.now(),
        "question": question,
        "context": context[:1000],
        "answer": answer
    }

    filename = "support_log.csv"

    if os.path.exists(filename):
        df = pd.read_csv(filename)
        df = pd.concat(
            [df, pd.DataFrame([log_entry])],
            ignore_index=True
        )
    else:
        df = pd.DataFrame([log_entry])

    df.to_csv(filename, index=False)

    print("Interaction logged.")