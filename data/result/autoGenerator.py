import sys
import pandas as pd
import numpy as np
# print(sys.argv)

filepath = sys.argv[1]
outputPath = sys.argv[2]

prediction_from_file = np.load(filepath)
true_df = pd.DataFrame(prediction_from_file)
true_df = true_df.rename(columns={
    0: 'Predicted'
})
true_df['Id'] = true_df.index + 1
cols = true_df.columns.tolist()
cols = cols[-1:] + cols[:-1]
true_df = true_df[cols]
true_df.to_csv(sys.argv[2], sep=',', index=False)