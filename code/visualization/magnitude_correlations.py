import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
output_path = "correlation_matrix.png"

def plot_two_cols(dataframe, row_1, row_2):

    plt.clf()
    row_1_list = dataframe[row_1].tolist()
    row_2_list = dataframe[row_2].tolist()
    
    plt.scatter(row_1_list, row_2_list)

    plt.xlabel(row_1)
    plt.ylabel(row_2)

    output_path = "plots/grads/" + row_1 + "_" + row_2 + ".png"
    plt.savefig(output_path, dpi=400)


def plot_col_over_time(dataframe, row_1):

    plt.clf()
    row_1_list = dataframe[row_1].tolist()
    row_2_list = range(0, len(dataframe)*3, 3)
    
    plt.scatter(row_2_list, row_1_list, s=1)

    plt.ylabel(row_1)
    plt.xlabel("sample_number")

    output_path = "plots/grads/time_" + row_1 + ".png"
    plt.savefig(output_path, dpi=400)

data = pd.read_csv('mags.csv')
data = data.drop(['image_name'], axis=1)
cols = data.columns.tolist()
cols = cols[1:3] + cols[4:8] + cols[3:4] + cols[8:] + cols[0:1]
data = data[cols]
data = data.iloc[::3, :]

# for col_1 in data.columns:
#     for col_2 in data.columns:
#         if col_1 != col_2:
#             plot_two_cols(data, col_1, col_2)
            # print(col_1, col_2)
            
print(len(data))

for col_1 in data.columns:
    plot_col_over_time(data, col_1)

# plot_two_cols(data, 'conf', 'correct')
# plot_two_cols(data, 'layers_-1', 'layer_1')

# fig = plt.figure()
# ax1 = fig.add_subplot()
# ax1 = data.plot.scatter(x='conf', y='correct')




# plt.savefig("conf_correct.png", dpi=400)


# print(data)

# print(data)
# corr = data.corr()
# print(corr)

# fig = plt.figure()
# ax = fig.add_subplot(111)
# cax = ax.matshow(corr,cmap='coolwarm', vmin=-1, vmax=1)
# fig.colorbar(cax)
# ticks = np.arange(0,len(data.columns),1)
# ax.set_xticks(ticks)
# plt.xticks(rotation=90)
# ax.set_yticks(ticks)
# ax.set_xticklabels(data.columns)
# ax.set_yticklabels(data.columns)
# plt.savefig(output_path, dpi=400)