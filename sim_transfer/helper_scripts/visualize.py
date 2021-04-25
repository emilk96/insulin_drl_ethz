import matplotlib.pyplot as plt
import pandas as pd
import datetime



raw_data_file = 'child#008'

raw_data_path = 'results/test/'
df = pd.read_csv(raw_data_path + raw_data_file + '.csv')
print(df.head())

fig, axs = plt.subplots(3, figsize=(10,15))
fig.suptitle('Stats ' + raw_data_file, fontweight="bold", fontsize =20)

axs[0].plot(df['Time'], df['BG'], label='BG')
axs[0].plot(df['Time'], df['CGM'])
axs[0].axhline(y=70, color='r', linestyle='-')
axs[0].axhline(y=180, color='r', linestyle='-')
axs[0].set_title('Blood Glucose model output/ cgm measurement')
axs[0].set_ylabel('BG [mg/dL]')

axs[1].plot(df['Time'], df['insulin'])
axs[1].set_title('Insulin Dosage')
axs[1].set_ylabel('Insulin [UI]')

axs[2].scatter(df['Time'], df['CHO'])
axs[2].set_title('Carbohydrate Intake')
axs[2].set_ylabel('CHO [g]')

for i in range (3):
    ticklist = []
    every_nth = 120
    for n, label in enumerate(axs[i].axes.get_xticks()):
        if n % every_nth == 0:
            ticklist.append(label)
    print(ticklist)
    axs[i].axes.get_xaxis().set_ticks(ticklist)

plt.savefig('pid_plot.png')




# positions = 7,8,9,10,11,12,113
# labels = ("7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","01","02","03","04","05","06")
# plt.xticks(positions, labels)

# for i in range (len(df)):
#     val = df['Time'][0]
#     val = val[10:16]
#     print (val)