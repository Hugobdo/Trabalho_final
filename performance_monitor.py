from mongo_connector import MongoData
import matplotlib.pyplot as plt

all_data = MongoData('EURUSD').read_all_data()
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(all_data['close'])
plt.show()
