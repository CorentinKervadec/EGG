import matplotlib.pyplot as plt

sample_all = [[i+j for j in range(20)] for i in range(20)]
sample_all = [item for sublist in sample_all for item in sublist] # flatten the list
print(sample_all)
n, bins, patches = plt.hist(sample_all, list(range(40)), density=True, facecolor='g', alpha=0.75)

print('n', n)
print('bins', bins)

plt.xlabel('Sum')
plt.ylabel('Probability')
plt.title('Histogram of the sum (N=20)')
plt.xlim(0, 40)
plt.grid(True)
plt.savefig('hist_sum_20.png')

