import numpy             as np
import matplotlib.pyplot as plt
import globals           as gb
from os                  import path

YEAR_DELTA = 1975 # In csv-file years are subtracted by 1975.
data       = np.loadtxt(path.join(gb.DATASET_DIR, 'housing_price_index.csv'), delimiter=',')
X, y       = data[:, 0], data[:, 1]

# Plot the data in the matrix housing_price_index.
plt.scatter(X, y)
plt.show(block=False)

# Try to fit (and plot using subplot(2,2,i)) all polynomial models f (X) = β0 + β1X + β2X2 +
# . . . + βdXd for degrees d ∈ [1, 4]. Which polynomial degree do you think gives the best fit?
# Motivate your answer!
X_e = np.ones((X.shape[0], 1))
for year in range(1, 5):
    X_e = np.c_[X_e, X**year]
    beta = gb.normal_equation(X_e, y)
    plt.subplot(2, 2, year)
    plt.title(f"Polynomial degree: {year}")
    plt.xlabel("Year")
    plt.ylabel("Housing price index")
    plt.plot(X + YEAR_DELTA, X_e.dot(beta))
    plt.scatter(X + YEAR_DELTA, y, marker='.')
plt.show(block=False)

# Jonas Nordqvist bought in 2015 a house in Växjö for 2.3 million SEK. What can he expect
# to get, using your “best fit model”, for his house when he (after completing his PhD) sells
# his house in 2022 (to start his new career as a data scientist in Stockholm)? Is your answer
# realistic?
X_2022 = X
for year in range(2018, 2023):
    X_2022 = np.append(X_2022, year-YEAR_DELTA)
X_2022_e = np.c_[np.ones((X_2022.shape[0], 1)), X_2022, X_2022**2, X_2022**3, X_2022**4]
hpi_2015 = 568                               # Actual house price index from csv.
hpi_2022 = X_2022_e.dot(beta)[2022-YEAR_DELTA]
## Index calculations formula source: https://www.scb.se/vara-tjanster/scbs-olika-index/att-anvanda-index-i-avtal/#R%C3%A4kna
print(f"Jonas Nordqvist can expect to sell his house for {int((hpi_2022/hpi_2015)*2300000)} year 2022.")
print("The answer is very realistic. (Source: My step dad).")

plt.show()