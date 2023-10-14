import numpy as np
import matplotlib.pyplot as plt 

# Initialize a dictionary to store data
data = {'SNo': [], 'Company': [], 'Daily': [], 'Sunday': []}

# Read data from a TSV file
with open("C:\\RC\\Documents\\IIT Patna Remastered 2023-25\\Course\\1st Semester\\CS561 Artificial Intelligence\\Assignments\\Assignment 7 Linear Regression\\newspaper_data.tsv", 'r') as file:
    file.readline()  # Skip the header
    for line in file:
        parts = line.strip().split('\t')
        if len(parts) == 4:
            data['SNo'].append(int(parts[0]))
            data['Company'].append(parts[1])
            data['Daily'].append(float(parts[2]))
            data['Sunday'].append(float(parts[3]))
        elif len(parts) == 3:
            data['SNo'].append(int(parts[0]))
            newspaper, daily = parts[1].rsplit(' ', 1)
            data['Company'].append(newspaper)
            data['Daily'].append(float(daily))
            data['Sunday'].append(float(parts[2]))

# Extract relevant columns into NumPy arrays
daily_sales = np.array(data['Daily'])
sunday_circulation = np.array(data['Sunday'])

# Add a column of ones to X for the intercept term in linear regression
X = daily_sales
Y = sunday_circulation.reshape(X.size, 1) 
X = np.vstack((np.ones((X.size, )), X)).T

print(X.shape)
print(Y.shape)

def model(X, Y, learning_rate, iteration):
    m = Y.size
    theta = np.zeros((2, 1))
    cost_list = []

    for i in range(iteration):
        y_pred = np.dot(X, theta)
        cost = (1/(2*m)) * np.sum(np.square(y_pred - Y))
        d_theta = (1/m) * np.dot(X.T, y_pred - Y)
        theta = theta - learning_rate * d_theta
        cost_list.append(cost)

    # Plot the best fit line along with the scatter plot
    plt.plot(X[:, 1], np.dot(X, theta), label="Best Fit Line", color='red')
    plt.legend()
    plt.xlabel('Daily Sales')
    plt.ylabel('Sunday Circulation')
    plt.title('Scatter Plot with Best Fit Line')
    plt.scatter(X[:, 1], Y)
    plt.show()

    return theta, cost_list

iteration = 100
learning_rate = 0.000005
theta, cost_list = model(X, Y, learning_rate=learning_rate, iteration=iteration)

# Define scenarios (minimum, maximum, average daily sales) for prediction
newspapers = np.array([[1, np.min(daily_sales)], [1, np.max(daily_sales)], [1, np.mean(daily_sales)]])

# Calculate predicted Sunday circulation for each scenario
predictions = np.dot(newspapers, theta)

# Create lists to store companies that should stop Sunday edition for each scenario
companies_to_stop_min = []
companies_to_stop_max = []
companies_to_stop_mean = []

for i, newspaper_sales in enumerate(newspapers):
    if i == 0:
        case = "Minimum Case"
    elif i == 1:
        case = "Maximum Case"
    elif i == 2:
        case = "Average Case"

    sunday_prediction = round(predictions[i][0], 2)
    print(f"{case} Sunday Prediction: ", sunday_prediction)
    print()

    # Compare each company's Sunday sales to the predicted Sunday sales for the scenario
    for j in range(len(data['Company'])):
        company = data['Company'][j]
        daily_sales = data['Daily'][j]

        # Determine if a company should stop Sunday edition based on the scenario
        if daily_sales * 1.3 < sunday_prediction:
            if i == 0:
                companies_to_stop_min.append(company)
            elif i == 1:
                companies_to_stop_max.append(company)
            elif i == 2:
                companies_to_stop_mean.append(company)

# Print companies to stop Sunday edition for each case
print("Companies to stop based on Minimum Case:")
for index, company in enumerate(companies_to_stop_min):
    print(f"{index + 1}. {company}")

print("\nCompanies to stop based on Maximum Case:")
for index, company in enumerate(companies_to_stop_max):
    print(f"{index + 1}. {company}")

print("\nCompanies to stop based on Average Case:")
for index, company in enumerate(companies_to_stop_mean):
    print(f"{index + 1}. {company}")
