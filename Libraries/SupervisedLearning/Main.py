#Supervisied Learning
#cLassificaion in which we are predicting we are doing whole number
#Either True or false
#like cat or dog

#Regression in which we are prediciting are a float number in a percentage
#We would be predicting continuous values 
#The predicition involves the range for didtirbuting the data that the oil prices would surge 30%

#Un supervised learning
#Clustering
#We do not have output in it in which we have to find the hidden data from the dataset

#Optimization
#In this we study tehcniques for optimization

#dimenisioanity deduction
#if the data has alot of features where we reduce and convert it into a situation where params are less
#with minimal data loss
#if we have a dataset where we record the temparature in degree and farenheight we remove one since both are correlated
#if we are doing flower's classification
#if we collect a data for pettel lenght and sample length and we get to konow
# both are correltated so we reduce the params for it

#Scatter graph 
#Via this we get to know which feature is defined for which
#In machine learning, a scatter graph (or scatter plot) is a fundamental 
# visualization tool used to represent the relationship between two numerical variables. 
# Each data point in a set is represented by a dot on a two-dimensional Cartesian plane,
# with the position determined by its values on the x and y axes

#lets say we gota organize a dataset for all the students and we have aprams for the 
#amoutn of marks for fsc and matric and then the realtive subjecrts
#then we will distirbute based on params on the scatter grpah that what posistion will be defined 
#Decision boundary is developed if it is a striaght line then it is reffered as linear problem
#basically it involves that if a scatter graph with 2 params have the categorized in a viibsle dimenson with 
#x and y axis meaning there ar e2 categories students above 50 or below then viisible shit  is seen 
#but if there involves a greater params it becomes non linear
#y = mx + c 

#Reninforcement Learning
#Lets say u moved into a house and u have unintorudced data where u do not know where the buttons are for the lights
#and fans if the houseowener tells where what is then that is unsupervisied learning
#if we do it our selves then we find it ourselves lets say u turn on the button of fan in winter
# not disirerable and u get a negaitve reward u do not do that and this is unsupervisied learning
#if state that is changed accordingy to env and a

#Visualtion of 4D-->2D On iris dataset
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.datasets import load_iris, load_digits
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler
# from mpl_toolkits.mplot3d import Axes3D

# iris = load_iris()
# X = iris.data
# y = iris.target

# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# pca = PCA(n_components=2)
# X_reduced = pca.fit_transform(X_scaled)

# fig = plt.figure(figsize=(14, 10))

# ax1 = fig.add_subplot(221, projection='3d')
# colors = ['red', 'green', 'blue']
# for i in range(3):
#     ax1.scatter(X[y==i, 0], X[y==i, 1], X[y==i, 2], c=colors[i], label=iris.target_names[i], s=40)
# ax1.set_xlabel('Sepal Length')
# ax1.set_ylabel('Sepal Width')
# ax1.set_zlabel('Petal Length')
# ax1.set_title('4D: 3D plot + color = petal width')

# ax2 = fig.add_subplot(222)
# corr = np.corrcoef(X.T)
# im = ax2.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
# ax2.set_xticks(range(4))
# ax2.set_yticks(range(4))
# ax2.set_xticklabels(['SL', 'SW', 'PL', 'PW'])
# ax2.set_yticklabels(['SL', 'SW', 'PL', 'PW'])
# ax2.set_title('4D Correlation Matrix')
# plt.colorbar(im, ax=ax2)

# ax3 = fig.add_subplot(223)
# for i in range(3):
#     ax3.scatter(X_reduced[y==i, 0], X_reduced[y==i, 1], c=colors[i], label=iris.target_names[i], alpha=0.7, s=50)
# ax3.set_xlabel('Principal Component 1')
# ax3.set_ylabel('Principal Component 2')
# ax3.set_title(f'4D → 2D (PCA) - {sum(pca.explained_variance_ratio_):.1%} variance')
# ax3.legend()

# digits = load_digits()
# X_digits = digits.data
# y_digits = digits.target

# scaler_digits = StandardScaler()
# X_digits_scaled = scaler_digits.fit_transform(X_digits)

# pca_digits = PCA(n_components=2)
# X_digits_reduced = pca_digits.fit_transform(X_digits_scaled)

# ax4 = fig.add_subplot(224)
# scatter = ax4.scatter(X_digits_reduced[:, 0], X_digits_reduced[:, 1], c=y_digits, cmap='tab10', alpha=0.6, s=15)
# ax4.set_xlabel('Principal Component 1')
# ax4.set_ylabel('Principal Component 2')
# ax4.set_title(f'Digits: 64D → 2D - {sum(pca_digits.explained_variance_ratio_):.1%} variance')
# plt.colorbar(scatter, ax=ax4)

# plt.tight_layout()
# plt.show()

#Neuron baiscally sum the inputs with weights passes through an acitvation function
#if the dimaensiton are increased it becomes latent space we train a NeuralNetwork such that
#the hidden layers reduce the shit such that the upcoming outputs layers get data that is more in linear form 
#so that it is easy to understand basically neural network and deeplearning difference is more hidden and input and output layer
#we do feature extraction we got iris flowers we get their heights by creating thier shit manually and creating the dataset
#scaling involves we got admission in fsc total marks are of 1100 and we got a student from afghasistan that has marks of total of 3k
#for this we calculate the percentage to find a relation where we distribute the shit into a scale a basic range

#Visualization of Non-linear Dataset
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.datasets import make_classification, make_moons, make_circles

# fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# X1, y1 = make_classification(n_samples=200, n_features=2, n_redundant=0, n_clusters_per_class=1, random_state=42)
# axes[0].scatter(X1[:,0], X1[:,1], c=y1, cmap='coolwarm', s=50, edgecolors='black')
# axes[0].axline([0, 0], [1, 1], color='black', linewidth=3, label='ONE STRAIGHT LINE')
# axes[0].set_title('LINEAR DATA\nOne straight line separates everything', fontsize=12)
# axes[0].legend()

# X2, y2 = make_moons(n_samples=200, noise=0.1, random_state=42)
# axes[1].scatter(X2[:,0], X2[:,1], c=y2, cmap='coolwarm', s=50, edgecolors='black')
# axes[1].set_title('NON-LINEAR DATA\nLook at the SHAPE - it curves like a moon', fontsize=12)
# axes[1].text(-1.5, 0.5, 'CANNOT DRAW\nONE STRAIGHT LINE', fontsize=10, color='red', weight='bold', ha='center')

# X3, y3 = make_circles(n_samples=200, noise=0.05, factor=0.5, random_state=42)
# axes[2].scatter(X3[:,0], X3[:,1], c=y3, cmap='coolwarm', s=50, edgecolors='black')
# axes[2].set_title('NON-LINEAR DATA\nOne class is INSIDE the other', fontsize=12)
# axes[2].text(0, 0, 'RED INSIDE BLUE\nNeed a CIRCLE to separate', fontsize=10, color='red', weight='bold', ha='center')

# plt.tight_layout()
# plt.show()

#Examples of Non-Linear
# Credit card fraud detection - Legitimate transactions cluster in a normal pattern,
# but fraudulent transactions are scattered randomly throughout the data space with 
# no straight line separating them from normal ones.

# Speech recognition - The same word spoken by different people creates interwoven 
# patterns in acoustic space that cannot be separated by a straight line because 
# of accents, pitch, and speed variations.

#KNN (K-Nearest Neighbors)
#in this we feind the nearest neighbours
#If we have a student we have to send him to a competiion 
#before this we have send students and we know which succeeded and all and we classify them based on their english
#and maths marks and we have students which are bad at both and all
#Now a new student has no background data and we have his maths and english maths or whatever data we have 
#that what he excels in and based on that shit we classify and send him to the nearest shit 
#lets say a person with both good is classified into speech competitions and both bad is send to sports competition
#so we classify the new person based upon the new data we have on him defiining the nearest neighbour we have on him
#based on the scatter plot we have we classify him and get the nearest data
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import accuracy_score
# from sklearn.decomposition import PCA

# iris = load_iris()
# X = iris.data
# y = iris.target

# pca = PCA(n_components=2)
# X_2d = pca.fit_transform(X)

# X_train, X_test, y_train, y_test = train_test_split(X_2d, y, test_size=0.2, random_state=42)

# knn = KNeighborsClassifier(n_neighbors=5)
# knn.fit(X_train, y_train)

# y_pred = knn.predict(X_test)

# print(f"KNN Accuracy on Iris: {accuracy_score(y_test, y_pred):.2%}")

# h = 0.02
# x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
# y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
# xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
# Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
# Z = Z.reshape(xx.shape)

# plt.figure(figsize=(12, 5))

# plt.subplot(121)
# plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
# plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='viridis', edgecolors='black', s=50)
# plt.xlabel('Principal Component 1')
# plt.ylabel('Principal Component 2')
# plt.title(f'KNN on Iris Dataset (k=5)\nAccuracy: {accuracy_score(y_test, y_pred):.1%}')

# plt.subplot(122)
# for i in range(len(y_test)):
#     color = 'green' if y_test[i] == y_pred[i] else 'red'
#     plt.scatter(i, y_test[i], c='blue', s=80, marker='o', alpha=0.6)
#     plt.scatter(i, y_pred[i], c=color, s=40, marker='x')
# plt.xlabel('Test Sample Index')
# plt.ylabel('Class (0=Setosa, 1=Versicolor, 2=Virginica)')
# plt.title('Blue=Actual, Green X=Correct, Red X=Wrong')
# plt.ylim(-0.5, 2.5)

# plt.tight_layout()
# plt.show()
# The code loads the Iris dataset (150 samples, 4 features), then uses PCA to reduce it to 2 dimensions 
# so you can visualize it. It splits into train/test, applies KNN with k=5, and plots:

# Left plot: The colored background shows the decision boundary 
# how KNN classifies every possible point in the 2D space based on nearest neighbors.
# The actual Iris flowers are overlaid as dots (color = species).

# Right plot: Shows each test sample - blue dots are true species
# (0=Setosa, 1=Versicolor, 2=Virginica), green X means KNN predicted correctly,
# red X means it was wrong.

#Limitiations
#Lets say we have a group of 10 people with income of 10k and 11th person with 100M
#then a new person has to be predicted that involves his income to be 100k
#then logically that person would be said to also have a ferrari cause his salry is more inclided to his 
#nerighost neighbour which is 1M since the scatterplot plots the 10k and 1M to extreme edges 
#but in reality classifying him with 1M category would not be right so for this we plot 
# more then one nieghbours And introduce randomization

#Euclidean Distance vs Manhatten Distance
#Euclidean distance measures the straight-line distance between two points in multi-dimensional space.
# Calculated using the Pythagorean theorem, it computes the square root of summed squared differences 
# across all dimensions. For points A(x₁, y₁) and B(x₂, y₂), distance = √[(x₂-x₁)² + (y₂-y₁)²].
# KNN uses this metric to find nearest neighbors.
#Manhattan distance measures the absolute differences between coordinates, summing them without squaring.
# Unlike Euclidean's straight-line "as the crow flies," Manhattan follows grid-like paths (like city blocks).
# Formula: |x₂-x₁| + |y₂-y₁|. It's less sensitive to outliers than Euclidean and works well for high-dimensional sparse data.

# #visualtization
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.datasets import load_iris
# from sklearn.decomposition import PCA
# from sklearn.model_selection import train_test_split
# from sklearn.neighbors import KNeighborsClassifier
# from scipy.spatial.distance import euclidean, cityblock

# iris = load_iris()
# X = iris.data
# y = iris.target

# pca = PCA(n_components=2)
# X_2d = pca.fit_transform(X)

# X_train, X_test, y_train, y_test = train_test_split(X_2d, y, test_size=0.2, random_state=42)

# test_point = X_test[0]

# euclidean_distances = []
# manhattan_distances = []

# for train_point in X_train:
#     euc = euclidean(test_point, train_point)
#     man = cityblock(test_point, train_point)
#     euclidean_distances.append(euc)
#     manhattan_distances.append(man)

# euclidean_neighbors = np.argsort(euclidean_distances)[:3]
# manhattan_neighbors = np.argsort(manhattan_distances)[:3]

# fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# colors = ['red', 'green', 'blue']
# for class_idx in range(3):
#     mask = y_train == class_idx
#     axes[0].scatter(X_train[mask, 0], X_train[mask, 1], c=colors[class_idx], 
#                    label=iris.target_names[class_idx], alpha=0.6, s=80)
#     axes[1].scatter(X_train[mask, 0], X_train[mask, 1], c=colors[class_idx], 
#                    alpha=0.6, s=80)

# axes[0].scatter(test_point[0], test_point[1], c='black', s=200, marker='*', label='Test Point')
# axes[1].scatter(test_point[0], test_point[1], c='black', s=200, marker='*')

# for idx in euclidean_neighbors:
#     axes[0].plot([test_point[0], X_train[idx, 0]], [test_point[1], X_train[idx, 1]], 'k--', alpha=0.5)
#     axes[0].scatter(X_train[idx, 0], X_train[idx, 1], c='yellow', s=150, edgecolors='black')

# for idx in manhattan_neighbors:
#     axes[1].plot([test_point[0], X_train[idx, 0]], [test_point[1], X_train[idx, 1]], 'r--', alpha=0.5)
#     axes[1].scatter(X_train[idx, 0], X_train[idx, 1], c='yellow', s=150, edgecolors='black')

# axes[0].set_title(f'EUCLIDEAN DISTANCE\nStraight line "as crow flies"\nPredicts: {iris.target_names[y_train[euclidean_neighbors[0]]]}')
# axes[1].set_title(f'MANHATTAN DISTANCE\nCity block "L-shaped" path\nPredicts: {iris.target_names[y_train[manhattan_neighbors[0]]]}')
# axes[0].legend()
# plt.tight_layout()
# plt.show()

# print("="*60)
# print(f"Test point actual class: {iris.target_names[y_test[0]]}")
# print("="*60)
# print("\nEUCLIDEAN")
# for i, idx in enumerate(euclidean_neighbors):
#     print(f"  Neighbor {i+1}: Distance = {euclidean_distances[idx]:.4f}, Class = {iris.target_names[y_train[idx]]}")

# print("\nMANHATTAN :")
# for i, idx in enumerate(manhattan_neighbors):
#     print(f"  Neighbor {i+1}: Distance = {manhattan_distances[idx]:.4f}, Class = {iris.target_names[y_train[idx]]}")

# print("Euclidean = √[(x₂-x₁)² + (y₂-y₁)²] - diagonal paths allowed")
# print("Manhattan = |x₂-x₁| + |y₂-y₁| - only horizontal/vertical movement")
#The code loads Iris data, reduces to 2D with PCA, and selects one test point.
# It calculates Euclidean (straight-line) and Manhattan (grid-based) distances to all training points,
# finds the 3 nearest neighbors for each metric, then visualizes the different paths and predictions
# each distance metric produces.