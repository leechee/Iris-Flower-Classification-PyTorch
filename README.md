# Iris-Flower-Classification-PyTorch
![[Iris Flowers]|100](assets/iris_flowers.png)

The Iris flower data set or Fisher's Iris data set is a multivariate data set that contains 50 samples from each of three species of Iris (Iris Setosa, Iris Virginica, and Iris Versicolor).

The [data set](IRIS.csv) contains a set of 150 records under 5 attributes:
- Petal Length, Petal Width, Sepal Length, Sepal width, and Species

## Objective

In this project, I created a basic neural network using PyTorch to classify the different specices of Iris. The model is a multilayer perceptron with two hidden layers, the first one has 128 neurons, and the second
has 64 neurons. The activation function is ReLU, and PyTorch.nn are implemented.

## Getting Started
### Python Environment
Download and install Python 3.8 or higher from the [official Python website](https://www.python.org/downloads/)

Optional, but I would recommend creating a venv. For Windows installation:
```
py -m venv .venv
.venv\Scripts\activate
```
For Unix/macOS:
```
python3 -m venv .venv
source .venv/bin/activate
```

Now install the necessary AI stack in the venv terminal. These libraries will aid with computational coding, data visualization, accuracy reports, preprocessing, etc. I used pip for this project.
```
pip install numPy
pip install matplotlib
pip install pandas
pip install scikit-learn
```

For Torch:
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Data Input
To input data from the Iris data set, make sure to replace ... with your correct file path. Use the pandas library:
```
data = pd.read_csv('...\IRIS.csv')
```

### Results (100% Training and Testing Accuracy)
![[plot]](assets/plot.png)
