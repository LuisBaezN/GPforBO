from math import sin, cos
from math import pi
from numpy import arange
from numpy import vstack
from numpy import argmax, argmin
from numpy import asarray
from numpy.random import normal
import random
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from warnings import catch_warnings
from warnings import simplefilter
from matplotlib import pyplot

# Función real
def objective(x, noise=0.9):
	noise = normal(loc=0, scale=noise)#noise=0.1
	#Funcion real para max: (x**2 * sin(5 * pi * x)**6.0) + noise
	return ((-5.1/(4*3.1416)**2)*x+(5/3.1416)*x-6)**2 + 10*(1-1/(8*3.1416))*cos(x) + 10 + noise

# Función sustituta
def surrogate(model, X):
	with catch_warnings():
		simplefilter("ignore")
		return model.predict(X, return_std=True)

# Función de adquisición: Mejora esperada (EI)
def acquisition(X, Xsamples, model):
	# calcula el mejor resultaado hasta ahora:
	yhat, _ = surrogate(model, X)
	best = min(yhat) #antes: max
	# calcula la media y la desviación estandar mediante la función sustituta
	mu, std = surrogate(model, Xsamples)
	print(f'> mu: {mu}\n> std: {std}')
	# calcula la probabilidad de mejora
	probs = norm.cdf((mu - best) / (std+1E-9)) #Orig:, mu + (1+7)*3

# optimiza la función de adquisición
def opt_acquisition(X, y, model):
	# Búsqueda aleatoria, genera muestras aleatorias
	Xsamples = sample_floats(0, 10, points) #Xsamples = random(100)
	Xsamples = asarray(Xsamples)
	Xsamples = Xsamples.reshape(len(Xsamples), 1)
	# calcula a AF para cada muestra
	scores = acquisition(X, Xsamples, model)
	# localiza el índice con el mejor resultado
	ix = argmin(scores)#antes: ix = argmax(scores) 
	return Xsamples[ix, 0]

# Grafica las observaciones realies vs la función sustituta
def plot(X, y, model):
	# scatter plot de la función real
	pyplot.scatter(X, y)
	# line plot de la función sustituta
	Xsamples = asarray(arange(0, 10, 0.1)) #Antes:Xsamples = asarray(arange(0, 1, 0.001))
	Xsamples = Xsamples.reshape(len(Xsamples), 1)
	ysamples, _ = surrogate(model, Xsamples)
	pyplot.plot(Xsamples, ysamples)
	pyplot.show()

def sample_floats(low, high, k=1):
    result = []
    seen = set()
    for i in range(k):
        x = random.uniform(low, high)
        while x in seen:
            x = random.uniform(low, high)
        seen.add(x)
        result.append(x)
    return result


if __name__ == '__main__':

	random.seed(7)

	points = 10
	opt_iter = 100

	X = sample_floats(0, 10, points) #Antes: X = random(100)
	X = asarray(X)

	y = asarray([objective(x) for x in X])

	X = X.reshape(len(X), 1)
	y = y.reshape(len(y), 1)

	model = GaussianProcessRegressor()
	model.fit(X, y)
	plot(X, y, model)

	# Optimizacion
	for i in range(opt_iter):
		x = opt_acquisition(X, y, model)
		actual = objective(x)
		est, _ = surrogate(model, [[x]])
		print('> x=%.3f, f()=%3f, actual=%.3f' % (x, est, actual))
		
		# Agregar al dataset
		X = vstack((X, [[x]]))
		y = vstack((y, [[actual]]))
		
		# actualiza el modelo
		model.fit(X, y)

	# Grafica todas las miestras de la función sustituta final
	plot(X, y, model)

	ix = argmin(y)
	print('> Mejor resultado: x=%.3f, y=%.3f' % (X[ix], y[ix]))