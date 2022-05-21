from math import sin, cos
from math import pi
from numpy import arange
from numpy import vstack
from numpy import argmax, argmin
from numpy import asarray
from numpy.random import normal
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from warnings import catch_warnings
from warnings import simplefilter
from matplotlib import pyplot
import random
import math

random.seed(3)

primitives = ['+', '-', '*', '/']
terminals_s = ['m','v']
terminals_c = ['1', '3', '7']

def generate_pt(min=4, max=17):
    #lenght of tree:
    deep_ops = random.randint(min,max)

    if deep_ops%2 == 0:
        term = deep_ops
        prim = deep_ops - 1
    else:
        term = deep_ops + 1
        prim = deep_ops
    
    p = []
    t = terminals_s
    for i in range(prim):            
        p.append(random.choice(primitives))

    for i in range(term - 3):
        t.append(random.choice(terminals_c))

    return p, t

def trees(pr, te, n = 10):
    parents = []
    for i in range(n):
        random.shuffle(pr)
        random.shuffle(te)
        temp = []
        for j in range(len(pr)-1):
            temp.append(te[j])
            temp.append(pr[j])
        temp.append(te[j+1])       

        izq = []
        for j in range(len(temp)//5):
            ind = random.randint(0,len(temp)-5+j)
            while(ind % 2 != 0):
                ind = random.randint(0,len(temp)-5+j)
            izq.append(ind)
        izq.sort()
        izq.reverse()
        
        der = []
        for j in izq:
            ind = random.randint(j+3,len(temp))
            while(ind % 2 == 0):
                ind = random.randint(j+3,len(temp))
            #if (ind+1)
            der.append(ind)
        
        #print(izq, der)

        cont = 0
        for j in range(0,len(izq)):
            #print(j)
            temp.insert(izq[j], '(')
            if (temp[der[j]+cont] == '('):
                if (temp[der[j]+cont] == '+' or temp[der[j]+cont] == '-' or temp[der[j]+cont] == '*' or temp[der[j]+cont] == '/'):
                    temp.insert(der[j]+3+cont, ')')
                else:
                    temp.insert(der[j]+2+cont, ')')
            else:
                if (temp[der[j]+cont] == '+' or temp[der[j]+cont] == '-' or temp[der[j]+cont] == '*' or temp[der[j]+cont] == '/'):
                    if (temp[der[j]+cont] == '('):
                        temp.insert(der[j]+3+cont, ')')
                    else:
                        temp.insert(der[j]+2+cont, ')')
                else:
                    temp.insert(der[j]+1+cont, ')')
            #print(temp)
            cont = cont + 1
        
        for j in range(len(temp)):
            if (temp[j] == '(' and temp[j+1] == ')'):
                
                if (temp[j] == '(' and temp[j+2] == ')'):
                    elem = temp[j+3]
                    temp.pop(j+1)
                    temp.pop(j+2)                    
                    temp.insert(j+1, elem)
                    temp.insert(j+2, ')')
                else:
                
                    temp.pop(j+1)
                    temp.insert(j+2, ')')
        

        #print(temp)    
        parents.append(temp)
    return parents

def evaluate(l):

    scores_t = []
    for i in range(len(l)):

        scores = []
        points = 10
        opt_iter = 37

        X = sample_floats(0, 10, points) #Antes: X = random(100)
        X = asarray(X)

        y = asarray([objective(x) for x in X])

        X = X.reshape(len(X), 1)
        y = y.reshape(len(y), 1)

        model = GaussianProcessRegressor()
        model.fit(X, y)
        #plot(X, y, model)

        # Optimizacion
        for j in range(opt_iter):
	        x = opt_acquisition(X, y, model, l[i])
	        actual = objective(x)
	        est, _ = surrogate(model, [[x]])
	        #print('>x=%.3f, f()=%3f, actual=%.3f' % (x, est, actual))
	
	        # Agregar al dataset
	        X = vstack((X, [[x]]))
	        y = vstack((y, [[actual]]))
	
	        # actualiza el modelo
	        model.fit(X, y)

        # Grafica todas las miestras de la función sustituta final
        #plot(X, y, model)

        ix = argmin(y)
        #print('>> Mejor resultado: x=%.3f, y=%.3f' % (X[ix], y[ix])) 

        err = abs(y[ix])

        scores.append(err)
        scores.append(i)
        #print('Error cuadratico medio: ', errcm)
        scores_t.append(scores)

    
    return scores_t    

def mate(parents, scores):
    merge_sort(scores, 0, len(scores)-1)
    #print(scores)
    porcent = int(len(scores)*0.2)
    if (porcent%2 != 0):
        porcent = porcent + 1

    indiv = []
    for i in range(porcent):
        indiv.append(scores[i][1])
    #print(indiv)


    indiv_red = indiv
    new_sons = []
    for i in  range(int(len(indiv)/2)): 
        partind_1 = indiv_red[0] 
        partind_2 = random.choice(indiv_red)
        while(partind_1 == partind_2):
            partind_2 = random.choice(indiv_red)
        #print(len(indiv_red))
        for j in range(len(indiv_red)):
            #print(j)
            if (indiv_red[j] == partind_2):
                #print(indiv_red[0], indiv_red[j])
                indiv_red.pop(j)
                indiv_red.pop(0)
                break

        partner_1 = parents[partind_1]
        partner_2 = parents[partind_2]
        #print(partner_1,partner_2)

        cont_1 = 0
        cont_2 = 0
        for j in range(len(partner_1)):
            if (partner_1[j] == '('):
                cont_1 = cont_1 + 1
        for j in range(len(partner_2)):
            if (partner_2[j] == '('):
                cont_2 = cont_2 + 1

        ind_i_1, ind_f_1, inter_1 = gen_sub_tree(partner_1,cont_1)
        ind_i_2, ind_f_2, inter_2 = gen_sub_tree(partner_2,cont_2)

        new_s = comb(ind_i_2, ind_f_2, inter_1, partner_2)  
        new_s = verif_p(new_s)
        #print(new_s)
        new_sons.append(new_s)

        new_s = comb(ind_i_1, ind_f_1, inter_2, partner_1)  
        new_s = verif_p(new_s)
        #print(new_s)
        new_sons.append(new_s)
    
    #print(new_sons)
    #print(len(new_sons))
    return new_sons

def comb(ind_i, ind_d, inter_1, part_2):
    new_son = []
    #print(part_2)
    for i in range(ind_i+1):
        new_son.append(part_2[i])
    new_son.extend(inter_1)
    for i in range(ind_d, len(part_2)):
        new_son.append(part_2[i])
    #print(new_son)
    return new_son


def gen_sub_tree(partner,cont_e):

    parti = random.randint(1,cont_e)
    cont = 0

    ind_izq_prim = 0 #REVISAR AQUII!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ind_i = 0 #REVISAR AQUII!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    #print(partner) 

    for j in range(len(partner)): 
        #print('Ciclo "(": ', partner[j])
        if partner[j] == ')':
            cont = cont + 1
            if cont == parti:
                ind_f = j
                cont = 0
                for k in range(j,0,-1):
                    #print('Ciclo ")": ',partner[k])
                    if partner[k] == '(':
                        cont = cont + 1
                        if cont == parti and cont == 1:
                            ind_i = k
                            ind_izq_prim = k
                            #print('entro 1')
                        elif cont == 1:
                            ind_izq_prim = k #REVISAR AQUII!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                            ind_i = k
                            #print('entro 2')
                        elif cont == parti:
                            ind_i = k
                            #ind_izq_prim = k #REVISAR AQUII!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                            #print('entro 3')
                            

                cont = 0        
                for k in range(j,0,-1):
                    if partner[k] == ')':
                        cont = cont + 1
                        if cont == 1:
                            ind_der_prim = k
                
                if (ind_izq_prim > ind_der_prim):
                    elec = random.randint(1,2)
                    if elec == 2:
                        ind_i = ind_izq_prim

    #print(partner)
    #print('Indice inicial: %.3f. Indice final: %.3f' % (ind_i, ind_f))

    inter = []
    for i in range(ind_i,ind_f+1):
        inter.append(partner[i])
    
    inter = verif_p(inter)

    #print(inter)

    return ind_i, ind_f, inter

def verif_p(formul):
    cont_1 = 0
    cont_2 = 0

    for i in formul:
        if i == '(':
            cont_1 = cont_1 + 1
        elif i == ')':
            cont_2 = cont_2 + 1
    
    #print(formul)
    #print(cont_1, cont_2)
    longit = len(formul)

    if cont_1 < cont_2:
        dif = cont_2 - cont_1
        cont_aux = 0
        for i in range(longit):
            if dif != cont_aux:
                #print('Longitud: %.3f. Indice: %.3f. Contador: %.3f' % (longit, i, cont_aux))
                #print('formul ', formul)
                if formul[i-cont_aux] == ')':
                    formul.pop(i-cont_aux)
                    cont_aux = cont_aux + 1
            elif cont_aux == dif:
                break

    
    elif cont_2 < cont_1:
        dif = cont_1 - cont_2
        cont_aux = 0
        for i in range(len(formul)-1,0,-1):
            if dif != cont_aux:
                #print('formul ', formul)
                #print('Longitud: %.3f. Indice: %.3f. Contador: %.3f' % (longit, i, cont_aux))
                if formul[i] == '(':
                    formul.pop(i)
                    cont_aux = cont_aux + 1
                    #print('formul ', formul)
            elif cont_aux == dif:
                break
    
    return formul
            


def merge_sort(array, left_index, right_index):
    if left_index >= right_index:
        return

    middle = (left_index + right_index)//2
    merge_sort(array, left_index, middle)
    merge_sort(array, middle + 1, right_index)
    merge(array, left_index, right_index, middle)

def merge(array, left_index, right_index, middle):
   
    left_copy = array[left_index:middle + 1]
    right_copy = array[middle+1:right_index+1]

    left_copy_index = 0
    right_copy_index = 0
    sorted_index = left_index

    while left_copy_index < len(left_copy) and right_copy_index < len(right_copy):

        if left_copy[left_copy_index] <= right_copy[right_copy_index]:
            array[sorted_index] = left_copy[left_copy_index]
            left_copy_index = left_copy_index + 1
        else:
            array[sorted_index] = right_copy[right_copy_index]
            right_copy_index = right_copy_index + 1

        sorted_index = sorted_index + 1

    while left_copy_index < len(left_copy):
        array[sorted_index] = left_copy[left_copy_index]
        left_copy_index = left_copy_index + 1
        sorted_index = sorted_index + 1

    while right_copy_index < len(right_copy):
        array[sorted_index] = right_copy[right_copy_index]
        right_copy_index = right_copy_index + 1
        sorted_index = sorted_index + 1

def reemplaz(parents, score, new_sons):
    #print(score)
    inds = []
    cont = 0
    for i in range(len(parents)-1,0,-1):
        inds.append(score[i][1])
        cont = cont + 1
        if cont == len(new_sons):
            break
    
    #print(inds)
    inds.sort()
    #print(inds)
    inds.reverse()
    #print(inds)

    #print(len(parents))
    for i in inds:
        parents.pop(i)
    
    #print(len(parents))

    parents.extend(new_sons)

    #print(len(parents))

    return parents

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
def acquisition(X, Xsamples, model, t_func):
	# calcula el mejor resultaado hasta ahora:
	yhat, _ = surrogate(model, X)
	best = min(yhat) #antes: max
	# calcula la media y la desviación estandar mediante la función sustituta
	m, v = surrogate(model, Xsamples)
	m = m[:, 0]
	stri = ''.join(t_func)
	try:
		AF = eval(stri)
	except ZeroDivisionError:
		AF = 100

	probs = norm.cdf(AF)
	return probs

def opt_acquisition(X, y, model, t_func):
	points = 10
    # Búsqueda aleatoria, genera muestras aleatorias
	Xsamples = sample_floats(0, 10, points) #Xsamples = random(100)
	Xsamples = asarray(Xsamples)
	Xsamples = Xsamples.reshape(len(Xsamples), 1)
	# calcula a AF para cada muestra
	scores = acquisition(X, Xsamples, model, t_func)
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

#def visual(funcs, scor)

if __name__ == "__main__":
    p, t = generate_pt()
    
    print('Primitivas: ', p[:])    
    print('Terminales: ', t[:])

    parents = trees(p, t, 100)

    sco = evaluate(parents)
    new_sons = mate(parents,sco)

    parents = reemplaz(parents, sco, new_sons)



    gen = 0
    for i in range(3): #31
    #while (sco[0][0] > 0.01):

        gen = i + 1 
        #gen = gen + 1

        print('>>>Generacion:', gen)
        sco = evaluate(parents)

        print('>>>Precision: ', sco[0][0])

        new_sons = mate(parents,sco)

        parents = reemplaz(parents, sco, new_sons)
        #print(len(parents[i][1]))



    print(sco[0][0])
    print('\n>>Mejor solucion: ', ''.join(parents[sco[0][1]]))