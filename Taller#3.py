
"""
Spyder Editor
This is a temporary script file.
"""
 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from gapminder import gapminder
import random

s = 0
while s != 1:
    #MENU
    print("1. Guardar gapminder")
    print("2. Leer el archivo gapminder.xlsx")
    print("3. Diagrama de dispersión lifeExp vs pop")
    print("4. Diagrama de dispersión gdpPercap vs pop.")
    print("5. diagramas de cajas de la variable gdpPercap discriminados por continentes desde 1990 a 2007.")
    print("6. Salir")
    
    jpr = int(input("Escoja una opción: " ))
    if jpr == 1:
        
        gapminder.to_excel('gapminder.xlsx', sheet_name='hoja 1', index = False)
    elif jpr == 2:
        
        df_nuevo = pd.read_excel("gapminder.xlsx", sheet_name="hoja 1")
        sz = df_nuevo["country"].size
        prcntg = round(sz*0.1)
        sq = list(np.arange(0,sz))
        indices = random.sample(sq, prcntg)
        df_nuevo.loc[indices, ["lifeEXP","pop", "gdpPercap"]] = np.nan
       
        print(df_nuevo)
    elif jpr == 3:
        
        fig, ax = plt.subplots()
        ax.scatter(df_nuevo["lifeExp"], df_nuevo["pop"] )
        plt.show()
    elif jpr == 4:
        
        fig, ax = plt.subplots()
        ax.scatter(df_nuevo["gdpPercap"], df_nuevo["pop"] )
        plt.show()
    elif jpr == 5:
       
        df = df_nuevo[(df_nuevo['year']>1990) & (df_nuevo['year']<2007)]
        df2 = df[["gdpPercap", "continent"]]
        df_filtered = df2[df2["gdpPercap"].notna()]
        africa = df_filtered[(df_filtered['continent'] == 'Africa')]
        africa = africa["gdpPercap"]
        asia = df_filtered[(df_filtered['continent'] == 'Asia')]
        asia = asia["gdpPercap"]
        americas = df_filtered[(df_filtered['continent'] == 'Americas')]
        americas = americas["gdpPercap"]
        europe = df_filtered[(df_filtered['continent'] == 'Europe')]
        europe = europe["gdpPercap"]
        oceania = df_filtered[(df_filtered['continent'] == 'Oceania')]
        oceania = oceania["gdpPercap"]
        cont = [africa, asia, americas, europe, oceania]
        val1 = [1, 2, 3, 4, 5]
        val2 = ['africa', 'asia', 'americas', 'europe', 'oceania']
        plt.boxplot(cont)
        plt.xticks(val1, val2)
        plt.show()
    elif jpr == 6:
        s = 1
    else:
        print("Daniel")
        
        
        



import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

n = int(input("tamaño: "))

print("Experimento A")
mu_a = int(input("Media: "))
std_a = int(input("Desviación estandar: "))
a = np.random.normal(loc = mu_a, scale = std_a , size = n)
exp_a = pd.DataFrame(a)
exp_a.to_csv("Experimento_a.csv", index= False)
Experimento_a = pd.read_csv("Experimento_a.csv" )

print("Experimento B")
mu_b = int(input("Media: "))
std_b = int(input("Desviación estandar: "))
b = np.random.normal(loc = mu_b, scale = std_b , size = n)
exp_b = pd.DataFrame(b)
exp_b.to_csv("Experimento_b.csv", index= False)
Experimento_b = pd.read_csv("Experimento_b.csv" )
print("------------------------------------------------------------------------------")



s = 0
while s!= 1:
    print("1. indicar si la diferencia en la media de los datos es estadísticamente significativa.")
    print("2. mostrar en pantalla la correlación de Pearson y Spearman de los datos.")
    print("3. graficar el diagrama de dispersión y la línea recta que aproxime los datos calculada por una regresión lineal por mínimos cuadrados.")
    print("4. Salir")
    jpr = int(input("Escoja una opción: "))
    if jpr == 1:
        ph = stats.ttest_ind(Experimento_a["0"], Experimento_b["0"])
        pv = round(ph[1],3)
        if pv > 0.05:
            print("-------------------------------------------------------")
            print(f"El p valor, {pv}, es mayor que 0.05 por lo que la diferencia entre medias no es estadisticamente significativa" )
            print("-------------------------------------------------------")
        elif pv <= 0.05:
            print("-------------------------------------------------------")
            print(f"El p valor, {pv}, es menor o igual que 0.05 por lo que la diferencia entre medias es estadisticamente significativa" )    
            print("-------------------------------------------------------")    
    #--------------------------------------------------------------------
    elif jpr == 2:
        print("-------------------------------------------------------")
        print("Correlación pearson: ", Experimento_a["0"].corr(Experimento_b["0"], method = 'pearson'))
        print("Correlación spearman: ", Experimento_a["0"].corr(Experimento_b["0"], method = 'spearman'))
        print("-------------------------------------------------------")
    #--------------------------------------------------------------------
    elif jpr == 3:
        datos = {'a' : Experimento_a["0"], 'b' : Experimento_b["0"]}
        exps = pd.DataFrame(datos)  
        sb.lmplot(x = 'a', y = 'b', data = exps)
    elif jpr == 4:
        s = 1
    else:
        print("Daniel")        
        


import numpy as np
from scipy import stats
from scipy.stats import expon
import matplotlib.pyplot as plt


var = np.linspace(expon.ppf(0.06),
                expon.ppf(0.60), 100)
plt.plot(var, expon.pdf(var))
plt.title("Distribución Exponencial")
plt.ylabel("probabilidad")
plt.xlabel("valores")
plt.show()


variable =  2.6 
poisson = stats.poisson(variable) 
variable = np.arange(poisson.ppf(0.04),
              poisson.ppf(0.55))
fmp = poisson.pmf(variable) 
plt.plot(variable, fmp, '--')
plt.vlines(variable, 0, fmp, colors='b', lw=6, alpha=0.10)
plt.title("Distribución Poisson")
plt.ylabel("probabilidad")
plt.xlabel("valores")
plt.show()


x =  0.2 
bernoulli = stats.bernoulli(x)
variable = np.arange(-6, 8)
fmp = bernoulli.pmf(variable) 
fig, ax = plt.subplots()
ax.plot(variable, fmp, 'bo')
ax.vlines(variable, 0, fmp, colors='b', lw=8, alpha=0.4)
ax.set_yticks([0., 0.2, 0.8, 0.14])
plt.title("Distribución Bernoulli")
plt.ylabel("probabilidad")
plt.xlabel("valores")
plt.show()


uniforme = stats.uniform()
variable = np.linspace(uniforme.ppf(0.12),
                uniforme.ppf(0.40), 100)
dt = uniforme.pdf(variable) 
fig, ax = plt.subplots()
ax.plot(variable, dt, '--')
ax.vlines(variable, 0, dt, colors='b', lw=5, alpha=0.4)
ax.set_yticks([0., 0.4, 0.8, 1.2, 1.6, 2.0, 2.4])
plt.title("Distribución Uniforme")
plt.ylabel("probabilidad")
plt.xlabel("valores")
plt.show()
