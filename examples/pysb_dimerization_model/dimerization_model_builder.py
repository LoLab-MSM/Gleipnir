'''
Dimerization model:
 A + A <> AA
'''
from pysb.builder import Builder
from scipy.stats import uniform, norm
import numpy as np

builder = Builder()

#######
V = 100.
#######
builder.parameter('kf', 0.001, prior=uniform(loc=np.log10(0.001)-1., scale=2.))
builder.parameter('kr', 1., prior=norm(loc=np.log10(1.), scale=2.))

builder.monomer('A', ['d'])

# Rules
builder.rule('ReversibleBinding',
             builder['A'](d=None) + builder['A'](d=None) | builder['A'](d=1) % builder['A'](d=1),
             builder['kf'], builder['kr'])


#Observables
builder.observable("A_dimer", builder['A'](d=1) % builder['A'](d=1))

# Inital Conditions
builder.parameter("A_0", 20.*V)
builder.initial(builder['A'](d=None), builder['A_0'])
