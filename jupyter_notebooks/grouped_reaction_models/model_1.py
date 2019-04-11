# exported from PySB model 'model'

from pysb import Model, Monomer, Parameter, Expression, Compartment, Rule, Observable, Initial, MatchOnce, Annotation, ANY, WILD

Model()

Monomer('A', ['B'])
Monomer('B', ['A', 'C'])
Monomer('C', ['B'])
Monomer('D', ['E'])
Monomer('E', ['D'])

Parameter('inhibition_0_A_inhibitor_B_inh_target_2kf_0', 1.5e-05)
Parameter('inhibition_0_A_inhibitor_B_inh_target_1kr_0', 0.00012)
Parameter('inhibition_0_B_inhibitor_C_inh_target_2kf_0', 5e-05)
Parameter('inhibition_0_B_inhibitor_C_inh_target_1kr_0', 6e-06)
Parameter('inhibition_0_D_inhibitor_E_inh_target_2kf_0', 5e-06)
Parameter('inhibition_0_D_inhibitor_E_inh_target_1kr_0', 6e-06)
Parameter('A_0', 200000.0)
Parameter('B_0', 50000.0)
Parameter('C_0', 20000.0)
Parameter('D_0', 10000.0)
Parameter('E_0', 10000.0)

Observable('A_obs', A())
Observable('B_obs', B())
Observable('C_obs', C())
Observable('D_obs', D())
Observable('E_obs', E())

Rule('inhibition_0_A_inhibitor_B_inh_target', A(B=None) + B(A=None, C=None) | A(B=1) % B(A=1, C=None), inhibition_0_A_inhibitor_B_inh_target_2kf_0, inhibition_0_A_inhibitor_B_inh_target_1kr_0)
Rule('inhibition_0_B_inhibitor_C_inh_target', B(A=None, C=None) + C(B=None) | B(A=None, C=1) % C(B=1), inhibition_0_B_inhibitor_C_inh_target_2kf_0, inhibition_0_B_inhibitor_C_inh_target_1kr_0)
Rule('inhibition_0_D_inhibitor_E_inh_target', D(E=None) + E(D=None) | D(E=1) % E(D=1), inhibition_0_D_inhibitor_E_inh_target_2kf_0, inhibition_0_D_inhibitor_E_inh_target_1kr_0)

Initial(A(B=None), A_0)
Initial(B(A=None, C=None), B_0)
Initial(C(B=None), C_0)
Initial(D(E=None), D_0)
Initial(E(D=None), E_0)

Observable('AB_complex',A(B=1)%B(A=1))