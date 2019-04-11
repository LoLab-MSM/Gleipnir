# exported from PySB model 'model'

from pysb import Model, Monomer, Parameter, Expression, Compartment, Rule, Observable, Initial, MatchOnce, Annotation, ANY, WILD

Model()

Monomer('A', ['B', 'C'])
Monomer('B', ['A', 'D'])
Monomer('C', ['A'])
Monomer('D', ['B'])

Parameter('inhibition_0_A_inhibitor_B_inh_target_2kf_0', 1.5e-05)
Parameter('inhibition_0_A_inhibitor_B_inh_target_1kr_0', 0.00012)
Parameter('inhibition_0_A_inhibitor_C_inh_target_2kf_0', 4e-05)
Parameter('inhibition_0_A_inhibitor_C_inh_target_1kr_0', 7e-06)
Parameter('inhibition_0_B_inhibitor_D_inh_target_2kf_0', 1e-05)
Parameter('inhibition_0_B_inhibitor_D_inh_target_1kr_0', 6e-06)
Parameter('A_0', 200000.0)
Parameter('B_0', 50000.0)
Parameter('C_0', 20000.0)
Parameter('D_0', 10000.0)

Observable('A_obs', A())
Observable('B_obs', B())
Observable('C_obs', C())
Observable('D_obs', D())

Rule('inhibition_0_A_inhibitor_B_inh_target', A(B=None, C=None) + B(A=None, D=None) | A(B=1, C=None) % B(A=1, D=None), inhibition_0_A_inhibitor_B_inh_target_2kf_0, inhibition_0_A_inhibitor_B_inh_target_1kr_0)
Rule('inhibition_0_A_inhibitor_C_inh_target', A(B=None, C=None) + C(A=None) | A(B=None, C=1) % C(A=1), inhibition_0_A_inhibitor_C_inh_target_2kf_0, inhibition_0_A_inhibitor_C_inh_target_1kr_0)
Rule('inhibition_0_B_inhibitor_D_inh_target', B(A=None, D=None) + D(B=None) | B(A=None, D=1) % D(B=1), inhibition_0_B_inhibitor_D_inh_target_2kf_0, inhibition_0_B_inhibitor_D_inh_target_1kr_0)

Initial(A(B=None, C=None), A_0)
Initial(B(A=None, D=None), B_0)
Initial(C(A=None), C_0)
Initial(D(B=None), D_0)

Observable('AB_complex',A(B=1)%B(A=1))