
# Lazy Imports (Avoids Circular Imports)
def get_euler_bernoulli():
    from pre_processing.element_library.euler_bernoulli.euler_bernoulli import EulerBernoulliBeamElement
    return EulerBernoulliBeamElement

def get_timoshenko():
    from pre_processing.element_library.timoshenko.timoshenko import TimoshenkoBeamElement
    return TimoshenkoBeamElement
