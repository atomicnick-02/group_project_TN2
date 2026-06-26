def normalize_actions(actions, a_min, a_max):
    """Linearly maps raw actions to a strict [-1.0, 1.0] workspace distribution."""
    return 2.0 * (actions - a_min) / (a_max - a_min + 1e-8) - 1.0

def unnormalize_actions(norm_actions, a_min, a_max):
    """Maps [-1.0, 1.0] model outputs back to raw physical values."""
    return (norm_actions + 1.0) * 0.5 * (a_max - a_min) + a_min

class NormalizationResults:
    norm_results = {
        "stack_d0" : {"ACTION_MIN": [-1.0, -1.0, -1.0, -1.0, -0.4951956135538213, -1.0, -1.0],
                      "ACTION_MAX": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]},
        "stack_d1" : {"ACTION_MIN": [-1.0, -1.0, -1.0, -0.34770733375298263, -0.5624572016862676, -1.0, -1.0],
                      "ACTION_MAX": [1.0, 1.0, 1.0, 0.44141399894147143, 0.4706261107692711, 1.0, 1.0]},
        "stack_three_d0": {"ACTION_MIN": [-1.0, -1.0, -1.0, -0.9341467246612831, -1.0, -1.0, -1.0],
                           "ACTION_MAX": [1.0, 1.0, 1.0, 1.0, 0.52750831708172, 1.0, 1.0]},
        "stack_three_d1": {"ACTION_MIN": [-1.0, -1.0, -1.0, -1.0, -0.6376954317092896, -1.0, -1.0],
                           "ACTION_MAX": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]}
    }