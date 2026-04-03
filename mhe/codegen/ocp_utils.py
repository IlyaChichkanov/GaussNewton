import os

import numpy as np
from acados_template import AcadosModel
from casadi import bilin, mtimes


def is_discrete(model: AcadosModel):
    try:
        return model.disc_dyn_expr != []
    except (AttributeError, RuntimeError):
        return False


def quadform(matrix: np.array, x):
    Q_squared = mtimes([matrix.T, matrix])
    return bilin(Q_squared, x)


def generate_header(file, definitions, define_name):
    """Generates a C header file with the given definitions."""
    directory = os.path.dirname(file)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    with open(file, 'w') as f:
        f.write(f'#ifndef {define_name} \n')
        f.write(f'#define {define_name}\n\n')
        for name, value in definitions.items():
            f.write(f"#define {name} {value}\n")
        f.write("\n#endif\n")
    print(f'generate_header {file}')
