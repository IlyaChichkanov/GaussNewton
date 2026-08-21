# Code style

The rules this repository follows, so that the explanations stay in `docs/` and
do not grow back into the source.

## Language

Everything in the source is **English**: comments, docstrings, exception
messages, `print` output, assert messages. The theory notebooks stay in Russian
— they are the derivations, not the code.

## Docstrings

- **Module**: one to three lines — what lives here and, if it needs an
  explanation, a link to the relevant page under `docs/`. No history, no
  benchmark numbers, no derivations.
- **Public function or class**: one line saying what it does. Add an
  `Args`/`Returns` block only where array shapes or units are not obvious from
  the names — integrator contracts, `ShootRows`, `SensitivityTrajectory`.
- **Private helper**: a docstring only when the name is not enough.
- **Test**: one line naming the property under test and, where there is one, the
  external reference it is checked against.

Everything else — why an alternative was rejected, measured timings, formulas,
refactor history — belongs in `docs/` or a notebook, reached by a
`see docs/<page>.md` pointer.

## Comments

Reserved for the non-obvious *why*: numerical tricks, CasADi/JAX quirks, sign
conventions, cache-key hazards. A comment that restates the code is noise.

## Naming

One quantity, one name — [notation.md](notation.md) is the register. A second
name appearing for something that already has one is a signal that a layer
boundary was crossed.

Public names are load-bearing: notebooks under `experiments/` and `mpc/` import
them, so renaming one is a separate, deliberate change. This includes the
misspelled package `commom_utils`.

## Changing numbers

`pytests/regression_test.py` freezes the numbers of a Gauss–Newton step. Any
change that moves them must be deliberate and explainable; see
[testing.md](testing.md).
