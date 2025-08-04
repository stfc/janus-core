from __future__ import annotations

import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium")


@app.cell
def _():
    from ase.build import bulk
    from ase.io import read, write
    import marimo as mo
    from pymatgen.io.ase import AseAtomsAdaptor
    import pymatviz as view

    from janus_core.calculations.geom_opt import GeomOpt

    return AseAtomsAdaptor, GeomOpt, bulk, mo, read, view, write


@app.cell
def _(bulk, write):
    device = "cpu"
    a = 6.0
    NaCl = bulk(
        "NaCl", crystalstructure="rocksalt", cubic=True, orthorhombic=True, a=a
    ) * (2, 2, 2)
    NaCl.rattle(stdev=0.1, seed=2042)
    write("NaCl.extxyz", NaCl)
    return (NaCl,)


@app.cell
def _(AseAtomsAdaptor, NaCl, view):
    struct = AseAtomsAdaptor().get_structure(NaCl)

    s_widget = view.StructureWidget(
        structure=struct, show_bonds=True, bonding_strategy="nearest_neighbor"
    )
    s_widget
    return


@app.cell
def _(mo):
    # Create a form with multiple elements
    form = (
        mo.md("""
        **Geomtry Optimisation**

        Upload Structure {structure}

        optimize {optimize}

        {fmax}

        ML model :{model}
    """)
        .batch(
            structure=mo.ui.file(label="Structure"),
            fmax=mo.ui.number(label="fmax", value=0.01),
            optimize=mo.ui.radio(
                options={"Just coordinates": 1, "Cell vectors only": 2, "Full": 3},
                value="Full",  # initial value
                label="choose a methood",
            ),
            model=mo.ui.dropdown(
                options={
                    "MACE_MP-small": ("mace_mp", "small"),
                    "MACE_MP-medium": ("mace_mp", "medium"),
                },
                value="MACE_MP-small",
            ),
        )
        .form(
            show_clear_button=True,
            bordered=True,
            submit_button_disabled=False,
            submit_button_tooltip="start the calculation",
        )
    )
    form
    return (form,)


@app.cell
def _(form):
    settings = form.value
    print(settings)
    return (settings,)


@app.cell
def _(GeomOpt, form, mo, read, settings):
    mo.stop(form.value is None, mo.md("Upload a file and submit"))

    from io import BytesIO

    x = settings["structure"][0]
    m = read(BytesIO(x.contents), format="cif")

    model = settings["model"]
    optimized_NaCl = GeomOpt(
        struct=m,
        model=model[1],
        arch=model[0],
        fmax=settings["fmax"],
        optimizer="FIRE",
        write_traj=True,
    )

    optimized_NaCl.run()
    return (m,)


@app.cell
def _(m):
    m
    return


if __name__ == "__main__":
    app.run()
