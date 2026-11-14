import marimo

__generated_with = "0.25.2"
app = marimo.App(
    width="medium",
    app_title="costplan: how much will my agent cost?",
)


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
async def _(mo):
    import sys

    if sys.platform == "emscripten":
        # In the browser, install the library from the wheel published with this page.
        # wheel.txt holds the wheel's file name, so a new version needs no change here.
        import micropip
        from pyodide.http import pyfetch

        _public = mo.notebook_location() / "public"
        _wheel = (await (await pyfetch(str(_public / "wheel.txt"))).string()).strip()
        await micropip.install(str(_public / _wheel))
    import costplan as cp
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.figsize": (7, 3.2),
            "figure.dpi": 110,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "font.size": 10,
        }
    )
    INK, ACCENT, MUTED = "#1f2937", "#0f766e", "#9ca3af"
    return ACCENT, MUTED, cp, plt


@app.cell
def _(mo):
    mo.md("""
    # How much will my agent cost?

    Plan an agent's budget from a handful of pilot runs. Most runs are cheap; a few loop, retry or
    read long documents and cost many times more. That makes averages slow to settle and makes a
    budget cap a question about the rare expensive runs. Everything runs in your browser, so nothing
    is sent anywhere. The first load takes a few seconds while Python starts up.

    1. [How many pilot runs do I need?](#pilot)
    2. [What do my runs cost, and what cap should I set?](#runs)
    3. [What will it cost at scale?](#spend)
    4. [Is the cheaper model cheaper per task?](#compare)

    The same calculations are in the Python library: `pip install costplan`.
    """)
    return


@app.cell
def _(mo):
    pl_stop = mo.ui.slider(0.005, 0.10, step=0.005, value=0.02, label="Share of normal runs the cap may stop", show_value=True)
    pl_conf = mo.ui.dropdown({"90%": 0.90, "95%": 0.95, "99%": 0.99}, value="95%", label="Confidence")
    pl_spread = mo.ui.slider(0.2, 3.0, step=0.1, value=1.0, label="Cost spread (std dev / average)", show_value=True)
    pl_margin = mo.ui.slider(0.02, 0.30, step=0.01, value=0.10, label="Average within plus or minus", show_value=True)
    return pl_conf, pl_margin, pl_spread, pl_stop


@app.cell
def _(ACCENT, MUTED, cp, mo, pl_conf, pl_margin, pl_spread, pl_stop, plt):
    _stop, _conf = pl_stop.value, pl_conf.value
    _need_cap = cp.runs_needed_for_cap(_stop, _conf)
    _need_avg = cp.pilot_runs_for_average(pl_spread.value, pl_margin.value, _conf)
    _ns = list(range(10, 401, 5))
    _fig, _ax = plt.subplots()
    _ax.plot(_ns, [100 * cp.guaranteed_stop_rate(n, _conf) for n in _ns], color=ACCENT, linewidth=2)
    _ax.axhline(100 * _stop, color=MUTED, linestyle="--")
    _ax.axvline(_need_cap, color=MUTED, linestyle=":")
    _ax.set_xlabel("Pilot runs")
    _ax.set_ylabel("Best promise (% of runs stopped)")
    _ax.set_ylim(0, 25)
    _fig.tight_layout()
    mo.vstack(
        [
            mo.md(
                """
                <h2 id="pilot">1. How many pilot runs do I need?</h2>

                A budget cap should almost never stop a normal run. Promising that takes more pilot
                runs than people expect, because the promise is about the rare expensive runs. Set
                the cap at the most expensive pilot run, and the chart shows the largest share of
                normal runs it might stop.
                """
            ),
            mo.hstack([pl_stop, pl_conf], wrap=True),
            mo.hstack(
                [
                    mo.stat(f"{_need_cap:,}", label="Pilot runs for a safe cap", caption=f"stops at most {_stop:.1%} of runs", bordered=True),
                    mo.stat(f"{100 * cp.guaranteed_stop_rate(30, _conf):.1f}%", label="Best promise with 30 runs", caption="share of runs a cap might stop", bordered=True),
                ],
                justify="start",
                wrap=True,
            ),
            _fig,
            mo.md("To pin down the **average** cost instead, the spread of costs decides the pilot size:"),
            mo.hstack([pl_spread, pl_margin], wrap=True),
            mo.hstack(
                [mo.stat(f"{_need_avg:,}", label="Pilot runs for the average", caption=f"within plus or minus {pl_margin.value:.0%}", bordered=True)],
                justify="start",
                wrap=True,
            ),
        ]
    )
    return


@app.cell
def _(mo):
    _example = [
        (0.07435, 1), (0.13482, 1), (0.17612, 1), (0.09204, 1), (0.08579, 1), (0.10099, 1), (0.1298, 1), (0.08936, 0),
        (0.03505, 1), (0.07808, 0), (0.34679, 1), (0.19913, 1), (0.27365, 0), (0.0544, 0), (0.19205, 0), (0.12568, 1),
        (0.09802, 1), (0.16278, 1), (0.3262, 0), (0.18754, 1), (0.09427, 1), (0.03137, 1), (0.06131, 1), (0.17072, 1),
        (0.05858, 1), (0.10863, 1), (1.42671, 0), (0.11433, 0), (0.08321, 1), (0.0937, 1), (0.16308, 1), (0.09513, 1),
        (0.0544, 1), (0.18464, 1), (0.11507, 1), (0.24641, 1), (0.07438, 1), (0.05154, 1), (0.2092, 1), (0.03676, 1),
        (0.08299, 1), (0.05469, 1), (0.18494, 1), (0.04687, 1), (0.10699, 1), (0.13711, 1), (0.02775, 0), (0.21641, 1),
        (0.14919, 1), (0.86747, 1), (0.17767, 1), (0.1264, 1), (0.11646, 1), (0.05424, 1), (0.11132, 1), (0.04413, 1),
        (0.0926, 1), (0.18189, 1), (0.1945, 1), (0.05065, 1),
    ]
    runs_text = mo.ui.text_area(
        value="\n".join(f"{c}, {s}" for c, s in _example),
        label="One run per line: cost in dollars, then 1 or 0 for success (optional)",
        rows=8,
        full_width=True,
    )
    runs_stop = mo.ui.slider(0.01, 0.10, step=0.01, value=0.02, label="Share of normal runs the cap may stop", show_value=True)
    return runs_stop, runs_text


@app.cell
def _(runs_text):
    def _parse(text):
        costs, flags = [], []
        for number, line in enumerate(text.strip().splitlines(), start=1):
            parts = [p.strip() for p in line.replace("\t", ",").split(",") if p.strip()]
            if not parts:
                continue
            costs.append(float(parts[0].lstrip("$")))
            if len(parts) > 1:
                flags.append(parts[1].lower() in {"1", "true", "yes", "pass", "success"})
        return costs, (flags if flags and len(flags) == len(costs) else None)

    try:
        run_costs, run_flags = _parse(runs_text.value)
        parse_error = None if len(run_costs) >= 2 else "Enter at least two runs."
    except ValueError:
        run_costs, run_flags, parse_error = [], None, "Each line should start with a number, for example: 0.12, 1"
    return parse_error, run_costs, run_flags


@app.cell
def _(
    ACCENT,
    MUTED,
    cp,
    mo,
    parse_error,
    plt,
    run_costs,
    run_flags,
    runs_stop,
    runs_text,
):
    _header = mo.md(
        """
        <h2 id="runs">2. What do my runs cost, and what cap should I set?</h2>

        Paste the cost of each pilot run, from your tracing tool or your provider's usage logs. The
        box starts with 60 simulated runs of a support agent, where a few runs got stuck in loops.
        """
    )
    if parse_error:
        _body = mo.callout(mo.md(parse_error), kind="warn")
    else:
        _s = cp.summarize(run_costs, run_flags)
        _cap = cp.recommend_cap(run_costs, runs_stop.value)
        _cards = [
            mo.stat(f"${_s.mean:.3f}", label="Average per run", caption=f"95% interval ${_s.mean_interval[0]:.3f} to ${_s.mean_interval[1]:.3f}", bordered=True),
            mo.stat(f"${_s.median:.3f}", label="Typical run", caption="the median", bordered=True),
            mo.stat(f"${_s.max:.2f}", label="Most expensive run", caption=f"{_s.max / _s.median:.0f} times the typical run", bordered=True),
        ]
        if _s.cost_per_success is not None:
            _cards.append(
                mo.stat(f"${_s.cost_per_success:.3f}", label="Per successful task", caption=f"success rate {_s.success_rate:.0%}", bordered=True)
            )
        _fig, _ax = plt.subplots()
        _ax.hist(run_costs, bins=30, color=ACCENT, alpha=0.85)
        if _cap.safe_cap is not None:
            _ax.axvline(_cap.safe_cap, color="#b91c1c", linestyle="--")
            _ax.text(_cap.safe_cap, _ax.get_ylim()[1] * 0.9, " recommended cap", color="#b91c1c", fontsize=8)
        _ax.axvline(_s.median, color=MUTED, linestyle=":")
        _ax.set_xlabel("Cost per run (dollars)")
        _ax.set_ylabel("Runs")
        _fig.tight_layout()
        _body = mo.vstack(
            [
                mo.hstack(_cards, justify="start", wrap=True),
                runs_stop,
                mo.callout(mo.md(_cap.summary().replace("\n", "<br>")), kind="success" if _cap.safe_cap else "info"),
                _fig,
            ]
        )
    mo.vstack([_header, runs_text, _body])
    return


@app.cell
def _(mo):
    sp_per_day = mo.ui.number(1, 10_000_000, value=10_000, label="Runs per day")
    sp_days = mo.ui.slider(1, 365, step=1, value=30, label="Days", show_value=True)
    return sp_days, sp_per_day


@app.cell
def _(cp, mo, parse_error, run_costs, sp_days, sp_per_day):
    _header = mo.md(
        """
        <h2 id="spend">3. What will it cost at scale?</h2>

        Uses the runs in section 2. At high volume, run to run variation averages out, and the range
        comes almost entirely from how well the pilot pinned down the average. A bigger pilot
        narrows it more than anything else.
        """
    )
    if parse_error:
        _body = mo.callout(mo.md(parse_error), kind="warn")
    else:
        _runs = int(sp_per_day.value) * sp_days.value
        _p = cp.project_spend(run_costs, _runs)
        _body = mo.hstack(
            [
                mo.stat(f"${_p.expected:,.0f}", label="Expected spend", caption=f"{_runs:,} runs", bordered=True),
                mo.stat(f"${_p.low:,.0f} to ${_p.high:,.0f}", label="95% range", caption=f"from {len(run_costs)} pilot runs", bordered=True),
            ],
            justify="start",
            wrap=True,
        )
    mo.vstack([_header, mo.hstack([sp_per_day, sp_days], wrap=True), _body])
    return


@app.cell
def _(mo):
    cm_a_cost = mo.ui.number(0.0001, 100, step=0.001, value=0.16, label="Large model: average cost per run ($)")
    cm_a_rate = mo.ui.slider(0.05, 1.0, step=0.01, value=0.85, label="Large model: success rate", show_value=True)
    cm_b_cost = mo.ui.number(0.0001, 100, step=0.001, value=0.11, label="Small model: average cost per run ($)")
    cm_b_rate = mo.ui.slider(0.05, 1.0, step=0.01, value=0.55, label="Small model: success rate", show_value=True)
    return cm_a_cost, cm_a_rate, cm_b_cost, cm_b_rate


@app.cell
def _(cm_a_cost, cm_a_rate, cm_b_cost, cm_b_rate, mo):
    _a = cm_a_cost.value / cm_a_rate.value
    _b = cm_b_cost.value / cm_b_rate.value
    _cheaper_run = cm_b_cost.value < cm_a_cost.value
    _verdict = (
        "The small model is cheaper per run but more expensive per finished task."
        if _cheaper_run and _b > _a
        else "The small model is cheaper per finished task too."
        if _b <= _a
        else "The small model costs more per run and per finished task."
    )
    mo.vstack(
        [
            mo.md(
                """
                <h2 id="compare">4. Is the cheaper model cheaper per task?</h2>

                A model with lower prices that fails more often, or loops more, can cost more per
                task that actually gets done. Compare cost per success, not cost per call. With your
                own pilot runs, `costplan compare a.csv b.csv` adds a confidence interval. Run both models on the same tasks and add `--paired` for a much tighter one.
                """
            ),
            mo.hstack([cm_a_cost, cm_a_rate], wrap=True),
            mo.hstack([cm_b_cost, cm_b_rate], wrap=True),
            mo.hstack(
                [
                    mo.stat(f"${_a:.3f}", label="Large model per success", bordered=True),
                    mo.stat(f"${_b:.3f}", label="Small model per success", caption=f"{_b / _a:.2f} times the large model", bordered=True),
                ],
                justify="start",
                wrap=True,
            ),
            mo.callout(mo.md(_verdict), kind="warn" if _cheaper_run and _b > _a else "info"),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    Built with [costplan](https://github.com/ai-with-md/costplan), MIT licensed. The methods:
    order statistics for quantiles and caps (no assumption about the shape of costs), the percentile
    bootstrap for averages and cost per success, Wilson intervals for success rates, and a normal
    approximation for spend projections. Every promise is checked against simulation in the test suite.
    """)
    return


if __name__ == "__main__":
    app.run()
