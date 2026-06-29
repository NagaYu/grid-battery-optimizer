"""
Battery charge/discharge optimization model (PuLP) — scalable & robust edition.

Techniques that keep solve time practical even when the data grows ~100x
(thousands of time steps):

  [1] Solver control (calc speed)
      - Cut off with time_limit (seconds); accept a "good enough" approximate
        solution via mip_gap.
      - Prefer HiGHS (arm64-native); fall back to CBC if unavailable.

  [2] Sparsity (sparse matrix)
      - Do not create variables that are always zero (e.g., curtail at night
        when solar=0).
      - Build constraints with lpSum / dict comprehensions, kept minimal.

  [3] Exception handling (infeasibility tolerance)
      - Check status strictly. Never crash on Infeasible etc.
      - A two-stage fallback ("constraint relaxation with a penalty on unmet
        demand" -> "rule-based alternative") always returns a feasible solution.

Decision variables (for each time step t):
    grid_buy[t]   : grid purchase (MWh)          0 <= . <= grid_limit
    charge[t]     : charge amount (MWh)           0 <= . <= max_rate
    discharge[t]  : discharge amount (MWh)        0 <= . <= max_rate
    soc[t]        : state of charge at end of t   0 <= . <= capacity
    curtail[t]    : solar curtailment (created only at steps where solar>0)
    is_charge[t]  : charging flag (binary, only when no_simultaneous=True)
    unmet[t]      : unmet demand (only when allow_unmet=True, heavily penalized)
"""

import pulp
import numpy as np

# Status labels
_STATUS_OPTIMAL = "Optimal"
_STATUS_INFEASIBLE = "Infeasible"


def _get_solver(time_limit=None, mip_gap=None, threads=None, msg=False):
    """
    Build an available solver configured with time_limit / mip_gap.

    - Both HiGHS and CBC accept the PuLP-common args timeLimit, gapRel.
    - These correspond to the legacy CBC terms maxSeconds=timeLimit,
      fractionGap=gapRel.
    """
    kwargs = {"msg": msg}
    if time_limit is not None:
        kwargs["timeLimit"] = float(time_limit)
    if mip_gap is not None:
        kwargs["gapRel"] = float(mip_gap)
    if threads is not None:
        kwargs["threads"] = int(threads)

    for solver_cls in ("HiGHS", "PULP_CBC_CMD"):
        try:
            solver = getattr(pulp, solver_cls)(**kwargs)
            if solver.available():
                return solver
        except Exception:
            continue
    # Last resort
    return pulp.PULP_CBC_CMD(**kwargs)


def _loss_breakpoints(flow_cap, n_segments):
    """Return the tangent points (flow values) used to PWL-approximate loss."""
    K = max(2, int(n_segments))
    return [flow_cap * (k / K) for k in range(1, K + 1)]


def _build_and_solve(
    solar, demand, price,
    capacity, max_rate, initial_soc, eff_charge, eff_discharge,
    grid_limit, no_simultaneous, allow_unmet, unmet_penalty,
    loss_coeff, loss_segments,
    time_limit, mip_gap, threads, msg,
):
    """
    Internal function that builds and solves the LP/MIP, kept strictly sparse.
    Returns: (status_str, result_dict_or_None)
    """
    T = len(solar)
    rng = range(T)

    prob = pulp.LpProblem("Battery_Cost_Min", pulp.LpMinimize)

    # --- Variables (create only what is needed = sparse) ---
    grid_buy = pulp.LpVariable.dicts("g", rng, lowBound=0, upBound=grid_limit)
    charge = pulp.LpVariable.dicts("c", rng, lowBound=0, upBound=max_rate)
    discharge = pulp.LpVariable.dicts("d", rng, lowBound=0, upBound=max_rate)
    soc = pulp.LpVariable.dicts("s", rng, lowBound=0, upBound=capacity)

    # curtail only at steps with solar>0 (no curtailment to do at night)
    sun_hours = [t for t in rng if solar[t] > 0]
    curtail = pulp.LpVariable.dicts("cu", sun_hours, lowBound=0)

    # No-simultaneous-charge/discharge flag (MIP). Skip if pure LP is fine.
    is_charge = (
        pulp.LpVariable.dicts("b", rng, cat="Binary") if no_simultaneous else None
    )

    # Unmet-demand slack for constraint relaxation (created only on fallback)
    unmet = (
        pulp.LpVariable.dicts("u", rng, lowBound=0) if allow_unmet else None
    )

    # Transmission loss loss[t] (MWh): represents I^2R loss on grid-side flow.
    # Physically loss is proportional to the square of flow (convex), so it is
    # approximated from below with multiple tangents (piecewise linear).
    # loss_coeff=0 disables losses (pure cost minimization).
    use_loss = loss_coeff is not None and loss_coeff > 0
    if use_loss:
        flow_cap = float(grid_limit) if grid_limit is not None \
            else float(max(demand) + max_rate)
        bpts = _loss_breakpoints(flow_cap, loss_segments)
        loss = pulp.LpVariable.dicts("loss", rng, lowBound=0)
    else:
        loss = None

    # --- Objective (built in one shot with lpSum) ---
    # grid_buy is the "amount taken at the grid side" = served demand +
    # transmission loss; the loss portion is also a purchase cost.
    obj = pulp.lpSum(grid_buy[t] * price[t] for t in rng)
    if allow_unmet:
        obj += pulp.lpSum(unmet[t] * unmet_penalty for t in rng)
    prob += obj, "TotalCost"

    # --- Constraints (only the minimum necessary) ---
    for t in rng:
        used_solar = solar[t] - (curtail[t] if t in curtail else 0)
        # Power reaching the load = solar + (purchase - transmission loss) + discharge
        supply = used_solar + grid_buy[t] + discharge[t]
        if use_loss:
            supply = supply - loss[t]
        if allow_unmet:
            supply += unmet[t]
        prob += (supply == demand[t] + charge[t]), f"bal_{t}"

        # curtail <= solar via upBound (a bound on the variable is sparser than a row)
        if t in curtail:
            curtail[t].upBound = solar[t]

        # PWL approximation of transmission loss: loss >= 2a*x0*grid_buy - a*x0^2
        # (a family of tangents to a*x^2)
        if use_loss:
            a = loss_coeff
            for ki, x0 in enumerate(bpts):
                prob += (
                    loss[t] >= 2 * a * x0 * grid_buy[t] - a * x0 * x0
                ), f"loss_{t}_{ki}"

        # SoC transition
        prev = initial_soc if t == 0 else soc[t - 1]
        prob += (
            soc[t] == prev + charge[t] * eff_charge - discharge[t] / eff_discharge
        ), f"soc_{t}"

        # No simultaneous charge/discharge (MIP): charge<=M*b, discharge<=M*(1-b)
        if no_simultaneous:
            prob += charge[t] <= max_rate * is_charge[t], f"cx_{t}"
            prob += discharge[t] <= max_rate * (1 - is_charge[t]), f"dx_{t}"

    prob.solve(_get_solver(time_limit, mip_gap, threads, msg))
    status = pulp.LpStatus[prob.status]

    # Strictly verify that a solution (incumbent) exists
    obj_val = pulp.value(prob.objective)
    has_solution = obj_val is not None and pulp.value(soc[0]) is not None

    if not has_solution:
        return status, None

    result = {
        "status": status,
        "method": "MIP" if no_simultaneous else "LP",
        "relaxed": bool(allow_unmet),
        "total_cost": float(obj_val),
        "grid_buy": [float(pulp.value(grid_buy[t]) or 0.0) for t in rng],
        "charge": [float(pulp.value(charge[t]) or 0.0) for t in rng],
        "discharge": [float(pulp.value(discharge[t]) or 0.0) for t in rng],
        "soc": [float(pulp.value(soc[t]) or 0.0) for t in rng],
        "solar_curtail": [
            float(pulp.value(curtail[t]) or 0.0) if t in curtail else 0.0 for t in rng
        ],
        "unmet": [float(pulp.value(unmet[t]) or 0.0) for t in rng] if allow_unmet else [0.0] * T,
        "transmission_loss": (
            [float(pulp.value(loss[t]) or 0.0) for t in rng] if use_loss else [0.0] * T
        ),
        "n_vars": prob.numVariables(),
        "n_constraints": prob.numConstraints(),
    }
    return status, result


def rule_based_dispatch(
    solar, demand, price,
    capacity=100.0, max_rate=20.0, initial_soc=0.0,
    eff_charge=1.0, eff_discharge=1.0, grid_limit=None, loss_coeff=0.0,
):
    """
    Solver-free rule-based alternative (final fallback).
    Always returns a feasible solution (buy the shortfall within grid_limit,
    and record any excess as unmet).

    Strategy: charge during cheap hours (bottom 40% price) when there is room,
    discharge during expensive hours (top 40%) to avoid peak purchases.
    Accounts for transmission loss = loss_coeff * (grid-side flow)^2.
    """
    T = len(solar)
    p = np.asarray(price, dtype=float)
    thr_low = np.percentile(p, 40)
    thr_high = np.percentile(p, 60)
    cap_buy = grid_limit if grid_limit is not None else float("inf")
    a = loss_coeff if loss_coeff else 0.0

    soc = float(initial_soc)
    out = {k: [0.0] * T for k in
           ("grid_buy", "charge", "discharge", "soc", "solar_curtail",
            "unmet", "transmission_loss")}
    total = 0.0

    for t in range(T):
        surplus = solar[t] - demand[t]
        ch = dis = buy = curt = unmet = 0.0

        if surplus > 0:
            # Solar exceeds demand -> charge with the surplus, curtail the rest
            ch = min(surplus, max_rate, (capacity - soc) / eff_charge)
            curt = surplus - ch
        else:
            need = -surplus  # shortfall (amount needed at the load side)
            if price[t] >= thr_high:
                # High price -> discharge to reduce grid purchases
                dis = min(need, max_rate, soc * eff_discharge)
                delivered = need - dis
            elif price[t] <= thr_low:
                # Low price -> buy extra to charge if there is room (arbitrage)
                ch = min(max_rate, (capacity - soc) / eff_charge)
                delivered = need + ch
            else:
                delivered = need

            # Grid-side flow = amount delivered to load + loss.
            # Approximate loss = a*flow^2 using `delivered`.
            buy = delivered + a * delivered * delivered

            # Anything above the grid purchase cap is unmet
            if buy > cap_buy:
                unmet = buy - cap_buy
                buy = cap_buy

        loss = a * buy * buy
        soc = soc + ch * eff_charge - dis / eff_discharge
        soc = min(max(soc, 0.0), capacity)

        out["grid_buy"][t] = buy
        out["charge"][t] = ch
        out["discharge"][t] = dis
        out["solar_curtail"][t] = curt
        out["unmet"][t] = unmet
        out["transmission_loss"][t] = loss
        out["soc"][t] = soc
        total += buy * price[t]

    out.update({
        "status": "RuleBased",
        "method": "rule_based",
        "relaxed": True,
        "total_cost": total,
        "n_vars": 0,
        "n_constraints": 0,
    })
    return out


def optimize_battery(
    solar, demand, price,
    capacity: float = 100.0,
    max_rate: float = 20.0,
    initial_soc: float = 0.0,
    eff_charge: float = 1.0,
    eff_discharge: float = 1.0,
    grid_limit=None,            # Grid purchase cap (MWh/h). None = unlimited.
    no_simultaneous: bool = True,  # Forbid simultaneous charge/discharge (MIP). False = pure LP (fast).
    loss_coeff: float = 0.0006,  # Transmission loss coeff a (loss=a*flow^2). 0 disables losses.
    loss_segments: int = 6,     # Number of tangents for the PWL loss approximation
    time_limit: float = 30.0,   # Solver cutoff seconds
    mip_gap: float = 0.01,      # Acceptable MIP gap (settle for a 1% approximation)
    threads=None,
    msg: bool = False,
):
    """
    Robust optimization entry point.

    Flow:
      1) Solve the intended model with time_limit / mip_gap.
      2) If a solution exists (Optimal, or a feasible incumbent even when cut
         off) -> return it.
      3) If no solution due to Infeasible etc. -> re-solve with constraint
         relaxation (penalty on unmet demand).
      4) If that still fails -> return the rule-based alternative (never crash).

    Returns:
        dict: status, method, relaxed, total_cost, grid_buy[], charge[],
              discharge[], soc[], solar_curtail[], unmet[], n_vars, n_constraints,
              fallback_used (bool)
    """
    # --- 1) Normal solve ---
    try:
        status, result = _build_and_solve(
            solar, demand, price, capacity, max_rate, initial_soc,
            eff_charge, eff_discharge, grid_limit, no_simultaneous,
            allow_unmet=False, unmet_penalty=0.0,
            loss_coeff=loss_coeff, loss_segments=loss_segments,
            time_limit=time_limit, mip_gap=mip_gap, threads=threads, msg=msg,
        )
    except Exception as e:
        status, result = ("SolverError:" + type(e).__name__), None

    if result is not None and status not in (_STATUS_INFEASIBLE, "Undefined"):
        result["fallback_used"] = False
        return result

    # --- 2) Fallback A: relax constraints and re-solve ---
    try:
        status2, result2 = _build_and_solve(
            solar, demand, price, capacity, max_rate, initial_soc,
            eff_charge, eff_discharge, grid_limit, no_simultaneous,
            allow_unmet=True, unmet_penalty=1e6,
            loss_coeff=loss_coeff, loss_segments=loss_segments,
            time_limit=time_limit, mip_gap=mip_gap, threads=threads, msg=msg,
        )
    except Exception:
        status2, result2 = "SolverError", None

    if result2 is not None:
        result2["fallback_used"] = True
        result2["status"] = f"{_STATUS_INFEASIBLE}->Relaxed({status2})"
        return result2

    # --- 3) Fallback B: rule-based last resort ---
    rb = rule_based_dispatch(
        solar, demand, price, capacity, max_rate, initial_soc,
        eff_charge, eff_discharge, grid_limit, loss_coeff,
    )
    rb["fallback_used"] = True
    rb["status"] = f"{_STATUS_INFEASIBLE}->RuleBased"
    return rb


def baseline_cost(solar, demand, price, loss_coeff=0.0006):
    """
    Pre-optimization baseline: no battery, curtail any solar surplus, and buy
    the entire shortfall from the grid at each step's price.
    For a fair comparison, transmission loss = loss_coeff*(flow)^2 is included
    under the same conditions as the optimized side.
    """
    T = len(solar)
    a = loss_coeff if loss_coeff else 0.0
    grid_buy, curtail, losses = [], [], []
    total = 0.0
    for t in range(T):
        net = demand[t] - solar[t]
        if net >= 0:
            # Load needs `net` -> buy net + a*net^2 at the grid side (loss included)
            delivered = net
            buy = delivered + a * delivered * delivered
            cut = 0.0
        else:
            buy, cut = 0.0, -net
        loss = a * buy * buy
        grid_buy.append(buy)
        curtail.append(cut)
        losses.append(loss)
        total += buy * price[t]
    return {
        "total_cost": total,
        "grid_buy": grid_buy,
        "solar_curtail": curtail,
        "transmission_loss": losses,
    }
