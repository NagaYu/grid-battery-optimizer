"""
バッテリー充放電の最適化モデル (PuLP) — 大規模・頑健性強化版

データ規模が100倍（数千〜時刻）になっても実務時間内に解を返すための工夫:

  [1] ソルバー制御 (calc speed)
      - time_limit (秒) で打ち切り、mip_gap で「十分良い近似解」を許容。
      - HiGHS を優先（arm64 ネイティブ）、無ければ CBC にフォールバック。

  [2] スパース性 (疎なマトリクス)
      - 値が常に0になる変数（夜間 solar=0 の curtail 等）は生成しない。
      - 制約は lpSum / 辞書内包で必要最小限だけ構築。

  [3] 例外処理 (解なし耐性)
      - status を厳格にチェック。Infeasible 等でクラッシュさせない。
      - 「制約緩和（未充足需要にペナルティ）」→「ルールベース代替」の
        2段フォールバックで必ず実行可能な解を返す。

意思決定変数（各時刻 t）:
    grid_buy[t]   : グリッド購入量 (MWh)      0 <= . <= grid_limit
    charge[t]     : 充電量 (MWh)              0 <= . <= max_rate
    discharge[t]  : 放電量 (MWh)              0 <= . <= max_rate
    soc[t]        : 時刻 t 終了時の蓄電量      0 <= . <= capacity
    curtail[t]    : 太陽光抑制量 (solar>0 の時刻のみ生成)
    is_charge[t]  : 充電中フラグ (binary, no_simultaneous=True の時のみ)
    unmet[t]      : 未充足需要 (allow_unmet=True の緩和時のみ, 高ペナルティ)
"""

import pulp
import numpy as np

# ステータス分類
_STATUS_OPTIMAL = "Optimal"
_STATUS_INFEASIBLE = "Infeasible"


def _get_solver(time_limit=None, mip_gap=None, threads=None, msg=False):
    """
    利用可能なソルバーを time_limit / mip_gap 付きで構築する。

    - HiGHS / CBC とも PuLP 共通引数 timeLimit, gapRel を受け付ける。
    - 古い CBC 用語の maxSeconds=timeLimit, fractionGap=gapRel に相当。
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
    # 最後の手段
    return pulp.PULP_CBC_CMD(**kwargs)


def _loss_breakpoints(flow_cap, n_segments):
    """送電損失の区分線形近似に使う接線の接点（流量）を返す。"""
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
    LP/MIP を構築して解く内部関数。スパース構築を徹底する。
    Returns: (status_str, result_dict_or_None)
    """
    T = len(solar)
    rng = range(T)

    prob = pulp.LpProblem("Battery_Cost_Min", pulp.LpMinimize)

    # --- 変数（必要なものだけ生成 = スパース）---
    grid_buy = pulp.LpVariable.dicts("g", rng, lowBound=0, upBound=grid_limit)
    charge = pulp.LpVariable.dicts("c", rng, lowBound=0, upBound=max_rate)
    discharge = pulp.LpVariable.dicts("d", rng, lowBound=0, upBound=max_rate)
    soc = pulp.LpVariable.dicts("s", rng, lowBound=0, upBound=capacity)

    # curtail は solar>0 の時刻だけ生成（夜間は捨てる余地が無いので変数不要）
    sun_hours = [t for t in rng if solar[t] > 0]
    curtail = pulp.LpVariable.dicts("cu", sun_hours, lowBound=0)

    # 充放電同時禁止フラグ（MIP）。LP のままで良ければ生成しない。
    is_charge = (
        pulp.LpVariable.dicts("b", rng, cat="Binary") if no_simultaneous else None
    )

    # 制約緩和用の未充足需要スラック（フォールバック時のみ生成）
    unmet = (
        pulp.LpVariable.dicts("u", rng, lowBound=0) if allow_unmet else None
    )

    # 送電損失 loss[t] (MWh)。grid からの送電流量に対する I^2R 損を表現する。
    # 物理的に損失は流量の2乗に比例（凸）するため、複数の接線で下から近似（区分線形）。
    # loss_coeff=0 なら損失を無効化（純コスト最小化）。
    use_loss = loss_coeff is not None and loss_coeff > 0
    if use_loss:
        flow_cap = float(grid_limit) if grid_limit is not None \
            else float(max(demand) + max_rate)
        bpts = _loss_breakpoints(flow_cap, loss_segments)
        loss = pulp.LpVariable.dicts("loss", rng, lowBound=0)
    else:
        loss = None

    # --- 目的関数（lpSum で一括構築）---
    # grid_buy は「送電端での取得量」= 需要充足分 + 送電損失。損失分も買電コストになる。
    obj = pulp.lpSum(grid_buy[t] * price[t] for t in rng)
    if allow_unmet:
        obj += pulp.lpSum(unmet[t] * unmet_penalty for t in rng)
    prob += obj, "TotalCost"

    # --- 制約（必要最小限のみ）---
    for t in rng:
        used_solar = solar[t] - (curtail[t] if t in curtail else 0)
        # 負荷端に届く電力 = 太陽光 + (購入 - 送電損失) + 放電
        supply = used_solar + grid_buy[t] + discharge[t]
        if use_loss:
            supply = supply - loss[t]
        if allow_unmet:
            supply += unmet[t]
        prob += (supply == demand[t] + charge[t]), f"bal_{t}"

        # curtail <= solar は upBound で表現（変数自体に上限を持たせる方が疎）
        if t in curtail:
            curtail[t].upBound = solar[t]

        # 送電損失の区分線形近似: loss >= 2a*x0*grid_buy - a*x0^2 （a*x^2 の接線群）
        if use_loss:
            a = loss_coeff
            for ki, x0 in enumerate(bpts):
                prob += (
                    loss[t] >= 2 * a * x0 * grid_buy[t] - a * x0 * x0
                ), f"loss_{t}_{ki}"

        # SoC 遷移
        prev = initial_soc if t == 0 else soc[t - 1]
        prob += (
            soc[t] == prev + charge[t] * eff_charge - discharge[t] / eff_discharge
        ), f"soc_{t}"

        # 充放電同時禁止（MIP）: charge<=M*b, discharge<=M*(1-b)
        if no_simultaneous:
            prob += charge[t] <= max_rate * is_charge[t], f"cx_{t}"
            prob += discharge[t] <= max_rate * (1 - is_charge[t]), f"dx_{t}"

    prob.solve(_get_solver(time_limit, mip_gap, threads, msg))
    status = pulp.LpStatus[prob.status]

    # 解（インカンベント）が存在するか厳格に確認
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
    ソルバーを使わないルールベースの次善策（最終フォールバック）。
    必ず実行可能な解を返す（足りない分は grid_limit の範囲で購入、
    超過分は unmet として計上）。

    戦略: 安い時間帯（下位40%価格）に余力があれば充電、
          高い時間帯（上位40%）に放電してピークの購入を回避。
    送電損失 loss = loss_coeff * (送電端流量)^2 を加味する。
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
            # 太陽光が需要を上回る → 余剰で充電、残りは抑制
            ch = min(surplus, max_rate, (capacity - soc) / eff_charge)
            curt = surplus - ch
        else:
            need = -surplus  # 不足分（負荷端で必要な量）
            if price[t] >= thr_high:
                # 高価格 → 放電してグリッド購入を抑える
                dis = min(need, max_rate, soc * eff_discharge)
                delivered = need - dis
            elif price[t] <= thr_low:
                # 安価格 → 余力があれば追加購入して充電（アービトラージ）
                ch = min(max_rate, (capacity - soc) / eff_charge)
                delivered = need + ch
            else:
                delivered = need

            # 送電端流量 = 負荷端到達量 + 損失。loss = a*flow^2 を delivered で近似。
            buy = delivered + a * delivered * delivered

            # グリッド購入上限を超える分は未充足
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
    grid_limit=None,            # グリッド購入の上限 (MWh/h)。None で無制限。
    no_simultaneous: bool = True,  # 充放電同時禁止 (MIP化)。False で純LP（高速）。
    loss_coeff: float = 0.0006,  # 送電損失係数 a（loss=a*flow^2）。0 で損失無効。
    loss_segments: int = 6,     # 損失の区分線形近似に使う接線本数
    time_limit: float = 30.0,   # ソルバー打ち切り秒数
    mip_gap: float = 0.01,      # 許容 MIP ギャップ（1% 近似解で妥協）
    threads=None,
    msg: bool = False,
):
    """
    頑健な最適化エントリポイント。

    フロー:
      1) 本来のモデルを time_limit / mip_gap 付きで解く。
      2) 解あり（Optimal もしくは打ち切りでも実行可能なインカンベント）→ 返す。
      3) Infeasible 等で解なし → 制約緩和（未充足需要にペナルティ）で再求解。
      4) それでも駄目 → ルールベースの代替案を返す（クラッシュさせない）。

    Returns:
        dict: status, method, relaxed, total_cost, grid_buy[], charge[],
              discharge[], soc[], solar_curtail[], unmet[], n_vars, n_constraints,
              fallback_used (bool)
    """
    # --- 1) 通常求解 ---
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

    # --- 2) フォールバックA: 制約緩和して再求解 ---
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

    # --- 3) フォールバックB: ルールベースの最終手段 ---
    rb = rule_based_dispatch(
        solar, demand, price, capacity, max_rate, initial_soc,
        eff_charge, eff_discharge, grid_limit, loss_coeff,
    )
    rb["fallback_used"] = True
    rb["status"] = f"{_STATUS_INFEASIBLE}->RuleBased"
    return rb


def baseline_cost(solar, demand, price, loss_coeff=0.0006):
    """
    最適化前（ベースライン）: バッテリーを使わず、太陽光の余剰は捨て、
    不足分はすべてその時刻の価格でグリッドから購入する。
    公平な比較のため、送電損失 loss = loss_coeff*(流量)^2 も最適化側と同条件で加味する。
    """
    T = len(solar)
    a = loss_coeff if loss_coeff else 0.0
    grid_buy, curtail, losses = [], [], []
    total = 0.0
    for t in range(T):
        net = demand[t] - solar[t]
        if net >= 0:
            # 負荷端で net 必要 → 送電損失込みで net + a*net^2 を送電端から購入
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
