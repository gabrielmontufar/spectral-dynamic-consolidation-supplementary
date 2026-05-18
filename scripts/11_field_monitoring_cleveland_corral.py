from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "external_data" / "cleveland_corral"
DATA = EXT / "daily" / "Cleveland_Corral_Daily_Data"
OUT = ROOT / "outputs"
FIG = ROOT / "figures"


def survival_double(pi: np.ndarray, terms: int = 120) -> np.ndarray:
    pi = np.maximum(np.asarray(pi, dtype=float), 0.0)
    m = np.arange(1, 2 * terms, 2, dtype=float)
    weights = 8.0 / (m * m * np.pi * np.pi)
    rate = m * m * np.pi * np.pi
    return np.exp(-pi[:, None] * rate[None, :]) @ weights


def read_middle_daily() -> pd.DataFrame:
    frames = []
    for name in ["CCmiddle_daily_1997_2002.csv", "CCmiddle_daily_2002_2018.csv"]:
        df = pd.read_csv(DATA / name, skiprows=3)
        df["date"] = pd.to_datetime(df["date"])
        frames.append(df)
    df = pd.concat(frames, ignore_index=True).sort_values("date").drop_duplicates("date")
    for col in df.columns:
        if col != "date":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.set_index("date")
    for col in [c for c in df.columns if "precipitation" in c]:
        raw = df[col].astype(float)
        # The public daily file stores the rain gauge as a cumulative counter in
        # several water-year periods. Use positive daily increments for event
        # detection and rainfall-only baselines.
        if raw.quantile(0.95) > 500.0:
            inc = raw.diff()
            inc = inc.where(inc >= 0.0, raw)
            df[col] = inc.clip(lower=0.0)
    return df


def sensor_depths() -> dict[str, float]:
    desc = pd.read_csv(EXT / "Cleveland_Corral_Sensor_Descriptions.csv", skiprows=4)
    depth_col = "depth: ground to sensor diaphragm (m) "
    depths: dict[str, float] = {}
    for _, row in desc.iterrows():
        sensor = str(row.get("instrument ID", "")).strip()
        depth = pd.to_numeric(row.get(depth_col), errors="coerce")
        if sensor and pd.notna(depth):
            depths[sensor] = float(depth)
    return depths


def piezometer_columns(df: pd.DataFrame) -> dict[str, str]:
    mapping = {}
    for col in df.columns:
        if "piezometer" not in col:
            continue
        # Column names end with the public sensor id, for example mid_P1.
        sensor = "_".join(col.split("_")[-2:])
        mapping[col] = sensor
    return mapping


def event_starts(df: pd.DataFrame, rain_col: str) -> list[pd.Timestamp]:
    rain3 = df[rain_col].fillna(0.0).rolling(3, min_periods=1).sum()
    starts: list[pd.Timestamp] = []
    last = pd.Timestamp("1900-01-01")
    for date, value in rain3.items():
        if value >= 25.0 and (date - last).days >= 45:
            starts.append(date)
            last = date
    return starts


def extract_event_decay(df: pd.DataFrame, starts: list[pd.Timestamp], col: str, sensor: str, h_m: float) -> pd.DataFrame:
    rows = []
    rain_col = [c for c in df.columns if "precipitation" in c][0]
    disp_cols = [c for c in df.columns if "extensometer" in c]
    disp = df[disp_cols].bfill(axis=1).iloc[:, 0] if disp_cols else pd.Series(index=df.index, dtype=float)
    for event_id, start in enumerate(starts):
        pre = df.loc[start - pd.Timedelta(days=20) : start - pd.Timedelta(days=1), col].dropna()
        early = df.loc[start : start + pd.Timedelta(days=15), col].dropna()
        if len(pre) < 5 or len(early) < 5:
            continue
        base = float(pre.median())
        peak_date = early.idxmax()
        peak = float(early.loc[peak_date])
        amplitude = peak - base
        if amplitude < 5.0:
            continue
        window = df.loc[peak_date : peak_date + pd.Timedelta(days=30), [col, rain_col]].copy()
        if len(window) < 10:
            continue
        t_days = (window.index - peak_date).days.to_numpy(dtype=float)
        r_obs = (window[col].to_numpy(dtype=float) - base) / amplitude
        valid = np.isfinite(r_obs)
        if valid.sum() < 10:
            continue
        disp_pre = disp.loc[start - pd.Timedelta(days=7) : start].dropna()
        disp_post = disp.loc[start : start + pd.Timedelta(days=30)].dropna()
        disp_increment = float(disp_post.max() - disp_pre.median()) if len(disp_pre) and len(disp_post) else np.nan
        event_rain_7d = float(df.loc[start : start + pd.Timedelta(days=7), rain_col].fillna(0.0).sum())
        for day, obs, rain in zip(t_days[valid], r_obs[valid], window[rain_col].to_numpy(dtype=float)[valid]):
            rows.append(
                {
                    "dataset": "Cleveland Corral",
                    "record_id": sensor,
                    "event_id": f"{sensor}_{event_id:03d}_{start.date()}",
                    "event_start": start.date().isoformat(),
                    "peak_date": peak_date.date().isoformat(),
                    "split": "train" if start.year <= 2010 else "test",
                    "t_days_since_peak": float(day),
                    "T_s": float(day * 86400.0),
                    "h_m": h_m,
                    "pressure_head_cm": float(obs * amplitude + base),
                    "pre_event_head_cm": base,
                    "peak_head_cm": peak,
                    "event_amplitude_cm": amplitude,
                    "rain_mm_day": float(rain) if np.isfinite(rain) else np.nan,
                    "event_rain_7d_mm": event_rain_7d,
                    "displacement_increment_30d_cm": disp_increment,
                    "R_obs": float(obs),
                }
            )
    return pd.DataFrame(rows)


def mae(obs: np.ndarray, pred: np.ndarray) -> float:
    mask = np.isfinite(obs) & np.isfinite(pred)
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs(obs[mask] - pred[mask])))


def rmse(obs: np.ndarray, pred: np.ndarray) -> float:
    mask = np.isfinite(obs) & np.isfinite(pred)
    if not np.any(mask):
        return float("nan")
    err = obs[mask] - pred[mask]
    return float(np.sqrt(np.mean(err * err)))


def fit_and_predict(matrix: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cv_grid = np.logspace(-10, -5, 100)
    pred_rows = []
    metric_rows = []
    for sensor, g in matrix.groupby("record_id"):
        train = g[g["split"] == "train"].copy()
        test = g[g["split"] == "test"].copy()
        if len(train) < 50 or len(test) < 20:
            continue
        h = float(g["h_m"].median())
        y_train = np.clip(train["R_obs"].to_numpy(dtype=float), -0.25, 1.25)
        t_train = train["T_s"].to_numpy(dtype=float)
        losses = [mae(y_train, survival_double(cv * t_train / (h * h))) for cv in cv_grid]
        cv = float(cv_grid[int(np.nanargmin(losses))])
        train_day = train.groupby("t_days_since_peak")["R_obs"].median()

        for split_name, part in [("train", train), ("test", test)]:
            t = part["T_s"].to_numpy(dtype=float)
            obs = np.clip(part["R_obs"].to_numpy(dtype=float), -0.25, 1.25)
            predictions = {
                "M0_drained": np.zeros(len(part)),
                "M1_undrained": np.ones(len(part)),
                "M4_field_monitoring_Pi_R": survival_double(cv * t / (h * h)),
                "M6_event_median_decay_baseline": part["t_days_since_peak"].map(train_day).fillna(train_day.median()).to_numpy(dtype=float),
            }
            for model, pred in predictions.items():
                metric_rows.append(
                    {
                        "dataset": "Cleveland Corral",
                        "scale": "public field monitoring",
                        "validation": "storm-window pore-pressure retention transfer",
                        "record_id": sensor,
                        "split": split_name,
                        "model": model,
                        "cv_train_m2_s": cv if model == "M4_field_monitoring_Pi_R" else np.nan,
                        "h_m": h,
                        "n": int(np.isfinite(obs).sum()),
                        "mae": mae(obs, pred),
                        "rmse": rmse(obs, pred),
                        "spearman": float(pd.Series(obs).corr(pd.Series(pred), method="spearman")),
                    }
                )
            sample = part.copy()
            sample = sample.iloc[:: max(1, len(sample) // 250)].copy()
            for model, pred in predictions.items():
                pp = pred[:: max(1, len(part) // 250)]
                for (_, row), r_pred in zip(sample.iterrows(), pp):
                    pred_rows.append(
                        {
                            "dataset": "Cleveland Corral",
                            "record_id": sensor,
                            "event_id": row["event_id"],
                            "split": split_name,
                            "model": model,
                            "t_days_since_peak": row["t_days_since_peak"],
                            "Pi": cv * row["T_s"] / (h * h),
                            "R_obs": row["R_obs"],
                            "R_pred": float(r_pred),
                        }
                    )
    return pd.DataFrame(metric_rows), pd.DataFrame(pred_rows)


def pressure_displacement_screen(matrix: pd.DataFrame, metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    phys = metrics[(metrics["split"] == "test") & (metrics["model"] == "M4_field_monitoring_Pi_R")]
    if phys.empty:
        return pd.DataFrame(), pd.DataFrame()
    event = (
        matrix.groupby(["event_id", "record_id", "split"], as_index=False)
        .agg(
            event_start=("event_start", "first"),
            max_R_obs=("R_obs", "max"),
            event_amplitude_cm=("event_amplitude_cm", "first"),
            event_rain_7d_mm=("event_rain_7d_mm", "first"),
            displacement_increment_30d_cm=("displacement_increment_30d_cm", "first"),
        )
        .dropna(subset=["displacement_increment_30d_cm"])
    )
    if event.empty:
        return event, pd.DataFrame()
    event["active_window"] = event["displacement_increment_30d_cm"] >= 0.5
    train = event[event["split"] == "train"]
    test = event[event["split"] == "test"]
    rows = []
    for score in ["event_amplitude_cm", "event_rain_7d_mm"]:
        if train.empty or test.empty or train["active_window"].nunique() < 2:
            continue
        candidates = np.nanquantile(train[score], np.linspace(0.1, 0.9, 17))
        best_thr = candidates[0]
        best_f1 = -1.0
        for thr in candidates:
            pred = train[score] >= thr
            tp = int((pred & train["active_window"]).sum())
            fp = int((pred & ~train["active_window"]).sum())
            fn = int((~pred & train["active_window"]).sum())
            f1 = 2 * tp / max(2 * tp + fp + fn, 1)
            if f1 > best_f1:
                best_f1 = f1
                best_thr = float(thr)
        pred = test[score] >= best_thr
        tp = int((pred & test["active_window"]).sum())
        fp = int((pred & ~test["active_window"]).sum())
        fn = int((~pred & test["active_window"]).sum())
        tn = int((~pred & ~test["active_window"]).sum())
        rows.append(
            {
                "dataset": "Cleveland Corral",
                "screen": score,
                "split": "test",
                "threshold_train": best_thr,
                "n_events": int(len(test)),
                "accuracy": float((pred == test["active_window"]).mean()),
                "f1_active": float(2 * tp / max(2 * tp + fp + fn, 1)),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "claim_level": "auxiliary pressure-displacement screening only",
            }
        )
    return event, pd.DataFrame(rows)


def make_figures(matrix: pd.DataFrame, preds: pd.DataFrame, metrics: pd.DataFrame, event_metrics: pd.DataFrame) -> None:
    FIG.mkdir(exist_ok=True)
    fig, axs = plt.subplots(1, 2, figsize=(9.4, 4.2), dpi=300, constrained_layout=True)
    phys = preds[(preds["split"] == "test") & (preds["model"] == "M4_field_monitoring_Pi_R")]
    for sensor, g in phys.groupby("record_id"):
        axs[0].plot(g["t_days_since_peak"], np.clip(g["R_obs"], -0.25, 1.25), ".", ms=2.0, alpha=0.22)
        med = g.groupby("t_days_since_peak")[["R_obs", "R_pred"]].median().reset_index()
        axs[0].plot(med["t_days_since_peak"], med["R_pred"], lw=1.1, label=sensor)
    axs[0].set_xlabel("Days since pressure peak")
    axs[0].set_ylabel("Normalized retained pressure head")
    axs[0].set_title("Cleveland Corral field-monitoring transfer")
    axs[0].grid(alpha=0.25)
    axs[0].legend(fontsize=7)

    test_metrics = metrics[metrics["split"] == "test"].copy()
    order = (
        test_metrics.groupby("model")["mae"]
        .median()
        .sort_values()
        .index.tolist()
    )
    vals = [test_metrics[test_metrics["model"] == m]["mae"].median() for m in order]
    axs[1].barh([m.replace("_", " ") for m in order], vals, color="#4f7f9f")
    axs[1].set_xlabel("Median test MAE")
    axs[1].set_title("Baseline comparison")
    axs[1].grid(axis="x", alpha=0.25)
    fig.savefig(FIG / "fig17_cleveland_corral_field_monitoring_transfer.png", bbox_inches="tight")
    plt.close(fig)

    if not event_metrics.empty:
        fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=300, constrained_layout=True)
        ax.bar(event_metrics["screen"], event_metrics["f1_active"], color="#6b8f5d")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Test F1 for active displacement windows")
        ax.set_title("Auxiliary pressure-displacement screen")
        ax.grid(axis="y", alpha=0.25)
        fig.savefig(FIG / "fig18_cleveland_corral_displacement_screen.png", bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    df = read_middle_daily()
    depths = sensor_depths()
    starts = event_starts(df, [c for c in df.columns if "precipitation" in c][0])
    parts = []
    for col, sensor in piezometer_columns(df).items():
        if sensor not in depths:
            continue
        part = extract_event_decay(df, starts, col, sensor, depths[sensor])
        if not part.empty:
            parts.append(part)
    matrix = pd.concat(parts, ignore_index=True)
    matrix.to_csv(OUT / "cleveland_corral_field_monitoring_matrix.csv", index=False)
    metrics, preds = fit_and_predict(matrix)
    metrics.to_csv(OUT / "cleveland_corral_transfer_metrics.csv", index=False)
    preds.to_csv(OUT / "cleveland_corral_transfer_predictions.csv", index=False)
    events, screen_metrics = pressure_displacement_screen(matrix, metrics)
    events.to_csv(OUT / "cleveland_corral_pressure_displacement_events.csv", index=False)
    screen_metrics.to_csv(OUT / "cleveland_corral_pressure_displacement_screen.csv", index=False)
    summary = metrics[metrics["split"] == "test"].groupby("model", as_index=False).agg(
        median_mae=("mae", "median"),
        median_rmse=("rmse", "median"),
        median_spearman=("spearman", "median"),
        n_points=("n", "sum"),
    )
    summary["claim_level"] = "independent public field-monitoring transfer check; not full field calibration"
    summary.to_csv(OUT / "field_monitoring_transfer_summary.csv", index=False)
    make_figures(matrix, preds, metrics, screen_metrics)
    print("wrote Cleveland Corral field-monitoring transfer outputs")


if __name__ == "__main__":
    main()
