"""Shared feature schema for StatsBomb training shots and ESPN application shots."""
import numpy as np, pandas as pd

X_SCALE = 48.1          # metres per ESPN fx unit (CLAUDE.md §4)
Y_SCALE = 68.0          # metres per ESPN fy unit: ASSUMED, see REPORT.md sensitivity
SB_TO_M = 0.9144        # StatsBomb 120x80 yard pitch -> metres
GOAL_W = 7.32

MODEL_FEATURES = ["distance", "angle", "is_header", "from_corner", "from_fast_break", "from_set_piece",
                  "assisted", "assist_was_cross", "assist_was_through_ball"]
AUX = ["is_penalty", "is_free_kick"]
SCHEMA = MODEL_FEATURES + AUX


def _geom(dx, dy):
    """dx = metres from goal line, dy = lateral metres from goal centre."""
    dist = np.hypot(dx, dy)
    ay = np.abs(dy)
    # angle subtended by the posts: atan2 form stays positive for shots level with / beyond the goal line
    angle = np.arctan2(GOAL_W * dx, dx**2 + ay**2 - (GOAL_W / 2) ** 2)
    angle = np.where(angle < 0, angle + np.pi, angle)
    return dist, angle


def _finish(df):
    df = df[SCHEMA].copy()
    for c in SCHEMA:
        df[c] = df[c].astype("float64") if c in ("distance", "angle") else df[c].astype("int8")
    return df


def features_statsbomb(events: pd.DataFrame) -> pd.DataFrame:
    dx = (120 - events.x) * SB_TO_M
    dy = (events.y - 40) * SB_TO_M
    dist, angle = _geom(dx, dy)
    out = pd.DataFrame(index=events.index)
    out["distance"], out["angle"] = dist, angle
    out["is_header"] = events.shot_body_part.eq("Head")
    out["from_corner"] = events.play_pattern.eq("From Corner")
    out["from_fast_break"] = events.play_pattern.eq("From Counter")
    out["from_set_piece"] = events.play_pattern.eq("From Free Kick")
    out["assisted"] = events.shot_key_pass_id.notna()
    out["assist_was_cross"] = events.kp_cross.fillna(False).astype(bool)
    out["assist_was_through_ball"] = events.kp_through.fillna(False).astype(bool)
    out["is_penalty"] = events.shot_type.eq("Penalty")
    out["is_free_kick"] = events.shot_type.eq("Free Kick")
    return _finish(out)


def features_espn(shots: pd.DataFrame, y_scale: float = Y_SCALE) -> pd.DataFrame:
    t = shots.text.fillna("").str.lower()
    dx = X_SCALE * shots.fx
    dy = (shots.fy - 0.5) * y_scale
    dist, angle = _geom(dx, dy)
    out = pd.DataFrame(index=shots.index)
    out["distance"], out["angle"] = dist, angle
    out["is_header"] = t.str.contains("header") | shots.type.eq("goal---header")
    out["from_corner"] = t.str.contains("following a corner")
    out["from_fast_break"] = t.str.contains("following a fast break")
    out["from_set_piece"] = t.str.contains("following a set piece")
    out["assisted"] = t.str.contains("assisted by")
    out["assist_was_cross"] = t.str.contains("with a cross")
    out["assist_was_through_ball"] = t.str.contains("with a through ball")
    out["is_penalty"] = shots.type.str.startswith("penalty")
    out["is_free_kick"] = False   # ESPN text never says "free kick" (0/309): cannot be identified
    return _finish(out)


def assert_same_schema(a: pd.DataFrame, b: pd.DataFrame):
    assert list(a.columns) == list(b.columns), (list(a.columns), list(b.columns))
    assert (a.dtypes == b.dtypes).all(), pd.concat([a.dtypes, b.dtypes], axis=1)
