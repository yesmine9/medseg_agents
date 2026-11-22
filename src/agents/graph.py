"""Graphe LangGraph : orchestrateur -> segmentation -> critic -> (accepter | re-segmenter | humain).

Le segmenteur est injecté (duck typing) : il doit fournir
  .segment(img)        -> (mask, proba)        proba : float HxW
  .segment_mc(img, T)  -> (..., std)           std : HxW (dernier élément du tuple)
"""
import cv2
import numpy as np
from langgraph.graph import StateGraph, START, END

from .critic import DEFAULT_CFG, critic_features, dice_bool, risk_score
from .state import AgentState

STRATEGIES = ["base", "tta", "clahe"]   # une par tentative (0, 1, 2)


def _clahe_view(img: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def _proba(seg, img):
    p = np.asarray(seg.segment(np.ascontiguousarray(img))[1], dtype=np.float32)
    return np.squeeze(p)


def run_attempt(seg, img, strategy, cfg):
    x = _clahe_view(img) if strategy == "clahe" else img
    p_plain = _proba(seg, x)
    p_h = np.flip(_proba(seg, np.flip(x, 1)), 1)
    views = [p_plain, p_h]
    if strategy == "tta":
        views.append(np.flip(_proba(seg, np.flip(x, 0)), 0))
    proba = np.mean(views, axis=0)
    mask = proba > 0.5
    std = np.squeeze(np.asarray(seg.segment_mc(np.ascontiguousarray(x), cfg["mc_T"])[-1], dtype=np.float32))
    flip_dice = dice_bool(p_plain > 0.5, p_h > 0.5)
    return mask, std, flip_dice


def build_graph(segmenter, cfg: dict | None = None):
    cfg = {**DEFAULT_CFG, **(cfg or {})}

    def orchestrator(s: AgentState):
        k = s.get("attempt", 0)
        strategy = STRATEGIES[min(k, len(STRATEGIES) - 1)]
        hist = s.get("history", []) + [f"[orchestrateur] tentative {k} -> stratégie '{strategy}'"]
        return {"attempt": k, "strategy": strategy, "history": hist}

    def segment(s: AgentState):
        mask, std, flip_dice = run_attempt(segmenter, s["image"], s["strategy"], cfg)
        return {"mask": mask, "features": critic_features(mask, std, flip_dice, cfg)}

    def critic(s: AgentState):
        f, k = s["features"], s["attempt"]
        risk = risk_score(f, cfg)
        best = s.get("best")
        if best is None or risk < best["risk"]:
            best = {"risk": risk, "mask": s["mask"], "attempt": k, "features": f}
        if risk < cfg["tau_risk"]:
            decision = "accept"
        elif k < cfg["max_retries"]:
            decision = "retry"
        else:
            decision = "escalate"
        msg = (f"[critic] tentative {k} : risque={risk:.3f} (inc={f['unc_ratio']:.2f}, "
               f"flip={f['flip_dice']:.2f}, comp={f['n_components']}, aire={f['area_frac']:.3f}) -> {decision}")
        out = {"risk": risk, "decision": decision, "best": best, "history": s["history"] + [msg]}
        if decision == "retry":
            out["attempt"] = k + 1
        return out

    def human_review(s: AgentState):
        return {"history": s["history"] + ["[humain] cas escaladé : validation manuelle requise"]}

    def finalize(s: AgentState):
        b = s["best"]
        if s["decision"] == "escalate":
            status = "needs_human"
        else:
            status = "accepted" if b["attempt"] == 0 else "accepted_after_retry"
        return {"final_mask": b["mask"], "status": status, "risk": b["risk"]}

    g = StateGraph(AgentState)
    for name, fn in [("orchestrator", orchestrator), ("segment", segment), ("critic", critic),
                     ("human_review", human_review), ("finalize", finalize)]:
        g.add_node(name, fn)
    g.add_edge(START, "orchestrator")
    g.add_edge("orchestrator", "segment")
    g.add_edge("segment", "critic")
    g.add_conditional_edges("critic", lambda s: s["decision"],
                            {"accept": "finalize", "retry": "orchestrator", "escalate": "human_review"})
    g.add_edge("human_review", "finalize")
    g.add_edge("finalize", END)
    return g.compile()


def run_one(graph, image: np.ndarray, stem: str = ""):
    return graph.invoke({"image": image, "stem": stem, "attempt": 0, "history": []})
