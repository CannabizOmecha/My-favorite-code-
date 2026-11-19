"""Processing & QC swarm (Chapters 3, 11): drying, decarboxylation
planning, COA audits, label-claim uncertainty, SPC and batch release.

ctx.observations keys: "drying_rooms", "decarb_requests", "coas" (COA objects),
"market_total_thc", "batch_potency" {cultivar: [total_thc by batch]},
"label_claims" [{product_id, form, P_pct, mass_mg, n_units, eta..., rel_sigmas, finished_product_coa}]
"""
from __future__ import annotations

from ..science.chemistry import plan_decarboxylation
from ..science.quality import audit_coa, label_claim_mg, label_relative_uncertainty, shewhart
from .base import Agent, Context, Proposal

DRY_T = (18.0, 21.0)
DRY_RH = (55.0, 62.0)
AW = (0.55, 0.65)


class DryingCureAgent(Agent):
    name, domain = "drying_cure", "processing"

    async def propose(self, ctx: Context) -> list[Proposal]:
        out = []
        for rid, d in ctx.observations.get("drying_rooms", {}).items():
            issues = []
            if not DRY_T[0] <= d["temp"] <= DRY_T[1]:
                issues.append(f"T {d['temp']} C outside {DRY_T}")
            if not DRY_RH[0] <= d["rh"] <= DRY_RH[1]:
                issues.append(f"RH {d['rh']}% outside {DRY_RH}")
            aw = d.get("water_activity")
            if issues or (aw is not None and not AW[0] <= aw <= AW[1]):
                out.append(Proposal(self.name, "drying_environment",
                                    f"{rid}: " + "; ".join(issues or [f"a_w {aw} outside {AW}"]) +
                                    "; return to 18-21 C / 55-62% RH.",
                                    {"room_id": rid, "setpoint": {"T": 19.5, "rh": 58.0},
                                     "alarm": f"Drying room {rid} out of range: monoterpenes are the first casualty."},
                                    {"safety": 0.3, "accuracy": 0.7}, ["Ch 11.1"]))
        return out


class DecarbPlannerAgent(Agent):
    name, domain = "decarb_planner", "processing"

    async def propose(self, ctx: Context) -> list[Proposal]:
        out = []
        for req in ctx.observations.get("decarb_requests", []):
            plans = plan_decarboxylation(req.get("target_conversion", 0.95), acid=req.get("acid", "THCA"),
                                         temps_c=tuple(range(95, 116, 5)))   # stay near the measured range
            if not plans:
                continue
            p = plans[0]
            out.append(Proposal(self.name, "decarb_plan",
                                f"Batch {req['batch_id']}: {p.temperature_c:.0f} C for {p.time_min:.0f} min -> "
                                f"{p.conversion:.0%} conversion, predicted volatile retention {p.volatile_retention:.0%}.",
                                {"batch_id": req["batch_id"], "plan": p.__dict__,
                                 "alternatives": [x.__dict__ for x in plans[1:3]]},
                                {"safety": 0.2, "accuracy": 0.6, "efficiency": p.volatile_retention},
                                ["Arrhenius fit to Wang 2016 (ref 73): Ea 87.1 kJ/mol"],
                                uncertainty=p.assumption_note))
        return out


class COAAuditAgent(Agent):
    """Audits every COA automatically (ten-point check). Failing COAs put the
    batch on hold (reversible, supervised); release always needs QA."""
    name, domain = "coa_auditor", "processing"

    async def propose(self, ctx: Context) -> list[Proposal]:
        out = []
        market = ctx.observations.get("market_total_thc")
        for coa in ctx.observations.get("coas", []):
            res = audit_coa(coa, market)
            msgs = [m for _, m in res.findings]
            out.append(Proposal(self.name, "coa_audit",
                                f"COA {coa.coa_id} (batch {coa.batch_id}): {'PASS' if res.passed else 'FAIL'}, "
                                f"trust {res.trust_score:.2f}, total THC {coa.total_thc:.2f}%.",
                                {"coa_id": coa.coa_id, "batch_id": coa.batch_id, "passed": res.passed,
                                 "trust": res.trust_score, "findings": msgs, "total_thc": round(coa.total_thc, 3),
                                 "total_cbd": round(coa.total_cbd, 3)},
                                {"safety": 0.5, "accuracy": 1.0}, ["Ch 3.4 ten-point check"]))
            if not res.passed:
                out.append(Proposal(self.name, "batch_hold", f"Hold batch {coa.batch_id}: COA failed audit.",
                                    {"batch_id": coa.batch_id, "reasons": msgs}, {"safety": 1.0}, ["Ch 3.4"],
                                    target=f"batch:{coa.batch_id}:status"))
            else:
                out.append(Proposal(self.name, "batch_release",
                                    f"Batch {coa.batch_id} COA passed; QA may release.",
                                    {"batch_id": coa.batch_id, "warnings": msgs}, {"safety": 0.2, "profitability": 0.6},
                                    ["Ch 3.4", "Ch 11.3"], target=f"batch:{coa.batch_id}:status",
                                    uncertainty=f"COA trust score {res.trust_score:.2f}; lab measurement uncertainty "
                                                f"{coa.uncertainty_pct or 'not stated'}%."))
        return out


class SPCAgent(Agent):
    name, domain = "spc", "processing"

    async def propose(self, ctx: Context) -> list[Proposal]:
        out = []
        for cultivar, series in ctx.observations.get("batch_potency", {}).items():
            if len(series) < 6:
                continue
            r = shewhart(series, baseline_n=max(5, len(series) // 2))
            if r.out_of_control or r.cv > 0.10:
                out.append(Proposal(self.name, "spc_monitoring",
                                    f"{cultivar}: batch potency CV {r.cv:.1%}; out-of-control batches {r.out_of_control}."
                                    " Inconsistent product undermines patient titration.",
                                    {"cultivar": cultivar, "cv": r.cv, "ucl": r.ucl, "lcl": r.lcl,
                                     "out_of_control": r.out_of_control},
                                    {"safety": 0.5, "accuracy": 0.8}, ["Ch 9.3 consistency as a safety trait",
                                                                       "Ch 11.3 SPC"]))
        return out


class LabelClaimAgent(Agent):
    name, domain = "label_claims", "processing"

    async def propose(self, ctx: Context) -> list[Proposal]:
        out = []
        for lc in ctx.observations.get("label_claims", []):
            c = label_claim_mg(lc["P_pct"], lc["mass_mg"], lc["eta_decarb"], lc["eta_extract"], lc["eta_infuse"],
                               lc["n_units"])
            u = label_relative_uncertainty(lc["rel_sigmas"])
            if not lc.get("finished_product_coa"):
                out.append(Proposal(self.name, "batch_hold",
                                    f"{lc['product_id']}: label {c:.1f} mg/unit computed from plant material, +/-{u:.0%} "
                                    f"(1 sigma). Hold until finished-product assay.",
                                    {"batch_id": lc["product_id"], "computed_mg": round(c, 2), "rel_u": round(u, 3),
                                     "interval_1sigma": [round(c * (1 - u), 2), round(c * (1 + u), 2)]},
                                    {"safety": 1.0, "accuracy": 1.0}, ["Ch 11.2", "A.12"],
                                    target=f"batch:{lc['product_id']}:status"))
        return out


def processing_agents() -> list[Agent]:
    return [DryingCureAgent(), DecarbPlannerAgent(), COAAuditAgent(), SPCAgent(), LabelClaimAgent()]
