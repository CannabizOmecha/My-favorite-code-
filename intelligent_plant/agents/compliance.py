"""Compliance swarm (Chapters 12, 14): inventory reconciliation anomalies,
ledger integrity, regulatory change monitoring, hemp rules and 280E.

ctx.observations keys: "inventory" [{item_id, expected_g, recorded_g}],
"regulatory_feed" [{source, old, new}], "hemp_products" [...], "financials" {...}
"""
from __future__ import annotations

from datetime import timedelta

from ..knowledge.regulation import check_hemp_product, diff_regulation, subject_to_280e
from ..science.economics import tax_under_280e
from ..science.stats import robust_z
from .base import Agent, Context, Proposal
from .llm import GROUNDING


class InventoryReconciliationAgent(Agent):
    name, domain = "inventory_reconciliation", "compliance"

    async def propose(self, ctx: Context) -> list[Proposal]:
        inv = ctx.observations.get("inventory", [])
        if len(inv) < 5:
            return []
        disc = [i["recorded_g"] - i["expected_g"] for i in inv]
        z = robust_z(disc)
        out = [Proposal(self.name, "inventory_reconciliation",
                        f"Reconciled {len(inv)} items; total discrepancy {sum(disc):+.1f} g.",
                        {"items": len(inv), "net_discrepancy_g": round(sum(disc), 1)}, {"accuracy": 1.0},
                        ["Ch 12.1"])]
        for item, d, s in zip(inv, disc, z, strict=True):
            if abs(s) > 3.5:
                out.append(Proposal(self.name, "anomaly_investigation",
                                    f"{item['item_id']}: discrepancy {d:+.1f} g (robust z={s:.1f}). Investigate: unit "
                                    f"error, mis-tag, late entry, or diversion.",
                                    {"item_id": item["item_id"], "discrepancy_g": d, "robust_z": round(s, 2)},
                                    {"safety": 0.3, "accuracy": 0.8}, ["Ch 12.1 anomaly detection"],
                                    uncertainty="Statistical flag only; most anomalies are recording errors."))
        return out


class LedgerIntegrityAgent(Agent):
    name, domain = "ledger_integrity", "compliance"

    async def propose(self, ctx: Context) -> list[Proposal]:
        ok, bad = ctx.store.verify_chain()
        if ok:
            return []
        return [Proposal(self.name, "alerting", f"Record chain broken at seq {bad}: retrospective alteration "
                         "detected.", {"first_bad_seq": bad, "alarm": "Ledger tamper evidence"},
                         {"safety": 1.0, "ethics": 1.0, "accuracy": 1.0}, ["Ch 12.2"])]


class RegulatoryMonitorAgent(Agent):
    """Detection and triage are automated; interpretation goes to counsel."""
    name, domain = "regulatory_monitor", "compliance"

    async def propose(self, ctx: Context) -> list[Proposal]:
        out = []
        for item in ctx.observations.get("regulatory_feed", []):
            ch = diff_regulation(item["source"], item["old"], item["new"])
            if not (ch.added or ch.removed):
                continue
            summary = None
            if ctx.llm and ctx.llm.available:
                r = await ctx.llm.complete(
                    GROUNDING + "\nSummarise regulatory changes for an operations team. Label your output as a "
                    "LEAD requiring verification against the primary source.",
                    f"<data>\nsource: {ch.source}\nadded: {ch.added}\nremoved: {ch.removed}\n</data>")
                summary = r.text if r else None
            out.append(Proposal(self.name, "regulatory_monitoring",
                                f"{ch.source}: {len(ch.added)} added / {len(ch.removed)} removed lines; areas "
                                f"{ch.operational_areas or ['unclassified']}.",
                                {"source": ch.source, "areas": ch.operational_areas}, {"accuracy": 0.8}, ["Ch 12.3"]))
            out.append(Proposal(self.name, "regulatory_interpretation",
                                f"Counsel review needed: {ch.source} changed ({', '.join(ch.operational_areas) or 'n/a'}).",
                                {"source": ch.source, "added": ch.added, "removed": ch.removed,
                                 "machine_summary_LEAD_ONLY": summary},
                                {"safety": 0.3, "ethics": 0.2, "accuracy": 0.5},
                                [f"Primary source: {ch.source}"],
                                uncertainty="A language-model summary is a lead, not a legal conclusion."))
        return out


class HempComplianceAgent(Agent):
    name, domain = "hemp_compliance", "compliance"

    def __init__(self, lookahead_days: int = 60):
        self.lookahead_days = lookahead_days

    async def propose(self, ctx: Context) -> list[Proposal]:
        out = []
        for hp in ctx.observations.get("hemp_products", []):
            future = ctx.now.date() + timedelta(days=self.lookahead_days)
            fv = check_hemp_product(ctx.config, future, hp["delta9_pct"], hp["total_thc_pct"],
                                    hp.get("container_total_thc_mg"), hp.get("contains_synthesized", False))
            now_v = check_hemp_product(ctx.config, ctx.now.date(), hp["delta9_pct"], hp["total_thc_pct"],
                                       hp.get("container_total_thc_mg"), hp.get("contains_synthesized", False))
            if now_v.compliant and not fv.compliant:
                out.append(Proposal(self.name, "regulatory_interpretation",
                                    f"{hp['product_id']} is compliant today but will NOT be by {future}: "
                                    + " ".join(fv.reasons) + " Plan reformulation or sell-through.",
                                    {"product_id": hp["product_id"], "reasons": fv.reasons},
                                    {"safety": 0.2, "ethics": 0.3, "accuracy": 0.8, "profitability": 0.5},
                                    ["P.L. 119-37 s.781 (ref 47)", "P.L. 119-103 (ref 48)"],
                                    uncertainty="Statutory dates as of Nov 2026; counsel must confirm.",
                                    target=f"hemp:{hp['product_id']}:upcoming"))
            v = check_hemp_product(ctx.config, ctx.now.date(), hp["delta9_pct"], hp["total_thc_pct"],
                                   hp.get("container_total_thc_mg"), hp.get("contains_synthesized", False))
            if not v.compliant:
                out.append(Proposal(self.name, "hemp_compliance_check",
                                    f"{hp['product_id']} non-compliant as of {ctx.now.date()}: " + " ".join(v.reasons),
                                    {"product_id": hp["product_id"], "reasons": v.reasons, "basis": v.rule_basis,
                                     "action": "quarantine_from_sale"},
                                    {"safety": 0.5, "ethics": 0.5, "accuracy": 0.9},
                                    ["P.L. 119-37 s.781 (ref 47)", "P.L. 119-103 (ref 48)"],
                                    uncertainty="Verify against current law and counsel."))
        return out


class TaxPositionAgent(Agent):
    name, domain = "tax_position", "compliance"

    async def propose(self, ctx: Context) -> list[Proposal]:
        fin = ctx.observations.get("financials")
        if not fin:
            return []
        sub = subject_to_280e(ctx.config, fin["category"])
        t = tax_under_280e(fin["revenue"], fin["cogs"], fin["opex"], subject_to_280e=sub)
        alt = tax_under_280e(fin["revenue"], fin["cogs"], fin["opex"], subject_to_280e=not sub)
        return [Proposal(self.name, "regulatory_interpretation",
                         f"Category {fin['category']}: 280E {'applies' if sub else 'does not apply'}; effective rate "
                         f"{t.effective_rate:.0%} (vs {alt.effective_rate:.0%} otherwise).",
                         {"subject_to_280e": sub, "tax": t.__dict__, "counterfactual": alt.__dict__},
                         {"accuracy": 0.6, "profitability": 0.5}, ["26 U.S.C. 280E (ref 50)", "A.13"],
                         uncertainty="Treasury guidance on Schedule III implementation pending; counsel/CPA decide.")]


def compliance_agents() -> list[Agent]:
    return [InventoryReconciliationAgent(), LedgerIntegrityAgent(), RegulatoryMonitorAgent(), HempComplianceAgent(),
            TaxPositionAgent()]
