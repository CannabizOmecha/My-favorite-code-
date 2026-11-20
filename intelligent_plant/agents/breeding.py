"""Breeding swarm (Chapter 9). ctx.observations["breeding"] =
{"traits": {name: {"X": ndarray, "y": ndarray}}, "candidates": [CandidateCultivar],
 "target_ratio": float, "kinship_mean": float, "seedling_sex_calls": [{id, call, confidence}]}"""
from __future__ import annotations

from ..science.breeding import choose_method, effective_population_size, patient_first_index
from .base import Agent, Context, Proposal

MIN_NE = 50.0


class BreedingStrategist(Agent):
    name, domain = "breeding_strategist", "breeding"

    async def propose(self, ctx: Context) -> list[Proposal]:
        b = ctx.observations.get("breeding")
        if not b:
            return []
        out = []
        for trait, data in b.get("traits", {}).items():
            adv = choose_method(data["X"], data["y"])
            out.append(Proposal(self.name, "breeding_selection",
                                f"Trait '{trait}': use {adv.method} (CV r: GBLUP {adv.cv_accuracy_gblup:.2f}, "
                                f"top-5 MAS {adv.cv_accuracy_mas:.2f}). {adv.reason}",
                                {"trait": trait, **adv.__dict__}, {"accuracy": max(adv.cv_accuracy_gblup,
                                                                                    adv.cv_accuracy_mas)},
                                ["Ch 9.2, Fig 9.1"],
                                uncertainty="Naive CV is inflated when relatives span folds; accuracy is capped by "
                                            "sqrt(heritability)."))
        cands = b.get("candidates", [])
        if cands:
            scored = [(c, patient_first_index(c, b["target_ratio"])) for c in cands]
            eligible = [(c, s) for c, s in scored if s["eligible"]]
            # Lexicographic: consistency (safety) > contamination resistance > terpene stability > efficiency
            eligible.sort(key=lambda cs: (-round(cs[1]["consistency"], 2), -round(cs[1]["contamination_resistance"], 2),
                                          -cs[1]["terpene_stability"], -cs[1]["efficiency"]))
            out.append(Proposal(self.name, "breeding_selection",
                                f"Patient-first ranking for CBD:THC {b['target_ratio']}:1 +/-10%: "
                                f"{[c.name for c, _ in eligible] or 'no eligible candidates'}.",
                                {"ranking": [{"name": c.name, **s} for c, s in eligible],
                                 "ineligible": [c.name for c, s in scored if not s["eligible"]]},
                                {"safety": 0.6, "accuracy": 0.6}, ["Ch 9.3 breeding backward from the patient"],
                                uncertainty="Genotype x environment: chemistry still varies with VPD, light, nutrition."))
        ne = effective_population_size(b.get("kinship_mean", 0.0))
        if ne < MIN_NE:
            out.append(Proposal(self.name, "breeding_selection",
                                f"Effective population size ~{ne:.0f} < {MIN_NE:.0f}: germplasm narrowing. Add "
                                f"unrelated accessions and conserve resistance sources.",
                                {"Ne": ne}, {"safety": 0.4, "ethics": 0.2}, ["Ch 9.4 (refs 130, 131)"],
                                uncertainty="Ne from mean coancestry is an approximation."))
        for s in b.get("seedling_sex_calls", []):
            if s["call"] == "male" and s["confidence"] >= 0.99:
                out.append(Proposal(self.name, "sex_cull_by_marker",
                                    f"Seedling {s['id']}: Y-linked marker = male ({s['confidence']:.3f}). Recommend cull "
                                    f"from sinsemilla rooms.", s, {"efficiency": 0.8}, ["Ch 9.1 (ref 126)"],
                                    uncertainty=f"Marker call confidence {s['confidence']:.3f}."))
        return out


def breeding_agents() -> list[Agent]:
    return [BreedingStrategist()]
