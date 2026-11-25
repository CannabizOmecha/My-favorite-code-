"""python -m intelligent_plant demo  -  run one full autonomous seed-to-patient
cycle on a synthetic world and print what the swarm did and what it is
waiting on humans for."""
from __future__ import annotations

import argparse
import asyncio
import json
import textwrap

from .orchestrator import IntelligentEnterprise
from .simulation import NOW, build_world

ICON = {"executed": "AUTO ", "executed_with_alarm": "ALARM", "executed_notify": "AUTO*", "queued": "QUEUE",
        "info": "INFO ", "vetoed": "VETO ", "refused": "REFUS", "superseded": "SUPER",
        "actuation_failed": "FAIL!"}


def _w(s: str, indent: int = 10) -> str:
    return textwrap.fill(s, 110, initial_indent="", subsequent_indent=" " * indent)


async def demo(use_llm: bool | None) -> None:
    ent = IntelligentEnterprise(use_llm=use_llm)
    world = build_world(ent)
    report = await ent.run_cycle(world, now=NOW)
    print("=" * 110)
    print("THE INTELLIGENT PLANT - autonomous cycle on SYNTHETIC data  (Claude reasoning: "
          f"{'on' if ent.llm.available else 'off, deterministic fallback'})")
    print("=" * 110)
    for domain in ("cultivation", "breeding", "processing", "compliance", "clinical", "enterprise"):
        names = {a.name for a in ent.swarm.agents if a.domain == domain}
        ds = [d for d in report.dispositions if d.proposal.agent in names]
        if not ds:
            continue
        print(f"\n## {domain.upper()}")
        for d in ds:
            print(f"  [{ICON[d.outcome]}] " + _w(f"{d.proposal.task}: {d.proposal.summary}"
                                                + (f"  -> {d.detail}" if d.detail else "")))
    print("\n## AWAITING HUMAN DECISION (named accountable owner)")
    for r in ent.approvals.pending():
        print(f"  - [{r.owner_role}] {r.task}: " + _w(r.summary, 6))
    print("\n## ALERTS / ALARMS")
    for a in report.alerts:
        print(f"  - {a['level']:6} {a['source']}: " + _w(a["message"], 6))
    print("\n## ACTUATED STATE")
    print("  setpoints:", json.dumps(ent.state.setpoints, default=str))
    print("  batch status:", ent.state.batch_status, "| quarantined:", sorted(ent.state.quarantined_products))
    print("  work orders:", ent.state.work_orders, "| messages sent:", len(ent.state.outbox))
    print("\n## WHAT THE PATIENT SEES (plain language; internal IDs never shown)")
    for msg in ent.messenger.sent:
        print(f"  [{msg.urgency}] " + _w(msg.text, 12))
    for msg in ent.messenger.held:
        print(f"  [held: {msg.held_reason}] " + _w(msg.text[:90] + "...", 12))
    print("  patient message stats:", ent.messenger.stats())
    ok, _ = ent.store.verify_chain()
    print(f"\n  {ent.store.count()} records in the hash-chained store; chain intact: {ok}")
    if report.agent_failures:
        print("  agent failures:", report.agent_failures)


def quantum_bench() -> None:
    import numpy as np

    from .quantum import algorithms as qa
    from .quantum.advantage import decide
    from .quantum.statevector import bell_pair
    print("Exact state-vector simulation (classical cost grows as 2**n; no speed-up is claimed).\n")
    b = bell_pair()
    print(f"Bell pair: P(00)={b.probabilities()[0]:.2f} P(11)={b.probabilities()[3]:.2f}, "
          f"entanglement entropy {b.entanglement_entropy([0]):.3f} bit")
    v = qa.vqe(qa.H2_HAMILTONIAN)
    print(f"VQE H2: {v.energy + qa.H2_NUCLEAR_REPULSION:.6f} Ha vs exact {v.exact_energy + qa.H2_NUCLEAR_REPULSION:.6f} "
          f"(chemical accuracy: {v.chemical_accuracy}, {v.evaluations} energy evaluations)")
    g = qa.grover_search(12, {1234})
    print(f"Grover N=4096: found {g.found} with P={g.success_probability:.4f} in {g.quantum_oracle_queries} oracle "
          f"queries vs {g.classical_expected_queries:.0f} expected classically")
    rng = np.random.default_rng(0)
    Q = rng.normal(size=(10, 10)); Q = (Q + Q.T) / 2
    d = decide(Q)
    if d.quantum is not None:
        print(f"QAOA 10-var QUBO: P(optimum)={d.quantum.success_probability:.3f} (random: {1/1024:.4f}), "
              f"approx ratio {d.quantum.approximation_ratio:.3f}; gate -> {d.chosen.method}. {d.rationale}")
    else:
        print(f"QAOA 10-var QUBO: no quantum backend; gate -> {d.chosen.method}. {d.rationale}")


def main() -> None:
    ap = argparse.ArgumentParser(prog="intelligent_plant")
    ap.add_argument("command", choices=["demo", "quantum"])
    ap.add_argument("--llm", choices=["auto", "on", "off"], default="auto")
    a = ap.parse_args()
    if a.command == "quantum":
        quantum_bench()
        return
    asyncio.run(demo({"auto": None, "on": True, "off": False}[a.llm]))


if __name__ == "__main__":
    main()
