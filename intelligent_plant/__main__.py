"""python -m intelligent_plant demo  -  run one full autonomous seed-to-patient
cycle on a synthetic world and print what the swarm did and what it is
waiting on humans for."""
from __future__ import annotations

import argparse
from pathlib import Path
import asyncio
import json
import textwrap

from .orchestrator import IntelligentEnterprise
from .simulation import NOW, build_world

ICON = {"executed": "AUTO ", "executed_with_alarm": "ALARM", "executed_notify": "AUTO*", "queued": "QUEUE",
        "info": "INFO ", "vetoed": "VETO ", "refused": "REFUS", "superseded": "SUPER"}


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
    print(f"QAOA 10-var QUBO: P(optimum)={d.quantum.success_probability:.3f} (random: {1/1024:.4f}), "
          f"approx ratio {d.quantum.approximation_ratio:.3f}; gate -> {d.chosen.method}. {d.rationale}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="intelligent-plant",
                                 description="The Intelligent Plant: governed seed-to-patient operations.")
    sub = ap.add_subparsers(dest="command", required=True)
    d = sub.add_parser("demo", help="one autonomous cycle on SYNTHETIC data")
    d.add_argument("--llm", choices=["auto", "on", "off"], default="auto")
    sub.add_parser("quantum", help="quantum algorithm benchmarks vs classical")

    rp = sub.add_parser("review-pack", help="write the clinician and patient message review packets")
    rp.add_argument("--out", default="review")

    c = sub.add_parser("calibrate", help="fit models to this facility's data (CSV exports)")
    c.add_argument("--facility", required=True)
    c.add_argument("--room-log", help="CSV: temp_air,rh,cool,dehum,humid,temp_air_next,rh_next")
    c.add_argument("--volatile", help="CSV: temp_c,time_s,retention (GC-measured terpene retention)")
    c.add_argument("--nutrients", help="CSV: stage,ec,ph,met_quality_target")
    c.add_argument("--sop", help='JSON SOP ranges: {"flowering": {"ec": [1.6, 2.4], "ph": [5.6, 6.4]}, ...}')
    c.add_argument("--out", default="facility_calibration.json")

    for name, helptext in (("signoff", "record YOUR review sign-off (run by the reviewer)"),
                           ("readiness", "show which reviews are still outstanding")):
        x = sub.add_parser(name, help=helptext)
        if name == "signoff":
            x.add_argument("item")
            x.add_argument("--name", required=True)
            x.add_argument("--role", required=True)
            x.add_argument("--statement", required=True)
        x.add_argument("--signoffs", default="signoffs.json")
        x.add_argument("--calibration", default="facility_calibration.json")
        x.add_argument("--regulatory", default=None, help="path to REGULATORY_REVIEW.md (default: docs/)")

    a = ap.parse_args(argv)
    if a.command == "quantum":
        quantum_bench()
    elif a.command == "demo":
        asyncio.run(demo({"auto": None, "on": True, "off": False}[a.llm]))
    elif a.command == "review-pack":
        from .review_pack import build
        for kind, path in build(a.out).items():
            print(f"{kind:9} packet: {path}")
    elif a.command == "calibrate":
        import json as _json
        from .calibration.facility import CalibrationError, calibrate
        try:
            cal = calibrate(a.facility, a.room_log, a.volatile, a.nutrients,
                            _json.loads(Path(a.sop).read_text()) if a.sop else None)
        except CalibrationError as e:
            print(f"CALIBRATION REFUSED: {e}")
            return 2
        cal.write(a.out)
        if cal.room_fit:
            f = cal.room_fit
            print(f"room model: R2 temp {f['r2_temp']:.2f}, RH {f['r2_rh']:.2f}; holdout RMSE "
                  f"{f['holdout_rmse_temp_c']:.2f} C, {f['holdout_rmse_rh_pct']:.2f} %RH ({f['n_holdout']} steps)")
        if cal.volatile_loss:
            v = cal.volatile_loss
            print(f"terpene loss: Ea {v['Ea_J'] / 1000:.1f} kJ/mol (R2 {v['r2']:.2f}); {v['direction']}")
        for chk in cal.nutrient_checks:
            print(f"nutrients [{chk['stage']}]: {chk['note']}")
        for src, rej in cal.rejected_rows.items():
            if rej:
                print(f"{src}: {len(rej)} row(s) rejected at entry, e.g. {rej[0]}")
        print(f"wrote {a.out} - next: facility_engineer and head_grower review and sign off")
    else:
        from .governance.readiness import check_readiness, default_artifacts, record_signoff
        arts = default_artifacts(a.calibration if Path(a.calibration).exists() else None, a.regulatory)
        if a.command == "signoff":
            rec = record_signoff(a.signoffs, a.item, a.name, a.role, a.statement, arts)
            print(f"recorded: {rec['item']} by {rec['name']} ({rec['role']}) on {rec['date']}, "
                  f"fingerprint {rec['sha256'][:16]}")
        report = check_readiness(a.signoffs, arts)
        print(report.summary())
        return 0 if report.ready else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
