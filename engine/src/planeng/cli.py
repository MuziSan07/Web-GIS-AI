"""Command line interface to the planning engine.

Thin by design. Every command maps to a library call that the agent layer will
invoke directly, so anything demonstrable here is demonstrable from a prompt.
"""

from __future__ import annotations

import argparse
import sys

from .brief import PlanningBrief, derive_programme
from .standards import load_standards


def _fmt_ha(v: float) -> str:
    return f"{v:>10,.1f}"


def cmd_programme(args: argparse.Namespace) -> int:
    std = load_standards(args.jurisdiction)

    brief = PlanningBrief(
        project_name=args.name,
        jurisdiction=args.jurisdiction,
        site_area_ha=args.area,
        target_population=args.population,
        target_residential_plots=args.plots,
        plan_horizon_years=args.horizon,
        expansion_allowance_pct=args.expansion,
    )

    prog = derive_programme(brief, std)

    print()
    print("=" * 74)
    print(f"  PLANNING PROGRAMME - {brief.project_name}")
    print("=" * 74)
    print(f"  Site area          {brief.site_area_ha:>10,.1f} ha")
    print(f"  Target population  {brief.target_population:>10,}")
    print(f"  Design population  {brief.design_population:>10,}  "
          f"(+{brief.expansion_allowance_pct:.0f}% expansion)")
    print(f"  Gross density      {brief.gross_density_pph:>10,.1f} persons/ha")
    print(f"  Plan horizon       {brief.plan_horizon_years:>10} years")
    print(f"  Ruleset            {std.jurisdiction} v"
          f"{std.data['meta']['ruleset_version']}")

    print()
    print("-" * 74)
    print("  DWELLING SCHEDULE")
    print("-" * 74)
    print(f"  {'Band':<12}{'Plots':>10}{'Plot m2':>10}{'Net ha':>12}{'Gross ha':>12}")
    for d in prog.dwellings:
        print(f"  {d.band:<12}{d.dwellings:>10,}{d.plot_area_typical_m2:>10,.0f}"
              f"{d.net_area_ha:>12,.1f}{d.gross_area_ha:>12,.1f}")
    print(f"  {'TOTAL':<12}{prog.total_dwellings:>10,}{'':>10}"
          f"{sum(d.net_area_ha for d in prog.dwellings):>12,.1f}"
          f"{prog.residential_gross_ha:>12,.1f}")

    print()
    print("-" * 74)
    print("  FACILITY SCHEDULE")
    print("-" * 74)
    current = None
    for f in sorted(prog.facilities, key=lambda x: (x.category, x.key)):
        if f.category != current:
            current = f.category
            print(f"  {current.upper()}")
        catchment = f"{f.catchment_m:,.0f} m" if f.catchment_m else "-"
        print(f"    {f.key:<24}{f.count:>4} x {f.site_area_ha:>5,.2f} ha "
              f"= {f.total_area_ha:>7,.2f} ha   catchment {catchment:>9}")
        if args.verbose:
            print(f"      driver: {f.driver}")
    print(f"    {'TOTAL FACILITY LAND':<24}{'':>4}   {'':>5}    "
          f"{prog.facilities_ha:>9,.2f} ha")

    print()
    print("-" * 74)
    print("  LAND BUDGET")
    print("-" * 74)
    for line in prog.land_budget:
        print(f"  {line.use:<26}{_fmt_ha(line.target_ha)} ha{line.target_pct:>9,.1f} %")
    balance = brief.site_area_ha - prog.allocated_ha
    print(f"  {'-' * 50}")
    print(f"  {'ALLOCATED':<26}{_fmt_ha(prog.allocated_ha)} ha"
          f"{prog.allocated_ha / brief.site_area_ha * 100:>9,.1f} %")
    print(f"  {'UNALLOCATED BALANCE':<26}{_fmt_ha(balance)} ha"
          f"{balance / brief.site_area_ha * 100:>9,.1f} %")

    print()
    print("-" * 74)
    print("  ENGINEERING DEMAND")
    print("-" * 74)
    infra = prog.infrastructure
    print(f"  Households                    {infra['households']:>12,.0f}")
    print(f"  Water, average day            {infra['water_average_day_m3']:>12,.0f} m3/day")
    print(f"  Water, peak day               {infra['water_peak_day_m3']:>12,.0f} m3/day")
    print(f"  Reservoir storage             {infra['reservoir_storage_m3']:>12,.0f} m3")
    print(f"  Sewage, average day           {infra['sewage_average_day_m3']:>12,.0f} m3/day")
    print(f"  Electrical demand             {infra['electrical_demand_mva']:>12,.2f} MVA")
    print(f"  Solid waste                   {infra['solid_waste_tonnes_per_day']:>12,.1f} t/day")

    if prog.warnings:
        print()
        print("=" * 74)
        print("  WARNINGS")
        print("=" * 74)
        for w in prog.warnings:
            print(f"  ! {w}")

    print()
    return 0


def cmd_standards(args: argparse.Namespace) -> int:
    std = load_standards(args.jurisdiction)
    placeholders = std.placeholder_paths()

    print()
    print(f"  Ruleset      {std.path.name}")
    print(f"  Jurisdiction {std.jurisdiction}")
    print(f"  Version      {std.data['meta']['ruleset_version']}")
    print(f"  Status       {std.data['meta']['status']}")
    print(f"  Validated by {std.data['meta']['validated_by'] or '-- NOT VALIDATED --'}")
    print()
    print(f"  {len(placeholders)} rule group(s) awaiting planner confirmation:")
    for p in placeholders:
        print(f"    - {p}")

    if std.issues:
        print()
        print("  Consistency issues:")
        for i in std.issues:
            print(f"    {i}")

    print()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="planeng",
        description="AI-assisted spatial planning engine",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("programme", help="Derive the planning programme from a brief")
    p.add_argument("--name", default="Untitled Settlement")
    p.add_argument("--area", type=float, required=True, help="Site area in hectares")
    p.add_argument("--population", type=int, required=True)
    p.add_argument("--plots", type=int, default=None,
                   help="Explicit residential plot target (overrides population-derived)")
    p.add_argument("--horizon", type=int, default=20)
    p.add_argument("--expansion", type=float, default=15.0)
    p.add_argument("--jurisdiction", default="botswana")
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(func=cmd_programme)

    p = sub.add_parser("standards", help="Inspect a jurisdiction ruleset")
    p.add_argument("--jurisdiction", default="botswana")
    p.set_defaults(func=cmd_standards)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
