"""Canonical Spain autonomous-community (CCAA) maps — single source of truth for region-code lookups.

Codes match the cube's ``AutonomousCommunities`` layer (ISO/INE ordering). Code 5 (Canarias) is intentionally
absent — it falls outside the peninsular+Balearic grid; 0 = no region (sea/outside).

  - ``CCAA_TO_SUBDIV`` : cube code -> ISO-3166-2 subdivision code (for the ``holidays`` library).
  - ``CCAA_NAMES``     : cube code -> display name.

Extracted 2026-07-03 to de-duplicate copies that had drifted across the pipeline (update_edge,
add_engineered_features, daily_job). **NB:** the deployable apps (``space/app.py``,
``docker/monolith/app_live.py``) keep their OWN inline copies on purpose — they ship without ``src/`` on the
path — so import this only from ``src/``-rooted scripts.
"""
from __future__ import annotations

CCAA_TO_SUBDIV = {1: "AN", 2: "AR", 3: "AS", 4: "IB", 6: "CB", 7: "CL", 8: "CM", 9: "CT", 10: "VC",
                  11: "EX", 12: "GA", 13: "MD", 14: "MC", 15: "NC", 16: "PV", 17: "RI"}

CCAA_NAMES = {1: "Andalucía", 2: "Aragón", 3: "Asturias", 4: "Baleares", 6: "Cantabria",
              7: "Castilla y León", 8: "Castilla-La Mancha", 9: "Cataluña", 10: "C. Valenciana",
              11: "Extremadura", 12: "Galicia", 13: "Madrid", 14: "Murcia", 15: "Navarra",
              16: "País Vasco", 17: "La Rioja"}
