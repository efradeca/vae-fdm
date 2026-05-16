"""Legal disclaimer and scope clarification for VAE-FDM."""

DISCLAIMER_SHORT = (
    "PRELIMINARY DESIGN ONLY - Verify with licensed PE using validated FEM software."
)

SCOPE_PAPER = """
PAPER-VALIDATED (Pastrana et al., ICLR 2025):
  - Equilibrium of pin-jointed bar systems (residual forces = 0)
  - Shape matching for compression-only shells (Bezier surfaces)
  - Shape matching for cable-net towers (mixed tension/compression)
  - Real-time inference (~1ms on Apple M2)
  - Force density values q per edge (model units)
""".strip()

SCOPE_EXTENSION = """
EXTENSIONS (this tool, NOT paper-validated):
  - Variational encoder for sampling diverse equilibrium solutions
    (shell VAE and tower VAE; tower VAE is experimental with higher
    reconstruction error due to 656-dim latent space)
  - GNN encoder for variable topologies (paper mentions as future work)
  - Classical baseline (L-BFGS-B direct optimization)
  - Interactive 3D explorer (standalone, no Rhino3D required)
  - Export to DXF, OBJ, STL, CSV, JSON, PNG
""".strip()

DISCLAIMER_FULL = f"""
DISCLAIMER - VAE-FDM STRUCTURAL FORM-FINDING TOOL

{SCOPE_PAPER}

{SCOPE_EXTENSION}

IMPORTANT:
  1. EQUILIBRIUM is guaranteed by the FDM decoder (paper-validated).
     Every predicted shape satisfies force balance by construction.
  2. This tool does NOT perform material checks, buckling analysis,
     or serviceability verification.
  3. The paper operates in dimensionless model units.
  4. VAE models trade reconstruction accuracy for solution diversity.
     The deterministic model is recommended for accurate predictions.

REQUIREMENTS FOR CONSTRUCTION:
  - All results must be verified by a licensed Professional Engineer
    (PE/CEng) using validated FEM software (ETABS, SAP2000, RFEM).
  - The engineer of record bears full responsibility for the final
    structural design.

Software: VAE-FDM v1.1.0
Paper: Pastrana et al. (2025) ICLR. arXiv:2409.02606
""".strip()
