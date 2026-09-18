"""D-Wave Ocean adapters for Q.NILM binary quadratic models."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from time import perf_counter
from typing import Any

import numpy as np

from .qubo import BinaryTemporalQUBO


OCEAN_PACKAGES = (
    "dwave-ocean-sdk",
    "dimod",
    "dwave-samplers",
    "dwave-system",
)


@dataclass(frozen=True)
class OceanSampleResult:
    """Lowest-energy Ocean sample plus auditable execution metadata."""

    sampler: str
    bits: np.ndarray
    energy: float
    occurrences: int
    wall_time_s: float
    metadata: dict[str, Any]
    sampleset: Any


def ocean_versions() -> dict[str, str]:
    """Return installed Ocean package versions for the reproducibility record."""
    versions: dict[str, str] = {}
    for package in OCEAN_PACKAGES:
        try:
            versions[package] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            versions[package] = "not installed"
    return versions


def qubo_dictionary(qubo: BinaryTemporalQUBO) -> dict[tuple[int, int], float]:
    """Return Ocean's QUBO dictionary without the constant objective offset."""
    coefficients: dict[tuple[int, int], float] = {}
    for variable, coefficient in enumerate(qubo.linear):
        if coefficient != 0.0:
            coefficients[(variable, variable)] = float(coefficient)
    for left in range(qubo.n_variables):
        for right in range(left + 1, qubo.n_variables):
            coefficient = float(qubo.quadratic[left, right])
            if coefficient != 0.0:
                coefficients[(left, right)] = coefficient
    return coefficients


def to_dimod_bqm(qubo: BinaryTemporalQUBO) -> Any:
    """Convert Q.NILM to an exactly energy-equivalent dimod BQM."""
    try:
        import dimod
    except ImportError as exc:  # pragma: no cover - exercised without the extra
        raise RuntimeError(
            "D-Wave Ocean is required; install with `pip install -e '.[dwave]'`."
        ) from exc
    return dimod.BinaryQuadraticModel.from_qubo(
        qubo_dictionary(qubo), offset=qubo.constant
    )


def logical_problem_profile(qubo: BinaryTemporalQUBO) -> dict[str, float | int]:
    """Summarize graph structure and coefficient range before QPU embedding."""
    upper = np.triu(qubo.quadratic, 1)
    interaction_locations = np.argwhere(upper != 0.0)
    degrees = np.zeros(qubo.n_variables, dtype=int)
    for left, right in interaction_locations:
        degrees[left] += 1
        degrees[right] += 1

    nonzero = np.concatenate(
        [
            np.abs(qubo.linear[qubo.linear != 0.0]),
            np.abs(upper[upper != 0.0]),
        ]
    )
    possible_edges = qubo.n_variables * (qubo.n_variables - 1) / 2
    minimum = float(np.min(nonzero)) if nonzero.size else 0.0
    maximum = float(np.max(nonzero)) if nonzero.size else 0.0
    return {
        "logical_variables": qubo.n_variables,
        "linear_terms": int(np.count_nonzero(qubo.linear)),
        "quadratic_terms": int(interaction_locations.shape[0]),
        "edge_density": (
            float(interaction_locations.shape[0] / possible_edges)
            if possible_edges
            else 0.0
        ),
        "maximum_degree": int(np.max(degrees)) if degrees.size else 0,
        "mean_degree": float(np.mean(degrees)) if degrees.size else 0.0,
        "coefficient_minimum_absolute": minimum,
        "coefficient_maximum_absolute": maximum,
        "coefficient_dynamic_range": maximum / minimum if minimum else 0.0,
    }


def estimate_zephyr_embedding(
    qubo: BinaryTemporalQUBO,
    *,
    zephyr_m: int = 12,
    seed: int = 7,
    timeout_s: int = 5,
) -> dict[str, Any]:
    """Estimate a minor embedding on an ideal, defect-free Zephyr graph.

    This is a topology-readiness estimate only. Actual physical-qubit use must
    be measured against the selected QPU's live working graph.
    """
    if zephyr_m <= 0 or timeout_s <= 0:
        raise ValueError("zephyr_m and timeout_s must be positive")
    try:
        import minorminer
        from dwave.graphs import zephyr_graph
    except ImportError as exc:  # pragma: no cover - exercised without the extra
        raise RuntimeError(
            "D-Wave Ocean is required; install with `pip install -e '.[dwave]'`."
        ) from exc

    target = zephyr_graph(zephyr_m)
    source_edges = [
        (left, right)
        for left in range(qubo.n_variables)
        for right in range(left + 1, qubo.n_variables)
        if qubo.quadratic[left, right] != 0.0
    ]
    start = perf_counter()
    embedding = minorminer.find_embedding(
        source_edges,
        target.edges,
        random_seed=seed,
        timeout=timeout_s,
        tries=5,
    )
    wall_time = perf_counter() - start
    success = len(embedding) == qubo.n_variables
    chains = [list(embedding[index]) for index in sorted(embedding)] if success else []
    lengths = [len(chain) for chain in chains]
    return {
        "topology": f"ideal Zephyr({zephyr_m})",
        "topology_note": (
            "Defect-free offline estimate; not an embedding on a live D-Wave QPU."
        ),
        "target_qubits": target.number_of_nodes(),
        "target_couplers": target.number_of_edges(),
        "logical_variables": qubo.n_variables,
        "logical_couplers": len(source_edges),
        "success": success,
        "physical_qubits": sum(lengths) if success else None,
        "maximum_chain_length": max(lengths) if success else None,
        "mean_chain_length": float(np.mean(lengths)) if success else None,
        "seed": seed,
        "timeout_s": timeout_s,
        "wall_time_s": wall_time,
        "embedding": {str(index): chain for index, chain in enumerate(chains)},
    }


def _embedding_metadata(sampleset: Any) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    context = sampleset.info.get("embedding_context", {})
    embedding = context.get("embedding")
    if embedding:
        chains = [list(chain) for chain in embedding.values()]
        physical = {qubit for chain in chains for qubit in chain}
        lengths = [len(chain) for chain in chains]
        metadata.update(
            {
                "physical_qubits": len(physical),
                "maximum_chain_length": max(lengths),
                "mean_chain_length": float(np.mean(lengths)),
                "chain_strength": context.get("chain_strength"),
            }
        )
    if (
        sampleset.record.dtype.names
        and "chain_break_fraction" in sampleset.record.dtype.names
    ):
        metadata["mean_chain_break_fraction"] = float(
            np.average(
                sampleset.record.chain_break_fraction,
                weights=sampleset.record.num_occurrences,
            )
        )
    timing = sampleset.info.get("timing")
    if timing:
        metadata["timing"] = _jsonable(timing)
    return metadata


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def serialize_sampleset(sampleset: Any) -> dict[str, Any]:
    """Serialize a SampleSet without opaque byte arrays."""
    return _jsonable(sampleset.to_serializable(use_bytes=False))


def sample_qubo(
    qubo: BinaryTemporalQUBO,
    sampler: str,
    *,
    num_reads: int = 100,
    seed: int | None = None,
    num_sweeps: int = 1000,
    tabu_timeout_ms: int = 100,
    solver: str | None = None,
    annealing_time_us: float | None = None,
    chain_strength: float | None = None,
    hybrid_time_limit_s: float | None = None,
    label: str = "Q.NILM",
) -> OceanSampleResult:
    """Sample a Q.NILM BQM with a local, QPU, or Leap hybrid Ocean solver."""
    if num_reads <= 0:
        raise ValueError("num_reads must be positive")
    bqm = to_dimod_bqm(qubo)
    normalized_sampler = sampler.lower().replace("-", "_")
    start = perf_counter()

    if normalized_sampler == "simulated":
        from dwave.samplers import SimulatedAnnealingSampler

        sampleset = SimulatedAnnealingSampler().sample(
            bqm, num_reads=num_reads, num_sweeps=num_sweeps, seed=seed
        )
    elif normalized_sampler == "tabu":
        from dwave.samplers import TabuSampler

        sampleset = TabuSampler().sample(
            bqm,
            num_reads=num_reads,
            seed=seed,
            timeout=tabu_timeout_ms,
        )
    elif normalized_sampler == "qpu":
        from dwave.system import DWaveSampler, EmbeddingComposite

        selector = solver if solver else {"topology__type": "zephyr"}
        qpu = DWaveSampler(solver=selector)
        embedded = EmbeddingComposite(qpu)
        parameters: dict[str, Any] = {
            "num_reads": num_reads,
            "return_embedding": True,
            "label": label,
        }
        if annealing_time_us is not None:
            parameters["annealing_time"] = annealing_time_us
        if chain_strength is not None:
            parameters["chain_strength"] = chain_strength
        sampleset = embedded.sample(bqm, **parameters)
        backend_metadata = {
            "backend_name": qpu.solver.id,
            "topology": _jsonable(qpu.properties.get("topology", {})),
            "qpu_num_qubits": qpu.properties.get("num_qubits"),
        }
    elif normalized_sampler == "hybrid":
        from dwave.system import LeapHybridSampler

        hybrid = LeapHybridSampler(solver=solver) if solver else LeapHybridSampler()
        parameters = {"label": label}
        if hybrid_time_limit_s is not None:
            parameters["time_limit"] = hybrid_time_limit_s
        sampleset = hybrid.sample(bqm, **parameters)
        backend_metadata = {"backend_name": hybrid.solver.id}
    else:
        raise ValueError(
            "sampler must be one of: simulated, tabu, qpu, or hybrid"
        )

    sampleset.resolve()
    wall_time = perf_counter() - start
    best = sampleset.first
    bits = np.asarray(
        [int(best.sample[variable]) for variable in range(qubo.n_variables)],
        dtype=np.int8,
    )
    verified_energy = qubo.energy(bits)
    if not np.isclose(
        verified_energy,
        float(best.energy),
        rtol=1e-10,
        atol=1e-7 * max(1.0, abs(verified_energy)),
    ):
        raise RuntimeError("Ocean sample energy does not match the Q.NILM objective")
    occurrences = int(getattr(best, "num_occurrences", 1))
    metadata = {
        "sampler": normalized_sampler,
        "wall_time_s": wall_time,
        "ocean_versions": ocean_versions(),
        **(
            backend_metadata
            if normalized_sampler in {"qpu", "hybrid"}
            else {"backend_name": normalized_sampler}
        ),
        **_embedding_metadata(sampleset),
    }
    return OceanSampleResult(
        sampler=normalized_sampler,
        bits=bits,
        energy=verified_energy,
        occurrences=occurrences,
        wall_time_s=wall_time,
        metadata=metadata,
        sampleset=sampleset,
    )
