"""Centralized numerical configuration for Track B producers.

The values in this module are normalized to the cavity angular frequency unless
explicitly labeled otherwise.  They reproduce the canonical single-qubit DCE
operating point used by the loss-adjusted QFI artifact.
"""

import math

from dataclasses import dataclass, field

# Physical constants used to convert declared temperatures and cavity
# frequencies into thermal occupations.  Values are CODATA-recommended and are
# kept here so that producers never hardcode physical numbers inline.
HBAR_OVER_KB_S_K = 7.638233147e-12  # hbar/k_B in seconds times kelvin
PLANCK_OVER_KB_S_K = 4.799243073e-11  # h/k_B in seconds times kelvin


@dataclass(frozen=True)
class DCEInterrogationConfig:
    """Configuration for the single-qubit DCE QFI/entropy trade-off study.

    Attributes:
        eta: Static normalized coupling, dimensionless.
        gd: Modulation amplitude normalized by the static coupling,
            dimensionless.
        wq_wc: Qubit-to-cavity angular-frequency ratio, dimensionless.
        wd_wc: Drive-to-cavity angular-frequency ratio, dimensionless.
        n_cav: Cavity Fock-space truncation, dimensionless.
        n_drive_periods: Simulated drive duration, dimensionless periods.
        n_time_points: Number of stored time points, dimensionless.
        delta: Finite-difference step in normalized coupling, dimensionless.
        kappa_values: Relative cavity-loss rates, dimensionless.
        entropy_log_base: Logarithm base for von Neumann entropy, dimensionless.
        min_objective_period: Earliest stored time included in the
            information-rate objective, in drive periods.
    """

    eta: float = 0.150
    gd: float = 0.11923
    wq_wc: float = 1.027
    wd_wc: float = 2.077
    n_cav: int = 20
    n_drive_periods: int = 40
    n_time_points: int = 501
    delta: float = 1.0e-3
    kappa_values: tuple[float, ...] = (
        0.0,
        1.0e-5,
        5.0e-5,
        1.0e-4,
        5.0e-4,
        1.0e-3,
        5.0e-3,
        1.0e-2,
        2.0e-2,
        4.1e-2,
    )
    entropy_log_base: float = 2.0
    min_objective_period: float = 1.0


DEFAULT_DCE_INTERROGATION = DCEInterrogationConfig()


@dataclass(frozen=True)
class QubitCoherenceBenchmarkConfig:
    """Declared qubit-coherence benchmark used by the qubit-loss screen.

    The benchmark is intentionally conservative and is not labeled as
    state-of-the-art.  SI-derived rates are exposed for audit, while the
    historical rates used by the existing numerical artifact are retained as
    explicit inputs until an authorized rerun resolves their discrepancy.

    Attributes:
        T1_us: Energy-relaxation time, in microseconds.
        Tphi_us: Pure-dephasing time, in microseconds.
        reference_frequency_GHz: Cavity frequency used for normalization,
            in gigahertz.
        benchmark_label: Human-readable interpretation of the benchmark.
        current_record_t1_max_ms: Directly verified current-record maximum
            relaxation time, in milliseconds; metadata only and not used by the
            simulation.
        literature_status: Provenance status of the current-record comparison.
        background_citation_key: Background circuit-QED citation key; this is
            not asserted to support the current-record range by itself.
        current_record_citation_key: Direct primary-source citation key for the
            current-record comparison, when added to the bibliography.
        rate_convention: Normalization convention for the active rates.
    """

    T1_us: float = 30.0
    Tphi_us: float = 5.0
    reference_frequency_GHz: float = 6.0
    benchmark_label: str = "conservative transmon-like coherence benchmark"
    current_record_t1_max_ms: float = 1.68
    literature_status: str = (
        "verified direct primary source for T1 up to 1.68 ms; no full range claim"
    )
    background_citation_key: str = "Blais2021"
    current_record_citation_key: str = "Bland2025"
    rate_convention: str = "literal SI coherence time divided by angular cavity frequency"

    def normalized_relaxation_rate(self) -> float:
        """Return the dimensionless relaxation rate ``gamma_1/omega_c``.

        Returns:
            float: Relaxation rate divided by the cavity angular frequency,
                dimensionless.
        """
        return 1.0 / (
            self.T1_us * 1.0e-6 * 2.0 * math.pi * self.reference_frequency_GHz * 1.0e9
        )

    def normalized_dephasing_rate(self) -> float:
        """Return the dimensionless pure-dephasing rate ``gamma_phi/omega_c``.

        Returns:
            float: Pure-dephasing rate divided by the cavity angular frequency,
                dimensionless.
        """
        return 1.0 / (
            self.Tphi_us * 1.0e-6 * 2.0 * math.pi * self.reference_frequency_GHz * 1.0e9
        )

    def simulation_rates(self) -> tuple[float, float]:
        """Return SI-consistent rates for the active Lindblad calculation.

        Returns:
            tuple[float, float]: Relaxation and pure-dephasing rates divided by
                the cavity angular frequency, dimensionless.
        """
        return self.normalized_relaxation_rate(), self.normalized_dephasing_rate()


DEFAULT_QUBIT_COHERENCE = QubitCoherenceBenchmarkConfig()


@dataclass(frozen=True)
class RoundedDCEFigureConfig:
    """Configuration for the rounded-point single-qubit DCE trace figure.

    Attributes:
        eta: Static normalized coupling g0/wc, dimensionless.
        gd: Additive normalized modulation coefficient, dimensionless.
        wq_wc: Qubit-to-cavity angular-frequency ratio, dimensionless.
        wd_wc: Drive-to-cavity angular-frequency ratio, dimensionless.
        n_cav: Cavity Fock-space dimension, dimensionless.
        n_drive_periods: Simulated drive duration, in drive periods.
        n_time_points: Number of stored time points, dimensionless.
        delta_eta: Coupling finite-difference increment, dimensionless.
        solver_nsteps: Maximum internal ODE steps, dimensionless.
    """

    eta: float = 0.150
    gd: float = 0.119
    wq_wc: float = 1.027
    wd_wc: float = 2.077
    n_cav: int = 20
    n_drive_periods: int = 40
    n_time_points: int = 500
    delta_eta: float = 1.0e-3
    solver_nsteps: int = 50000


DEFAULT_ROUNDED_DCE_FIGURE = RoundedDCEFigureConfig()


@dataclass(frozen=True)
class FloquetStabilityConfig:
    """Configuration for the Tier 1 finite-window Floquet stability screen.

    Attributes:
        wq_wc: Qubit-to-cavity angular-frequency ratio, dimensionless.
        eta_values: Static coupling ratios g0/wc, dimensionless.
        gd_over_g0_values: Modulation amplitudes relative to g0, dimensionless.
        drive_ratio_values: Drive-frequency ratios wd/(wc+wq), dimensionless.
        phase_values: Drive phases, in radians.
        kappa_values: Cavity-loss rates divided by wc, dimensionless.
        cavity_truncations: Cavity Fock-space dimensions, dimensionless.
        n_drive_periods: Duration of each stability window, in drive periods.
        n_time_points: Number of stored time samples, dimensionless.
        delta_eta: Coupling finite difference for QFI, dimensionless.
        boundary_fraction: Fraction of the final window used for growth fits,
            dimensionless.
        boundary_slope_tolerance: Relative slope threshold for finite-window
            boundary-growth classification, inverse drive periods.
        truncation_difference_tolerance: Relative difference threshold for the
            truncation-growth flag, dimensionless.
        numerical_eigenvalue_floor: Minimum accepted eigenvalue for numerical
            state diagnostics, dimensionless; this is not exact positivity.
        solver_atol: Absolute ODE tolerance, dimensionless normalized-state
            units.
        solver_rtol: Relative ODE tolerance, dimensionless.
        max_step_fraction: Maximum solver step divided by a drive period,
            dimensionless.
    """

    wq_wc: float = 1.027
    eta_values: tuple[float, ...] = (0.15, 0.20)
    gd_over_g0_values: tuple[float, ...] = (0.10, 0.11923, 0.20)
    drive_ratio_values: tuple[float, ...] = (0.98, 1.00, 1.02)
    phase_values: tuple[float, ...] = (0.0, 1.5707963267948966)
    kappa_values: tuple[float, ...] = (0.0, 1.0e-3, 4.1e-2)
    cavity_truncations: tuple[int, ...] = (12, 20)
    n_drive_periods: int = 40
    n_time_points: int = 201
    delta_eta: float = 1.0e-3
    boundary_fraction: float = 0.25
    boundary_slope_tolerance: float = 1.0e-3
    truncation_difference_tolerance: float = 0.10
    numerical_eigenvalue_floor: float = -1.0e-6
    solver_atol: float = 1.0e-10
    solver_rtol: float = 1.0e-8
    max_step_fraction: float = 0.05


DEFAULT_FLOQUET_STABILITY = FloquetStabilityConfig()


@dataclass(frozen=True)
class MultiparameterGaugeConfig:
    """Configuration for the multiparameter QFI and representation check.

    Attributes:
        static_eta_values: Static normalized couplings g0/wc, dimensionless.
        static_wq_values: Static qubit-to-cavity angular-frequency ratios,
            dimensionless.
        drive_detuning_offsets: Relative offsets from the canonical drive
            frequency, dimensionless.
        eta: Canonical driven coupling g0/wc, dimensionless.
        gd_over_g0: Modulation coefficient in the existing Track B normalized
            convention, dimensionless.
        wq_wc: Canonical qubit-to-cavity angular-frequency ratio, dimensionless.
        wd_wc: Canonical drive-to-cavity angular-frequency ratio, dimensionless.
        phase: Canonical drive phase, in radians.
        cavity_truncations: Cavity Fock-space dimensions, dimensionless.
        selected_gauge_truncation: Larger truncation for the representation
            comparison, dimensionless.
        drive_windows: Driven evolution durations, in drive periods.
        time_points: Number of stored solver times, dimensionless.
        finite_difference_steps: Central finite-difference steps for all
            normalized parameters and phase, dimensionless/radians as
            appropriate.
        solver_atol: Absolute solver tolerance for normalized state evolution.
        solver_rtol: Relative solver tolerance, dimensionless.
        numerical_eigenvalue_floor: Acceptance floor for density diagnostics,
            dimensionless; this is not exact positivity.
        qfi_convergence_tolerance: Relative matrix-change threshold,
            dimensionless.
        gauge_tolerance: Relative representation-consistency threshold,
            dimensionless.
        transform_angle: Fixed cavity phase-rotation angle, in radians.
    """

    static_eta_values: tuple[float, ...] = (0.10, 0.15, 1.00)
    static_wq_values: tuple[float, ...] = (1.027,)
    drive_detuning_offsets: tuple[float, ...] = (-0.02, 0.0, 0.02)
    eta: float = 0.150
    gd_over_g0: float = 0.11923
    wq_wc: float = 1.027
    wd_wc: float = 2.077
    phase: float = 0.0
    cavity_truncations: tuple[int, ...] = (12, 20)
    selected_gauge_truncation: int = 30
    drive_windows: tuple[int, ...] = (6, 10)
    time_points: int = 121
    finite_difference_steps: tuple[float, ...] = (1.0e-2, 1.0e-3, 1.0e-4)
    solver_atol: float = 1.0e-10
    solver_rtol: float = 1.0e-8
    numerical_eigenvalue_floor: float = -1.0e-8
    qfi_convergence_tolerance: float = 5.0e-2
    gauge_tolerance: float = 1.0e-8
    transform_angle: float = 0.37


DEFAULT_MULTIPARAMETER_GAUGE = MultiparameterGaugeConfig()


@dataclass(frozen=True)
class HomodyneConfig:
    """Configuration for the output-field Gaussian homodyne-CFI study.

    Attributes:
        eta: Static normalized coupling g0/wc, dimensionless.
        gd: Additive DCE modulation coefficient in the active convention,
            dimensionless.
        wq_wc: Qubit-to-cavity angular-frequency ratio, dimensionless.
        wd_wc: Drive-to-cavity angular-frequency ratio, dimensionless.
        kappa_values: Total cavity-loss rates divided by wc, dimensionless.
        external_fractions: Monitored external-loss fractions kappa_ext/kappa,
            dimensionless.
        detector_efficiencies: Homodyne detector efficiencies, dimensionless.
        local_oscillator_phases: Homodyne phases, in radians.
        drive_windows: Output-mode integration windows, in drive periods.
        mode_points: Temporal-mode quadrature resolutions, dimensionless.
        cavity_truncations: Cavity Fock-space dimensions, dimensionless.
        finite_difference_steps: Central eta steps, dimensionless.
        n_solver_points: Number of stored propagation times, dimensionless.
        gaussian_state_truncation: Fock dimension for the reconstructed
            single-mode Gaussian reference state, dimensionless.
        solver_atol: Absolute ODE tolerance for normalized states.
        solver_rtol: Relative ODE tolerance, dimensionless.
        max_step_fraction: Maximum ODE step divided by a drive period,
            dimensionless.
        numerical_tolerance: Numerical acceptance tolerance, dimensionless.
        hypothesis_tolerance: Relative tolerance for historical percentage
            reproduction, dimensionless.
    """

    eta: float = 0.150
    gd: float = 0.11923
    wq_wc: float = 1.027
    wd_wc: float = 2.077
    kappa_values: tuple[float, ...] = (0.0, 1.0e-2, 4.1e-2)
    external_fractions: tuple[float, ...] = (0.5, 1.0)
    detector_efficiencies: tuple[float, ...] = (0.5, 0.8, 1.0)
    local_oscillator_phases: tuple[float, ...] = (
        0.0,
        1.5707963267948966,
        3.141592653589793,
        4.71238898038469,
    )
    drive_windows: tuple[int, ...] = (6, 10)
    mode_points: tuple[int, ...] = (8, 12)
    cavity_truncations: tuple[int, ...] = (12, 20)
    finite_difference_steps: tuple[float, ...] = (1.0e-3, 5.0e-4)
    n_solver_points: int = 25
    gaussian_state_truncation: int = 40
    solver_atol: float = 1.0e-9
    solver_rtol: float = 1.0e-7
    max_step_fraction: float = 0.05
    numerical_tolerance: float = 1.0e-7
    hypothesis_tolerance: float = 0.01


DEFAULT_HOMODYNE = HomodyneConfig()


@dataclass(frozen=True)
class ThermalConfig:
    """Configuration for the finite-temperature thermal-competition study.

    Attributes:
        eta: Static normalized coupling g0/wc at the canonical operating
            point, dimensionless.
        gd: Additive DCE modulation coefficient in the existing Track B
            normalized convention, dimensionless (historically labeled
            g_d/g_0).
        wq_wc: Qubit-to-cavity angular-frequency ratio, dimensionless.
        wd_wc: Drive-to-cavity angular-frequency ratio, dimensionless.
        temperatures_mK: Declared bath temperatures, in millikelvin.
        cavity_frequencies_GHz: Declared cavity frequencies nu_c, in gigahertz;
            only used to map temperatures to thermal occupations.
        static_eta_values: Static (undriven) coupling rows, dimensionless.
        kappa_values: Total cavity-loss rates divided by wc, dimensionless.
        qubit_relaxation_rates: Qubit relaxation rates divided by wc for the
            declared qubit-thermalization rows, dimensionless.
        qubit_thermal_frequencies_GHz: Qubit transition frequencies used for
            the qubit thermal occupation, in gigahertz; must be consistent
            with wq_wc at the reference cavity frequency.
        qubit_temperatures_mK: Bath temperatures sampled for the explicit
            qubit-thermalization rows, in millikelvin.
        n_cav: Cavity Fock-space dimension, dimensionless.
        n_drive_periods: Evolution window, in drive periods.
        n_time_points: Number of stored solver times, dimensionless.
        delta_eta: Coupling finite-difference step, dimensionless.
        kappa_reference_GHz: Physical reference cavity frequency for
            converting normalized loss to megahertz, in gigahertz.
        retention_fraction: Sampled retention criterion used to flag the first
            temperature below the criterion, dimensionless.
        solver_atol: Absolute ODE tolerance for normalized states.
        solver_rtol: Relative ODE tolerance, dimensionless.
        max_step_fraction: Maximum ODE step divided by a drive period,
            dimensionless.
        numerical_eigenvalue_floor: Minimum accepted eigenvalue for numerical
            state diagnostics, dimensionless; this is not exact positivity.
    """

    eta: float = 0.150
    gd: float = 0.11923
    wq_wc: float = 1.027
    wd_wc: float = 2.077
    temperatures_mK: tuple[float, ...] = (10.0, 20.0, 50.0, 100.0, 150.0, 200.0)
    cavity_frequencies_GHz: tuple[float, ...] = (5.0, 6.0, 8.0)
    static_eta_values: tuple[float, ...] = (0.15, 1.0)
    kappa_values: tuple[float, ...] = (1.0e-3, 4.1e-2)
    qubit_relaxation_rates: tuple[float, ...] = (1.0e-4,)
    qubit_thermal_frequencies_GHz: tuple[float, ...] = (6.162,)
    qubit_temperatures_mK: tuple[float, ...] = (20.0, 100.0)
    n_cav: int = 20
    n_drive_periods: int = 40
    n_time_points: int = 401
    delta_eta: float = 1.0e-3
    kappa_reference_GHz: float = 6.0
    retention_fraction: float = 0.80
    solver_atol: float = 1.0e-13
    solver_rtol: float = 1.0e-12
    max_step_fraction: float = 0.05
    numerical_eigenvalue_floor: float = -1.0e-6


DEFAULT_THERMAL = ThermalConfig()


@dataclass(frozen=True)
class MultilevelConfig:
    """Configuration for the multilevel-emitter leakage assessment.

    Each platform enters through a declared effective spectrum only; the
    shared dipole-like ladder coupling is documented as a model assumption.

    Attributes:
        transition_ratios_by_model: Ordered level spacings Delta_{e,e+1}/wc
            for each effective emitter model, dimensionless.  The first entry
            is fixed to the canonical qubit ratio 1.027 so that the two-level
            baseline matches the Rabi model.
        static_eta_values: Static (undriven) coupling rows, dimensionless.
        emitter_levels: Emitter Hilbert-space dimensions, dimensionless.
        drive_coefficients: Additive DCE modulation coefficients in the
            existing normalized convention, dimensionless.
        kappa_values: Cavity-loss rates divided by wc, dimensionless.
        eta: Canonical static coupling g0/wc, dimensionless.
        wd_wc: Drive-to-cavity angular-frequency ratio, dimensionless.
        n_cav: Cavity Fock-space dimension, dimensionless.
        n_drive_periods: Driven evolution window, in drive periods.
        n_time_points: Number of stored solver times, dimensionless.
        delta_eta: Coupling finite-difference step, dimensionless.
        leakage_tolerance: Declared computational-subspace leakage criterion,
            dimensionless; a model choice, not a universal validity condition.
        numerical_eigenvalue_floor: Minimum accepted eigenvalue for numerical
            state diagnostics, dimensionless; not exact positivity.
        solver_atol: Absolute ODE tolerance for normalized states.
        solver_rtol: Relative ODE tolerance, dimensionless.
        max_step_fraction: Maximum ODE step divided by a drive period,
            dimensionless.
    """

    transition_ratios_by_model: dict[str, tuple[float, ...]] = field(
        default_factory=lambda: {
            # Weak-anharmonicity transmon-like ladder (alpha/wc=-0.05,
            # approximately -300 MHz at 6 GHz).
            "transmon_like": (1.027, 0.977, 0.927, 0.877),
            # Irregular effective flux-qubit-like spectrum proxy.
            "flux_qubit_like": (1.027, 0.800, 0.650, 1.100),
            # Strongly non-equidistant fluxonium-like spectrum proxy
            # (ground transition 1.027 wc; higher spacings 2.0-3.0 wc).
            "fluxonium_like": (1.027, 2.000, 2.550, 3.000),
        }
    )
    static_eta_values: tuple[float, ...] = (0.15, 1.0)
    emitter_levels: tuple[int, ...] = (2, 3, 4, 5)
    drive_coefficients: tuple[float, ...] = (0.11923, 0.5)
    kappa_values: tuple[float, ...] = (0.0, 4.1e-2)
    eta: float = 0.150
    wd_wc: float = 2.077
    n_cav: int = 12
    n_drive_periods: int = 40
    n_time_points: int = 401
    delta_eta: float = 1.0e-3
    leakage_tolerance: float = 0.01
    gamma1_rel: float = 1.0e-6
    gamma_phi_rel: float = 3.0e-5
    numerical_eigenvalue_floor: float = -1.0e-6
    solver_atol: float = 1.0e-10
    solver_rtol: float = 1.0e-8
    max_step_fraction: float = 0.05



DEFAULT_MULTILEVEL = MultilevelConfig()


@dataclass(frozen=True)
class HeomExtendedConfig:
    """Configuration for the extended HEOM/Lindblad validation screen.

    The rows are local spin-boson models with the canonical frequency ratios
    (wq normalized to 1.027 as in the Rabi model, drive ratio 2.077), coupled
    through sigma-x to a declared Drude-Lorentz bath.  The Lindblad comparator
    is the Markovian weak-coupling limit of the same bath with thermal Pauli
    rates Gamma = 2*pi*J(omega_q) weighted by Bose-Einstein occupations, the
    convention used by the stored local baseline artifact.

    Attributes:
        wq: Two-level transition angular frequency in bath units, 1.027.
        wd: Drive angular frequency in the same units, 2.077 (canonical
            DCE ratio wd/wq = 2.022).
        hierarchy_depths: Hierarchy depths L to sweep, dimensionless.
        temperatures: Normalized bath temperatures k_B T / (hbar wq),
            dimensionless.
        couplings_lam: Drude reorganization couplings lambda, in wq units.
        cutoffs_gamma: Drude cutoff frequencies gamma, in wq units.
        final_times: Final simulation times in inverse wq units; the stored
            local baseline used t_final = 1.0.
        n_time_points: Number of stored solver times, dimensionless.
        baseline_depth: Hierarchy depth used for the non-depth rows,
            dimensionless.
        baseline_lambda: Coupling used for the non-coupling rows, in wq units.
        baseline_gamma: Cutoff used for the non-cutoff rows, in wq units.
        baseline_temperature: Temperature used for the non-temperature rows,
            dimensionless.
        matsurbara_terms: Base Matsubara term count, dimensionless.
        driven_eta_values: Effective driving amplitudes eta_d for driven rows,
            dimensionless.
        driven_ratio: Amplitude-modulation ratio gd/eta_d, dimensionless.
        delta_eta: Finite-difference step for the driven QFI rows,
            dimensionless.
        hierarchy_convergence_tolerance: Relative change tolerance for
            depth-convergence classification, dimensionless.
        solver_atol: Absolute ODE tolerance for the hierarchy.
        solver_rtol: Relative ODE tolerance, dimensionless.
        numerical_eigenvalue_floor: Minimum accepted eigenvalue for numerical
            state diagnostics, dimensionless; not exact positivity.

    The default temperature range starts at 0.07 (about 21 mK equivalent at
    the 6 GHz cavity reference); QuTiP's Matsubara Drude-Lorentz hierarchy
    was numerically unstable (positivity loss) for sigma-x energy relaxation
    below this empirically found threshold in the tested environment, and
    rows that fail the stored positivity gate are retained as inconclusive.
    """

    wq: float = 1.027
    wd: float = 2.077
    hierarchy_depths: tuple[int, ...] = (2, 3, 4, 5)
    temperatures: tuple[float, ...] = (0.07, 0.17, 0.35)
    couplings_lam: tuple[float, ...] = (0.02, 0.1, 0.3)
    cutoffs_gamma: tuple[float, ...] = (10.0, 20.0, 40.0)
    final_times: tuple[float, ...] = (1.0, 2.0, 4.0)
    n_time_points: int = 81
    baseline_depth: int = 4
    baseline_lambda: float = 0.02
    baseline_gamma: float = 20.0
    baseline_temperature: float = 0.07
    matsurbara_terms: int = 2
    driven_eta_values: tuple[float, ...] = (0.10, 0.15)
    driven_ratio: float = 0.11923 / 0.15
    delta_eta: float = 1.0e-3
    hierarchy_convergence_tolerance: float = 0.05
    solver_atol: float = 1.0e-12
    solver_rtol: float = 1.0e-10
    numerical_eigenvalue_floor: float = -1.0e-6


DEFAULT_HEOM_EXTENDED = HeomExtendedConfig()


@dataclass(frozen=True)
class FiniteSizeConfig:
    """Configuration for the larger-N finite-size scaling and resource screen.

    The Dicke Hamiltonian and the ground-state gap-weighted QFI follow the
    recovered canonical conventions (``(g/sqrt(N)) Jx (a+a^dagger)``,
    ``F_Q = 4 sum_n |<0|V|n>|^2/(E_n-E_0)^2`` with ``V=(a+a^dagger)Jx/sqrt(N)``).
    The driven rows use the canonical Dicke+DCE operating point conventions
    from the loss-adjusted producer.

    Attributes:
        static_N_values: Emitter counts N for the static scaling grid,
            dimensionless.
        static_eta: Static coupling g0/wc, dimensionless (1.15 as published).
        static_wq_wc_values: Qubit-to-cavity ratios to sweep, dimensionless.
        static_truncation_pairs: (n_max_low, n_max_high) pairs applied by
            size bracket, dimensionless.
        static_truncation_brackets: N brackets for the truncation pairs,
            dimensionless.
        static_n_eig: Number of eigenstates retained in the gap-weighted sum,
            dimensionless.
        driven_N_values: Emitter counts N for the driven Dicke+DCE rows,
            dimensionless.
        driven_kappa_values: Cavity-loss rates divided by wc, dimensionless.
        driven_n_cav: Cavity Fock-space dimension for driven rows,
            dimensionless.
        driven_eta: Static Dicke coupling for driven rows, dimensionless.
        driven_gd_over_g0: Additive drive amplitude divided by g0 in the
            canonical Dicke convention, dimensionless.
        driven_wq_wc: Qubit-to-cavity ratio for driven rows, dimensionless
            (1.0 as published).
        driven_wd_wc: Drive-to-cavity ratio for driven rows, dimensionless
            (2.0 as published).
        driven_periods: Driven window in drive periods, dimensionless.
        driven_time_points: Number of stored driven times, dimensionless.
        delta_eta: Coupling finite-difference step for QFI, dimensionless.
        fit_N_min: Smallest N included in the power-law fit, dimensionless.
        fit_N_max: Largest N included in the power-law fit, dimensionless.
        boundary_exclusion_sensitivity: Whether the fit is also reported with
            the smallest and largest included points dropped.
        solver_atol: Absolute ODE tolerance for driven rows.
        solver_rtol: Relative ODE tolerance, dimensionless.
    """

    static_N_values: tuple[int, ...] = (4, 6, 8, 9, 12, 15, 16, 18, 21, 24, 27, 30, 32, 33, 36, 40, 48, 56, 64)
    static_eta: float = 1.15
    static_wq_wc_values: tuple[float, ...] = (1.0, 1.005)
    static_truncation_pairs: tuple[tuple[int, int], ...] = ((20, 28), (28, 36), (36, 48), (48, 60))
    static_truncation_brackets: tuple[tuple[int, int], ...] = ((2, 20), (21, 32), (33, 47), (48, 200))
    static_n_eig: int = 25
    driven_N_values: tuple[int, ...] = (4, 6, 8, 12)
    driven_kappa_values: tuple[float, ...] = (0.0, 1.0e-2, 4.1e-2)
    driven_n_cav: int = 6
    driven_eta: float = 1.15
    driven_gd_over_g0: float = 0.15
    driven_wq_wc: float = 1.0
    driven_wd_wc: float = 2.0
    driven_periods: int = 10
    driven_time_points: int = 101
    delta_eta: float = 1.0e-3
    fit_N_min: int = 6
    fit_N_max: int = 18
    boundary_exclusion_sensitivity: bool = True
    solver_atol: float = 1.0e-10
    solver_rtol: float = 1.0e-8


DEFAULT_FINITE_SIZE = FiniteSizeConfig()


@dataclass
class EtaSweepConfig:
    """Configuration for the static Dicke QFI sweeps in coupling at fixed N.

    Produces the panel-(b) data of the finite-size scaling figure: the
    ground-state gap-weighted QFI as a function of the static coupling eta
    for several emitter counts N at fixed frequency ratio and cavity
    truncation. The convention matches the historical sensitivity producer
    (production-quality settings: ``n_max = 35``, ``n_eig = 30``,
    ``wq_wc = 1.005``) so that the eta = 1.15 anchor of each curve is
    cross-consistent with the finite-size grid and the stored Dicke static
    reference.

    Attributes:
        N_values: Emitter counts N for the swept curves, dimensionless.
        eta_min: Smallest sampled coupling g0/wc, dimensionless (0.40,
            below the thermodynamic critical coupling eta_c = 0.5 at
            wq_wc = 1.0, so the onset is visible inside the grid).
        eta_max: Largest sampled coupling g0/wc, dimensionless (1.50).
        eta_points: Number of uniformly sampled coupling values,
            dimensionless (23 points, 0.05 spacing).
        anchor_eta: Coupling at which each curve is cross-checked against
            the finite-size grid and the canonical static reference,
            dimensionless (1.15 as published).
        wq_wc: Qubit-to-cavity angular-frequency ratio, dimensionless
            (1.005, the historical sensitivity-scan convention).
        n_max: Cavity Fock-space dimension, dimensionless (35, the
            production-quality truncation of the historical producer).
        n_eig: Number of eigenstates retained in the gap-weighted sum,
            dimensionless (30, the production-quality setting).
        anchor_tolerance: Maximum allowed relative deviation of the
            eta = anchor_eta QFI from the finite-size grid value,
            dimensionless.
    """

    N_values: tuple[int, ...] = (4, 8, 16, 32)
    eta_min: float = 0.40
    eta_max: float = 1.50
    eta_points: int = 23
    anchor_eta: float = 1.15
    wq_wc: float = 1.005
    n_max: int = 35
    n_eig: int = 30
    anchor_tolerance: float = 5.0e-3


DEFAULT_ETA_SWEEP = EtaSweepConfig()


@dataclass(frozen=True)
class HPCResourceProfile:
    """Bounded scheduler resources for one logical Track B task.

    Attributes:
        cpus: CPU cores requested by one task, dimensionless.
        memory_gb: Memory requested by one task, in gigabytes.
        walltime_minutes: Maximum wall time for one task, in minutes.
        max_concurrent: Maximum simultaneously running array tasks,
            dimensionless.
        partition: Optional cluster partition name.
    """

    cpus: int = 1
    memory_gb: int = 8
    walltime_minutes: int = 60
    max_concurrent: int = 8
    partition: str = ""


@dataclass(frozen=True)
class DickeLossExtensionConfig:
    """Configuration for the larger-truncation collective-loss extension.

    Attributes:
        cavity_truncations: Requested cavity Fock dimensions, dimensionless.
        kappa_values: Cavity-loss rates divided by wc, dimensionless.
        n_qubits: Collective emitter count, dimensionless.
        eta: Static coupling g0/wc, dimensionless.
        gd_over_g0: Additive drive amplitude divided by g0, dimensionless.
        wq_wc: Qubit-to-cavity frequency ratio, dimensionless.
        wd_wc: Drive-to-cavity frequency ratio, dimensionless.
        drive_periods: Evolution window in drive periods, dimensionless.
        time_points: Stored output times, dimensionless.
        delta_eta: Coupling finite-difference increment, dimensionless.
        solver_atol: Absolute ODE tolerance, dimensionless.
        solver_rtol: Relative ODE tolerance, dimensionless.
        numerical_eigenvalue_floor: Accepted numerical eigenvalue floor,
            dimensionless; not a proof of exact positivity.
        late_window_fraction: Fraction of stored samples used for the late mean,
            dimensionless.
        truncation_convergence_tolerance: Relative peak-QFI difference allowed
            between the 16- and 20-dimensional truncations, dimensionless.
        resources: Bounded per-row HPC profile.
    """

    cavity_truncations: tuple[int, ...] = (16, 20, 24)
    kappa_values: tuple[float, ...] = (0.0, 1.0e-2, 4.1e-2)
    n_qubits: int = 32
    eta: float = 1.15
    gd_over_g0: float = 0.15
    wq_wc: float = 1.0
    wd_wc: float = 2.0
    drive_periods: int = 10
    time_points: int = 200
    delta_eta: float = 1.0e-3
    solver_atol: float = 1.0e-10
    solver_rtol: float = 1.0e-10
    numerical_eigenvalue_floor: float = -1.0e-4
    late_window_fraction: float = 0.20
    truncation_convergence_tolerance: float = 0.05
    resources: HPCResourceProfile = field(default_factory=lambda: HPCResourceProfile(memory_gb=24, walltime_minutes=240, max_concurrent=4))


DEFAULT_DICKE_LOSS_EXTENSION = DickeLossExtensionConfig()


@dataclass(frozen=True)
class MultilevelDriveSweepConfig:
    """Configuration for staged multilevel drive-amplitude screening.

    Attributes:
        drive_coefficients: Additive normalized drive amplitudes, dimensionless.
        models: Effective-spectrum model names.
        emitter_levels: Emitter dimensions used in the screen, dimensionless.
        kappa_values: Cavity-loss rates divided by wc, dimensionless.
        eta: Static coupling g0/wc, dimensionless.
        drive_offset: Relative drive-frequency offset, dimensionless.
        leakage_tolerance: Declared adequacy criterion, dimensionless.
        gamma1_rel: Emitter relaxation rate divided by wc, dimensionless.
        gamma_phi_rel: Emitter dephasing rate divided by wc, dimensionless.
        resources: Bounded per-row HPC profile.
    """

    drive_coefficients: tuple[float, ...] = (0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.11923)
    models: tuple[str, ...] = ("transmon_like", "flux_qubit_like", "fluxonium_like")
    emitter_levels: tuple[int, ...] = (3,)
    kappa_values: tuple[float, ...] = (0.0, 4.1e-2)
    eta: float = 0.15
    drive_offset: float = 0.0
    n_cav: int = 12
    drive_periods: int = 40
    time_points: int = 401
    delta_eta: float = 1.0e-3
    leakage_tolerance: float = 0.01
    gamma1_rel: float = 1.0e-6
    gamma_phi_rel: float = 3.0e-5
    solver_atol: float = 1.0e-10
    solver_rtol: float = 1.0e-8
    numerical_eigenvalue_floor: float = -1.0e-6
    resources: HPCResourceProfile = field(default_factory=lambda: HPCResourceProfile(memory_gb=12, walltime_minutes=90, max_concurrent=8))


DEFAULT_MULTILEVEL_DRIVE_SWEEP = MultilevelDriveSweepConfig()


@dataclass(frozen=True)
class CavityHEOMConfig:
    """Configuration for a scoped structured-cavity-bath benchmark.

    Attributes:
        cavity_truncation: Cavity Fock dimension, dimensionless.
        cavity_frequency: Normalized cavity angular frequency, dimensionless.
        hierarchy_depths: HEOM hierarchy depths, dimensionless.
        bath_couplings: Drude reorganization couplings in cavity-frequency units.
        bath_cutoffs: Drude cutoffs in cavity-frequency units.
        temperatures: Normalized bath temperatures, dimensionless.
        final_times: Simulation windows in inverse cavity-frequency units.
        time_points: Stored time points, dimensionless.
        numerical_eigenvalue_floor: Numerical physicality floor, dimensionless.
        resources: Bounded per-row HPC profile.
    """

    cavity_truncation: int = 8
    cavity_frequency: float = 1.0
    hierarchy_depths: tuple[int, ...] = (2, 3, 4)
    bath_couplings: tuple[float, ...] = (0.005, 0.01, 0.02)
    bath_cutoffs: tuple[float, ...] = (10.0, 20.0, 40.0)
    temperatures: tuple[float, ...] = (0.01, 0.05)
    final_times: tuple[float, ...] = (1.0, 2.0)
    time_points: int = 81
    matsubara_terms: int = 2
    numerical_eigenvalue_floor: float = -1.0e-6
    solver_atol: float = 1.0e-10
    solver_rtol: float = 1.0e-8
    resources: HPCResourceProfile = field(default_factory=lambda: HPCResourceProfile(memory_gb=16, walltime_minutes=180, max_concurrent=4))


DEFAULT_CAVITY_HEOM = CavityHEOMConfig()


@dataclass(frozen=True)
class ResourceMatchedConfig:
    """Configuration for scenario-based partial resource comparisons.

    Attributes:
        loss_values: Sampled cavity-loss rates divided by wc, dimensionless.
        observation_periods: Observation windows in drive periods, dimensionless.
        detector_efficiency: Declared detector efficiency, dimensionless.
        readout_overhead: Additive preparation/readout cost in normalized time,
            dimensionless.
        scenarios: Named cost models to evaluate.
        resources: Bounded scheduler profile for optional proxy-generation tasks.
    """

    loss_values: tuple[float, ...] = (0.0, 1.0e-3, 1.0e-2, 4.1e-2)
    observation_periods: tuple[int, ...] = (10, 40)
    detector_efficiency: float = 1.0
    readout_overhead: float = 0.0
    scenarios: tuple[str, ...] = ("photon_matched", "pump_proxy_matched", "time_loss_matched")
    resources: HPCResourceProfile = field(default_factory=lambda: HPCResourceProfile(memory_gb=2, walltime_minutes=10, max_concurrent=16))


DEFAULT_RESOURCE_MATCHED = ResourceMatchedConfig()
