//! Fragmentation: replacing a collided pair with debris fragments.
//!
//! Every collision is treated as **catastrophic**: both parent objects are
//! destroyed and replaced by `N` fragments whose combined mass equals the
//! parents' combined mass. The model conserves **total mass** and **total
//! linear momentum** exactly, by construction. Kinetic energy is *not*
//! conserved: a hypervelocity impact is highly inelastic, so only a fraction
//! `energy_efficiency` of the center-of-mass kinetic energy is returned as
//! fragment dispersal energy; the remainder is the inelastic (heat/deformation)
//! sink.
//!
//! # Method
//!
//! Let the parents be `(m1, v1)` and `(m2, v2)`, with `M = m1 + m2`.
//!
//! - **Center of mass:** `v_cm = (m1 v1 + m2 v2) / M`; total momentum `M v_cm`.
//! - **Available energy:** in the CM frame, `E_cm = 1/2 mu |v1 - v2|^2`, with
//!   reduced mass `mu = m1 m2 / M`.
//! - **Count:** `N = round(count_coefficient * (E_cm / E_ref)^count_exponent)`,
//!   clamped to `[min_fragments, max_fragments]`. This is *loosely inspired by*,
//!   not a reproduction of, the NASA Standard Breakup Model (see References):
//!   we borrow the ~0.75 power-law exponent but apply it to collision energy,
//!   whereas the real model applies it to mass and uses energy only to classify
//!   a collision as catastrophic.
//! - **Masses:** drawn from a Pareto (power-law) size distribution — many small
//!   fragments, a few large — then normalized so they sum exactly to `M`.
//! - **Velocities:** each fragment gets `v_i = v_cm + dv_i`. Raw isotropic
//!   ejection directions are corrected so `sum(m_i dv_i) = 0` (momentum
//!   conservation) and then uniformly rescaled so the fragments' CM-frame
//!   kinetic energy equals `energy_efficiency * E_cm`. Both operations preserve
//!   the momentum constraint because they are linear in `dv_i`.
//! - **Radii:** derived from mass at the parents' (blended) density, which also
//!   conserves total volume: `r_i = ((m_i / M) * (r1^3 + r2^3))^(1/3)`.
//! - **Positions:** dispersed on a small cloud around the impact point so the
//!   newly-created, otherwise co-located fragments are not immediately flagged
//!   as mutually colliding on the next step.
//!
//! # References
//!
//! - NASA Standard Breakup Model (the inspiration for the power-law count and
//!   size distribution, though simplified here): Johnson, Krisko, Liou &
//!   Anz-Meador, "NASA's New Breakup Model of EVOLVE 4.0," Advances in Space
//!   Research, 28(9), 2001. The full model gives fragment count as
//!   `N(L_c) = 0.1 * M^0.75 * L_c^(-1.71)` (exponent on mass, not energy) and
//!   uses a specific-energy threshold (~40 J/g) to distinguish catastrophic
//!   from non-catastrophic collisions.

use std::f64::consts::TAU;

use nalgebra::Vector3;
use rand::{Rng, RngExt};

use crate::simulation::object::{Object, ObjectKind};

/// Converts `kg * (km/s)^2` to Joules (since `1 (km/s)^2 = 1e6 (m/s)^2`).
const KM2_S2_TO_JOULES: f64 = 1.0e6;

/// Tunable parameters for the catastrophic breakup model.
#[derive(Clone, Debug)]
pub struct FragmentationConfig {
    /// Fraction of CM-frame kinetic energy returned as fragment dispersal
    /// energy; the rest is lost inelastically. In `(0, 1]`.
    pub energy_efficiency: f64,
    /// Coefficient in the fragment-count power law.
    pub count_coefficient: f64,
    /// Exponent in the fragment-count power law (NASA-style ~0.75).
    pub count_exponent: f64,
    /// Reference energy (Joules) that normalizes the count power law.
    pub reference_energy_joules: f64,
    /// Minimum fragments produced by any collision.
    pub min_fragments: usize,
    /// Cap on fragments per collision (keeps the catalog bounded / fast).
    pub max_fragments: usize,
    /// Shape parameter of the Pareto fragment-mass distribution; smaller means
    /// a heavier tail (a few fragments dominate the mass).
    pub mass_pareto_shape: f64,
    /// Radius (km) of the cloud on which fragments are initially dispersed
    /// around the impact point. Small relative to orbital scales, but larger
    /// than hard-body radii to avoid spurious self-collisions.
    pub cloud_radius_km: f64,
}

impl Default for FragmentationConfig {
    /// Defaults tuned for a generic LEO population. They are deliberately
    /// order-of-magnitude choices, not calibrated constants — the goal is
    /// plausible cascade behavior at interactive speed, and every field is
    /// exposed for experiments to sweep.
    ///
    /// A reference point used below: two ~500 kg objects at a 10 km/s relative
    /// speed give a reduced mass `mu = 250 kg` and `E_cm ~ 1.25e10 J`.
    fn default() -> Self {
        Self {
            // Real hypervelocity impacts convert only a small fraction of the
            // impact energy into fragment *translational* KE (most goes to
            // heat, deformation, and comminution). 0.1 keeps ejecta speeds
            // physically modest (hundreds of m/s) while still dispersing the
            // cloud enough to seed secondary collisions.
            energy_efficiency: 0.1,
            // Chosen with reference_energy_joules so the reference collision
            // above yields a few dozen fragments: (1.25e10/1e9)^0.75 ~ 6.6,
            // times 5.0 ~ 33. Sets the overall scale of the count power law.
            count_coefficient: 5.0,
            // ~0.75 borrows the sub-linear power-law exponent from the NASA
            // Standard Breakup Model (where it applies to mass); we apply it to
            // energy instead, so larger collisions make disproportionately more
            // (but not linearly more) debris. See the module-level References.
            count_exponent: 0.75,
            // Normalizer for the count law: ~1 GJ is a moderate LEO impact, so
            // count_coefficient reads directly as "fragments per moderate hit."
            reference_energy_joules: 1.0e9,
            // A "collision" must yield at least a pair, even in the degenerate
            // near-zero-energy case, so the two parents are always replaced.
            min_fragments: 2,
            // Real catastrophic breakups can shed thousands of trackable
            // pieces; we cap at 1000 to bound catalog growth and keep the
            // O(n^2) broad phase fast. Raise this once a spatial broad phase
            // lands and more realism is wanted.
            max_fragments: 1000,
            // Pareto shape for the fragment-mass tail. ~1.6 gives a heavy but
            // finite-variance tail (a few large fragments retain most of the
            // mass, with a long tail of small debris), qualitatively matching
            // observed breakup size distributions.
            mass_pareto_shape: 1.6,
            // Numerical dispersal radius (100 m): far larger than the meter-
            // scale hard-body radii (so freshly-created, otherwise co-located
            // fragments don't self-collide on the next step) yet negligible
            // against orbital length scales (thousands of km).
            cloud_radius_km: 0.1,
        }
    }
}

/// Break a collided pair into debris fragments about `impact_point`.
///
/// Consumes the two parents (the caller is responsible for removing them from
/// the catalog) and returns the fragments, assigning ids from `next_id`.
pub fn fragment_collision<R: Rng + RngExt + ?Sized>(
    a: &Object,
    b: &Object,
    impact_point: Vector3<f64>,
    config: &FragmentationConfig,
    next_id: &mut u64,
    rng: &mut R,
) -> Vec<Object> {
    let total_mass = a.mass + b.mass;
    let v_cm = (a.mass * a.vel + b.mass * b.vel) / total_mass;
    let reduced_mass = a.mass * b.mass / total_mass;
    let v_rel = a.vel - b.vel;
    // CM-frame available energy, in kg*(km/s)^2 and in Joules.
    let e_cm_natural = 0.5 * reduced_mass * v_rel.norm_squared();
    let e_cm_joules = e_cm_natural * KM2_S2_TO_JOULES;

    let n = fragment_count(e_cm_joules, config);

    // Power-law masses, normalized so they sum to the combined parent mass.
    let raw_weights: Vec<f64> = (0..n)
        .map(|_| pareto_sample(config.mass_pareto_shape, rng))
        .collect();
    let weight_sum: f64 = raw_weights.iter().sum();
    let masses: Vec<f64> = raw_weights
        .iter()
        .map(|w| total_mass * w / weight_sum)
        .collect();

    // Isotropic ejection directions, reused for both dispersal and velocity.
    let dirs: Vec<Vector3<f64>> = (0..n).map(|_| random_unit_vector(rng)).collect();

    // Momentum correction: subtract the mass-weighted mean so sum(m_i dv_i) = 0.
    let mut weighted_mean = Vector3::zeros();
    for (dir, &mass) in dirs.iter().zip(&masses) {
        weighted_mean += dir.scale(mass);
    }
    weighted_mean /= total_mass;
    let mut dv: Vec<Vector3<f64>> = dirs.iter().map(|d| d - weighted_mean).collect();

    // Rescale so total CM-frame KE equals energy_efficiency * E_cm.
    let raw_ke: f64 = dv
        .iter()
        .zip(&masses)
        .map(|(d, &mass)| 0.5 * mass * d.norm_squared())
        .sum();
    let target_ke = config.energy_efficiency * e_cm_natural;
    let scale = if raw_ke > 1e-15 && target_ke > 0.0 {
        (target_ke / raw_ke).sqrt()
    } else {
        0.0
    };
    for d in &mut dv {
        *d *= scale;
    }

    // Volume term: total volume is proportional to r1^3 + r2^3 and is conserved
    // because the fragment masses sum to M at a common (blended) density.
    let volume_term = a.radius.powi(3) + b.radius.powi(3);

    (0..n)
        .map(|i| {
            let id = *next_id;
            *next_id += 1;
            let mass = masses[i];
            let radius = ((mass / total_mass) * volume_term).cbrt();
            let pos = impact_point + config.cloud_radius_km * dirs[i];
            let vel = v_cm + dv[i];
            Object::new(id, pos, vel, radius, mass, ObjectKind::Fragment)
        })
        .collect()
}

/// Number of fragments from the energy-scaled power law.
fn fragment_count(e_cm_joules: f64, config: &FragmentationConfig) -> usize {
    let scaled = (e_cm_joules / config.reference_energy_joules).powf(config.count_exponent);
    let n = (config.count_coefficient * scaled).round();
    (n as usize).clamp(config.min_fragments, config.max_fragments)
}

/// Sample a Pareto(shape) value in `[1, inf)` via inverse-CDF; larger values
/// are rarer, giving the "many small, few large" fragment-size distribution.
fn pareto_sample<R: Rng + RngExt + ?Sized>(shape: f64, rng: &mut R) -> f64 {
    let u: f64 = rng.random(); // [0, 1)
    (1.0 - u).powf(-1.0 / shape)
}

/// A direction sampled uniformly over the unit sphere.
fn random_unit_vector<R: Rng + RngExt + ?Sized>(rng: &mut R) -> Vector3<f64> {
    let z: f64 = rng.random_range(-1.0..1.0);
    let phi: f64 = rng.random_range(0.0..TAU);
    let r = (1.0 - z * z).max(0.0).sqrt();
    Vector3::new(r * phi.cos(), r * phi.sin(), z)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    fn parent(id: u64, pos: Vector3<f64>, vel: Vector3<f64>, radius: f64, mass: f64) -> Object {
        Object::new(id, pos, vel, radius, mass, ObjectKind::Intact)
    }

    /// A representative head-on-ish LEO collision.
    fn colliding_pair() -> (Object, Object) {
        let a = parent(1, Vector3::new(7000.0, 0.0, 0.0), Vector3::new(0.0, 7.5, 0.0), 0.003, 800.0);
        let b = parent(2, Vector3::new(7000.0, 0.1, 0.0), Vector3::new(0.0, -7.0, 1.0), 0.002, 400.0);
        (a, b)
    }

    fn fragment(seed: u64) -> (Object, Object, Vec<Object>) {
        let (a, b) = colliding_pair();
        let impact = (a.pos + b.pos) / 2.0;
        let mut rng = StdRng::seed_from_u64(seed);
        let mut next_id = 100;
        let frags = fragment_collision(&a, &b, impact, &FragmentationConfig::default(), &mut next_id, &mut rng);
        (a, b, frags)
    }

    #[test]
    fn conserves_total_mass() {
        let (a, b, frags) = fragment(1);
        let total: f64 = frags.iter().map(|f| f.mass).sum();
        assert_relative_eq!(total, a.mass + b.mass, epsilon = 1e-9);
    }

    #[test]
    fn conserves_linear_momentum() {
        let (a, b, frags) = fragment(2);
        let before = a.mass * a.vel + b.mass * b.vel;
        let after = frags
            .iter()
            .fold(Vector3::zeros(), |acc, f| acc + f.vel.scale(f.mass));
        assert_relative_eq!(after, before, epsilon = 1e-9);
    }

    #[test]
    fn returns_target_fraction_of_energy() {
        let (a, b, frags) = fragment(3);
        let total_mass = a.mass + b.mass;
        let v_cm = (a.mass * a.vel + b.mass * b.vel) / total_mass;
        let reduced = a.mass * b.mass / total_mass;
        let e_cm = 0.5 * reduced * (a.vel - b.vel).norm_squared();

        let frag_ke: f64 = frags
            .iter()
            .map(|f| 0.5 * f.mass * (f.vel - v_cm).norm_squared())
            .sum();
        assert_relative_eq!(
            frag_ke,
            FragmentationConfig::default().energy_efficiency * e_cm,
            epsilon = 1e-9
        );
    }

    #[test]
    fn conserves_total_volume() {
        let (a, b, frags) = fragment(4);
        let frag_vol: f64 = frags.iter().map(|f| f.radius.powi(3)).sum();
        assert_relative_eq!(frag_vol, a.radius.powi(3) + b.radius.powi(3), epsilon = 1e-9);
    }

    #[test]
    fn fragments_are_debris_with_sequential_ids() {
        let (_, _, frags) = fragment(5);
        assert!(frags.len() >= 2);
        for (i, f) in frags.iter().enumerate() {
            assert_eq!(f.kind, ObjectKind::Fragment);
            assert_eq!(f.id, 100 + i as u64);
            assert!(f.mass > 0.0);
            assert!(f.radius > 0.0);
        }
    }

    #[test]
    fn fragments_dispersed_on_cloud_around_impact() {
        let (a, b, frags) = fragment(6);
        let impact = (a.pos + b.pos) / 2.0;
        let cloud = FragmentationConfig::default().cloud_radius_km;
        for f in &frags {
            assert_relative_eq!((f.pos - impact).norm(), cloud, epsilon = 1e-9);
        }
    }

    #[test]
    fn higher_energy_yields_more_fragments() {
        let low = FragmentationConfig::default();
        let n_low = fragment_count(1.0e9, &low);
        let n_high = fragment_count(1.0e11, &low);
        assert!(n_high > n_low, "{n_high} !> {n_low}");
    }

    #[test]
    fn count_is_clamped_to_bounds() {
        let config = FragmentationConfig::default();
        assert_eq!(fragment_count(0.0, &config), config.min_fragments);
        assert_eq!(fragment_count(1.0e300, &config), config.max_fragments);
    }

    #[test]
    fn deterministic_for_a_given_seed() {
        let (_, _, first) = fragment(7);
        let (_, _, second) = fragment(7);
        assert_eq!(first, second);
    }

    #[test]
    fn equal_velocity_impact_has_zero_dispersal() {
        // No relative velocity => no available energy => fragments all move at v_cm.
        let a = parent(1, Vector3::new(7000.0, 0.0, 0.0), Vector3::new(0.0, 7.5, 0.0), 0.003, 800.0);
        let b = parent(2, Vector3::new(7000.0, 0.05, 0.0), Vector3::new(0.0, 7.5, 0.0), 0.002, 400.0);
        let mut rng = StdRng::seed_from_u64(9);
        let mut next_id = 0;
        let frags = fragment_collision(&a, &b, a.pos, &FragmentationConfig::default(), &mut next_id, &mut rng);
        for f in &frags {
            assert_relative_eq!(f.vel, a.vel, epsilon = 1e-12);
        }
    }
}
