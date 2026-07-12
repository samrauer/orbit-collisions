//! Analytical two-body (Keplerian) propagation in Cartesian coordinates.
//!
//! Uses the universal-variable formulation with the `S(z)`/`C(z)` (Stumpff)
//! functions and Lagrange `f`/`g` coefficients (see Bate, Mueller & White,
//! *Fundamentals of Astrodynamics*, Ch. 4). This advances a Cartesian
//! state `(r, v)` forward by an arbitrary time step and works for elliptical,
//! parabolic, and hyperbolic orbits alike.

use nalgebra::Vector3;

use crate::simulation::constants::MU_EARTH;

/// Convergence tolerance and iteration cap for the universal Kepler solve.
const KEPLER_TOL: f64 = 1e-9;
const KEPLER_MAX_ITER: usize = 200;

/// Stumpff function C(z).
fn stumpff_c(z: f64) -> f64 {
    if z > 0.0 {
        let sz = z.sqrt();
        (1.0 - sz.cos()) / z
    } else if z < 0.0 {
        let sz = (-z).sqrt();
        (sz.cosh() - 1.0) / (-z)
    } else {
        0.5
    }
}

/// Stumpff function S(z).
fn stumpff_s(z: f64) -> f64 {
    if z > 0.0 {
        let sz = z.sqrt();
        (sz - sz.sin()) / sz.powi(3)
    } else if z < 0.0 {
        let sz = (-z).sqrt();
        (sz.sinh() - sz) / sz.powi(3)
    } else {
        1.0 / 6.0
    }
}

/// Propagate a Cartesian state `(r0, v0)` forward by `dt` seconds under
/// two-body motion about a body with gravitational parameter [`MU_EARTH`].
///
/// Returns the new `(position, velocity)`. `dt` may be negative to propagate
/// backward in time.
pub fn propagate_two_body(
    r0: Vector3<f64>,
    v0: Vector3<f64>,
    dt: f64,
) -> (Vector3<f64>, Vector3<f64>) {
    if dt == 0.0 {
        return (r0, v0);
    }

    let mu = MU_EARTH;
    let sqrt_mu = mu.sqrt();

    let r0_mag = r0.norm();
    let v0_mag = v0.norm();
    // Radial velocity component times r0: r0 . v0 / sqrt(mu).
    let vr0_term = r0.dot(&v0) / sqrt_mu;
    // alpha = 1/a (reciprocal of semi-major axis). >0 ellipse, 0 parabola, <0 hyperbola.
    let alpha = 2.0 / r0_mag - v0_mag * v0_mag / mu;

    // Solve the universal Kepler equation for the universal anomaly chi.
    let mut chi = sqrt_mu * alpha.abs() * dt;
    for _ in 0..KEPLER_MAX_ITER {
        let z = alpha * chi * chi;
        let c = stumpff_c(z);
        let s = stumpff_s(z);
        let chi2 = chi * chi;
        let f = vr0_term * chi2 * c + (1.0 - alpha * r0_mag) * chi2 * chi * s + r0_mag * chi
            - sqrt_mu * dt;
        let f_prime =
            vr0_term * chi * (1.0 - alpha * chi2 * s) + (1.0 - alpha * r0_mag) * chi2 * c + r0_mag;
        let ratio = f / f_prime;
        chi -= ratio;
        if ratio.abs() < KEPLER_TOL {
            break;
        }
    }

    let z = alpha * chi * chi;
    let c = stumpff_c(z);
    let s = stumpff_s(z);

    // Lagrange f and g coefficients for position.
    let f = 1.0 - chi * chi / r0_mag * c;
    let g = dt - chi.powi(3) / sqrt_mu * s;
    let r = f * r0 + g * v0;
    let r_mag = r.norm();

    // Lagrange fdot and gdot for velocity.
    let f_dot = sqrt_mu / (r_mag * r0_mag) * (alpha * chi.powi(3) * s - chi);
    let g_dot = 1.0 - chi * chi / r_mag * c;
    let v = f_dot * r0 + g_dot * v0;

    (r, v)
}

/// Orbital period (seconds) of a bound state, or `None` if the orbit is not
/// elliptical (parabolic/hyperbolic).
pub fn orbital_period(r0: Vector3<f64>, v0: Vector3<f64>) -> Option<f64> {
    let alpha = 2.0 / r0.norm() - v0.norm_squared() / MU_EARTH;
    if alpha <= 0.0 {
        return None;
    }
    let a = 1.0 / alpha;
    Some(2.0 * std::f64::consts::PI * (a.powi(3) / MU_EARTH).sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// A circular orbit at `altitude` km above a 6378 km Earth radius.
    fn circular_leo(altitude: f64) -> (Vector3<f64>, Vector3<f64>) {
        let r = 6378.0 + altitude;
        let v = (MU_EARTH / r).sqrt();
        (Vector3::new(r, 0.0, 0.0), Vector3::new(0.0, v, 0.0))
    }

    #[test]
    fn zero_dt_is_identity() {
        let (r0, v0) = circular_leo(400.0);
        let (r, v) = propagate_two_body(r0, v0, 0.0);
        assert_eq!(r, r0);
        assert_eq!(v, v0);
    }

    #[test]
    fn full_period_returns_to_start() {
        let (r0, v0) = circular_leo(400.0);
        let period = orbital_period(r0, v0).unwrap();
        let (r, v) = propagate_two_body(r0, v0, period);
        assert_relative_eq!(r, r0, epsilon = 1e-6);
        assert_relative_eq!(v, v0, epsilon = 1e-9);
    }

    #[test]
    fn half_period_is_antipodal() {
        let (r0, v0) = circular_leo(400.0);
        let period = orbital_period(r0, v0).unwrap();
        let (r, _) = propagate_two_body(r0, v0, period / 2.0);
        assert_relative_eq!(r, -r0, epsilon = 1e-6);
    }

    #[test]
    fn quarter_period_circular_geometry() {
        // After a quarter period a circular orbit moves 90 degrees: from
        // (r, 0, 0) to (0, r, 0).
        let (r0, v0) = circular_leo(400.0);
        let r_mag = r0.norm();
        let period = orbital_period(r0, v0).unwrap();
        let (r, _) = propagate_two_body(r0, v0, period / 4.0);
        assert_relative_eq!(r, Vector3::new(0.0, r_mag, 0.0), epsilon = 1e-6);
    }

    #[test]
    fn propagation_conserves_energy_and_momentum() {
        // Eccentric orbit: perigee velocity higher than circular.
        let r0 = Vector3::new(7000.0, 0.0, 0.0);
        let v0 = Vector3::new(0.0, 9.0, 1.0);
        let energy0 = v0.norm_squared() / 2.0 - MU_EARTH / r0.norm();
        let h0 = r0.cross(&v0);

        let (r, v) = propagate_two_body(r0, v0, 1234.0);
        let energy = v.norm_squared() / 2.0 - MU_EARTH / r.norm();
        let h = r.cross(&v);

        assert_relative_eq!(energy, energy0, epsilon = 1e-9);
        assert_relative_eq!(h, h0, epsilon = 1e-6);
    }

    #[test]
    fn forward_then_backward_round_trips() {
        let r0 = Vector3::new(7000.0, 0.0, 0.0);
        let v0 = Vector3::new(0.0, 9.0, 1.0);
        let (r1, v1) = propagate_two_body(r0, v0, 3600.0);
        let (r2, v2) = propagate_two_body(r1, v1, -3600.0);
        assert_relative_eq!(r2, r0, epsilon = 1e-6);
        assert_relative_eq!(v2, v0, epsilon = 1e-9);
    }

    #[test]
    fn hyperbolic_orbit_round_trips() {
        // Speed well above escape at this radius => hyperbolic (alpha < 0).
        let r0 = Vector3::new(7000.0, 0.0, 0.0);
        let v0 = Vector3::new(1.0, 12.0, 0.0);
        assert!(orbital_period(r0, v0).is_none());
        let (r1, v1) = propagate_two_body(r0, v0, 600.0);
        let (r2, v2) = propagate_two_body(r1, v1, -600.0);
        assert_relative_eq!(r2, r0, epsilon = 1e-6);
        assert_relative_eq!(v2, v0, epsilon = 1e-9);
    }
}
