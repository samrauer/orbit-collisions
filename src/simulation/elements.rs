//! Classical (Keplerian) orbital elements and their conversion to a Cartesian
//! state vector.
//!
//! The elements-to-state conversion builds the position and velocity in the
//! perifocal frame and rotates them into Earth-centered inertial coordinates
//! via the 3-1-3 (RAAN, inclination, argument of perigee) rotation sequence
//! (see Bate, Mueller & White, *Fundamentals of Astrodynamics*, Ch. 2).

use nalgebra::{Rotation3, Vector3};

use crate::simulation::constants::MU_EARTH;

/// A set of classical orbital elements about the Earth. Angles are in radians.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClassicalElements {
    /// Semi-major axis (km).
    pub a: f64,
    /// Eccentricity (dimensionless).
    pub e: f64,
    /// Inclination (rad).
    pub inclination: f64,
    /// Right ascension of the ascending node (rad).
    pub raan: f64,
    /// Argument of perigee (rad).
    pub arg_pe: f64,
    /// True anomaly (rad).
    pub true_anomaly: f64,
}

impl ClassicalElements {
    /// Convert these elements to an Earth-centered inertial `(position,
    /// velocity)` state, in km and km/s.
    pub fn to_cartesian(&self) -> (Vector3<f64>, Vector3<f64>) {
        let mu = MU_EARTH;
        // Semi-latus rectum.
        let p = self.a * (1.0 - self.e * self.e);
        let (sin_nu, cos_nu) = self.true_anomaly.sin_cos();
        let r_mag = p / (1.0 + self.e * cos_nu);

        // State in the perifocal (PQW) frame.
        let r_pqw = Vector3::new(r_mag * cos_nu, r_mag * sin_nu, 0.0);
        let v_scale = (mu / p).sqrt();
        let v_pqw = Vector3::new(-v_scale * sin_nu, v_scale * (self.e + cos_nu), 0.0);

        // Rotate perifocal -> ECI via the 3-1-3 sequence R_z(raan) R_x(i) R_z(arg_pe).
        let rot = Rotation3::from_axis_angle(&Vector3::z_axis(), self.raan)
            * Rotation3::from_axis_angle(&Vector3::x_axis(), self.inclination)
            * Rotation3::from_axis_angle(&Vector3::z_axis(), self.arg_pe);

        (rot * r_pqw, rot * v_pqw)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn circular_equatorial_at_perigee() {
        // e=0, i=0, all angles 0: expect r along +x and v along +y.
        let a = 7000.0;
        let elems = ClassicalElements {
            a,
            e: 0.0,
            inclination: 0.0,
            raan: 0.0,
            arg_pe: 0.0,
            true_anomaly: 0.0,
        };
        let (r, v) = elems.to_cartesian();
        let v_circ = (MU_EARTH / a).sqrt();
        assert_relative_eq!(r, Vector3::new(a, 0.0, 0.0), epsilon = 1e-9);
        assert_relative_eq!(v, Vector3::new(0.0, v_circ, 0.0), epsilon = 1e-9);
    }

    #[test]
    fn eccentric_perigee_radius_and_perpendicularity() {
        // At true anomaly 0 (perigee), r = a(1-e) and v is perpendicular to r.
        let a = 8000.0;
        let e = 0.2;
        let elems = ClassicalElements {
            a,
            e,
            inclination: 0.5,
            raan: 1.0,
            arg_pe: 2.0,
            true_anomaly: 0.0,
        };
        let (r, v) = elems.to_cartesian();
        assert_relative_eq!(r.norm(), a * (1.0 - e), epsilon = 1e-6);
        assert_relative_eq!(r.dot(&v), 0.0, epsilon = 1e-6);
    }

    #[test]
    fn recovers_semi_major_axis_via_vis_viva() {
        // For any element set, the resulting state must have 1/a matching input.
        let elems = ClassicalElements {
            a: 9000.0,
            e: 0.35,
            inclination: 1.2,
            raan: 0.7,
            arg_pe: 2.5,
            true_anomaly: 1.9,
        };
        let (r, v) = elems.to_cartesian();
        let alpha = 2.0 / r.norm() - v.norm_squared() / MU_EARTH;
        assert_relative_eq!(1.0 / alpha, elems.a, epsilon = 1e-6);
    }

    #[test]
    fn inclination_sets_orbit_normal() {
        // Angular momentum should tilt from +z by the inclination angle.
        let inclination = 0.6;
        let elems = ClassicalElements {
            a: 7500.0,
            e: 0.1,
            inclination,
            raan: 0.0,
            arg_pe: 0.4,
            true_anomaly: 1.1,
        };
        let (r, v) = elems.to_cartesian();
        let h = r.cross(&v);
        let cos_i = h.z / h.norm();
        assert_relative_eq!(cos_i.acos(), inclination, epsilon = 1e-9);
    }
}
