use nalgebra::Vector3;

use crate::simulation::propagate::propagate_two_body;

/// Whether an object is an original (intact) satellite/body or a fragment
/// produced by a collision.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ObjectKind {
    Intact,
    Fragment,
}

/// A single object in the catalog, modelled as a sphere with a Cartesian
/// orbital state in an Earth-centered inertial frame.
///
/// Units are kilometers for position, kilometers per second for velocity, and
/// kilograms for mass.
#[derive(Clone, Debug, PartialEq)]
pub struct Object {
    pub id: u64,
    /// Inertial position (km).
    pub pos: Vector3<f64>,
    /// Inertial velocity (km/s).
    pub vel: Vector3<f64>,
    /// Hard-body radius (km); objects are treated as spheres for collisions.
    pub radius: f64,
    /// Mass (kg).
    pub mass: f64,
    pub kind: ObjectKind,
}

impl Object {
    pub fn new(
        id: u64,
        pos: Vector3<f64>,
        vel: Vector3<f64>,
        radius: f64,
        mass: f64,
        kind: ObjectKind,
    ) -> Self {
        Self {
            id,
            pos,
            vel,
            radius,
            mass,
            kind,
        }
    }

    /// Advance this object's state forward (or backward) by `dt` seconds under
    /// two-body Keplerian motion.
    pub fn propagate(&mut self, dt: f64) {
        let (pos, vel) = propagate_two_body(self.pos, self.vel, dt);
        self.pos = pos;
        self.vel = vel;
    }
}
