//! Randomized generation of catalog objects.
//!
//! Objects are created by sampling classical orbital elements from configurable
//! ranges, converting them to Cartesian state, and attaching a hard-body radius
//! and mass. This seeds the simulation without needing real TLE data.

use std::f64::consts::TAU;
use std::ops::RangeInclusive;

use rand::{Rng, RngExt};

use crate::simulation::elements::ClassicalElements;
use crate::simulation::object::{Object, ObjectKind};
use crate::simulation::constants::EARTH_RADIUS_KM;

/// Ranges from which random object properties are drawn (uniformly).
#[derive(Clone, Debug)]
pub struct SpawnConfig {
    /// Semi-major-axis altitude above Earth's surface (km); `a = R_earth + altitude`.
    pub altitude_km: RangeInclusive<f64>,
    /// Orbital eccentricity.
    pub eccentricity: RangeInclusive<f64>,
    /// Inclination (degrees).
    pub inclination_deg: RangeInclusive<f64>,
    /// Hard-body sphere radius (km).
    pub radius_km: RangeInclusive<f64>,
    /// Mass (kg).
    pub mass_kg: RangeInclusive<f64>,
}

impl Default for SpawnConfig {
    /// A generic low-Earth-orbit population.
    fn default() -> Self {
        Self {
            altitude_km: 300.0..=1200.0,
            eccentricity: 0.0..=0.02,
            inclination_deg: 0.0..=100.0,
            radius_km: 0.001..=0.005, // ~1-5 m spheres
            mass_kg: 10.0..=1000.0,
        }
    }
}

/// Generates random [`Object`]s from a [`SpawnConfig`].
///
/// Ids are supplied by the caller rather than by a counter of the spawner's
/// own, so that spawned objects and collision fragments — which the simulation
/// creates through entirely different paths — draw from the one counter the
/// catalog owns and cannot be issued the same id.
#[derive(Clone, Debug)]
pub struct Spawner {
    config: SpawnConfig,
}

impl Spawner {
    pub fn new(config: SpawnConfig) -> Self {
        Self { config }
    }

    /// Sample a single random intact object, given the id it should carry.
    ///
    /// Private so that `spawn_many` is the only way in: an id chosen by hand
    /// here would not have come from the catalog's counter.
    fn spawn_one<R: Rng + ?Sized>(&self, id: u64, rng: &mut R) -> Object {
        let c = &self.config;
        let a = EARTH_RADIUS_KM + rng.random_range(c.altitude_km.clone());
        let elems = ClassicalElements {
            a,
            e: rng.random_range(c.eccentricity.clone()),
            inclination: rng.random_range(c.inclination_deg.clone()).to_radians(),
            raan: rng.random_range(0.0..=TAU),
            arg_pe: rng.random_range(0.0..=TAU),
            true_anomaly: rng.random_range(0.0..=TAU),
        };
        let (pos, vel) = elems.to_cartesian();

        Object::new(
            id,
            pos,
            vel,
            rng.random_range(c.radius_km.clone()),
            rng.random_range(c.mass_kg.clone()),
            ObjectKind::Intact,
        )
    }

    /// Sample `count` random intact objects, drawing consecutive ids from
    /// `next_id` (typically the catalog's own counter).
    pub fn spawn_many<R: Rng + ?Sized>(
        &self,
        count: usize,
        next_id: &mut u64,
        rng: &mut R,
    ) -> Vec<Object> {
        (0..count)
            .map(|_| {
                let id = *next_id;
                *next_id += 1;
                self.spawn_one(id, rng)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulation::constants::MU_EARTH;
    use crate::simulation::propagate::orbital_period;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn ids_are_unique_and_sequential() {
        let mut rng = StdRng::seed_from_u64(42);
        let spawner = Spawner::new(SpawnConfig::default());
        let mut next_id = 0;
        let objects = spawner.spawn_many(100, &mut next_id, &mut rng);
        for (i, obj) in objects.iter().enumerate() {
            assert_eq!(obj.id, i as u64);
        }
        assert_eq!(next_id, 100);
    }

    #[test]
    fn ids_continue_from_the_supplied_counter() {
        // Spawning against a counter that has already issued ids (fragments,
        // an earlier batch) must not restart the numbering.
        let mut rng = StdRng::seed_from_u64(11);
        let spawner = Spawner::new(SpawnConfig::default());
        let mut next_id = 500;
        let objects = spawner.spawn_many(3, &mut next_id, &mut rng);
        let assigned: Vec<u64> = objects.iter().map(|o| o.id).collect();
        assert_eq!(assigned, vec![500, 501, 502]);
        assert_eq!(next_id, 503);
    }

    #[test]
    fn spawned_objects_respect_config_ranges() {
        let mut rng = StdRng::seed_from_u64(7);
        let config = SpawnConfig::default();
        let spawner = Spawner::new(config.clone());
        let mut next_id = 0;
        for obj in spawner.spawn_many(500, &mut next_id, &mut rng) {
            assert_eq!(obj.kind, ObjectKind::Intact);
            assert!(config.radius_km.contains(&obj.radius));
            assert!(config.mass_kg.contains(&obj.mass));

            // Recover semi-major axis and confirm it is within the altitude band.
            let alpha = 2.0 / obj.pos.norm() - obj.vel.norm_squared() / MU_EARTH;
            let a = 1.0 / alpha;
            let altitude = a - EARTH_RADIUS_KM;
            assert!(
                config.altitude_km.contains(&altitude),
                "altitude {altitude} out of range"
            );

            // Every spawned object should be on a bound (elliptical) orbit.
            assert!(orbital_period(obj.pos, obj.vel).is_some());
        }
    }

    #[test]
    fn deterministic_for_a_given_seed() {
        let spawner = Spawner::new(SpawnConfig::default());
        let a = spawner.spawn_many(10, &mut 0, &mut StdRng::seed_from_u64(1));
        let b = spawner.spawn_many(10, &mut 0, &mut StdRng::seed_from_u64(1));
        assert_eq!(a, b);
    }
}
