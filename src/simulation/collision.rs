//! Collision detection between propagation steps.
//!
//! We cannot rely on the discrete states alone: with a large step size two
//! objects could pass through each other entirely between samples. Instead each
//! object's motion across a step is approximated by a straight chord from its
//! previous position (start of step) to its current position (end of step).
//!
//! Because both objects are interpolated against the *same* step fraction
//! `s in [0, 1]`, their relative position is linear in `s`, so the closest
//! approach over the step is found by a single quadratic minimization. A
//! collision is registered when that closest approach is within the sum of the
//! two hard-body radii. Pairs that do not collide still contribute their
//! closest approach to the distance statistics.
//!
//! The current broad phase is the naive O(n^2) all-pairs scan; a spatial
//! acceleration structure can be slotted in later without changing the
//! narrow-phase geometry below.

use nalgebra::Vector3;

use crate::simulation::object::Object;

/// The closest approach of two points, each moving linearly across a step.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClosestApproach {
    /// Center-to-center distance at closest approach (km).
    pub separation: f64,
    /// Step fraction in `[0, 1]` at which the closest approach occurs.
    pub fraction: f64,
}

/// Compute the closest approach between two objects over a single step, given
/// each object's start position (previous PVT) and end position (current PVT).
///
/// The relative position is `rel(s) = d0 + s * (d1 - d0)` for `s in [0, 1]`,
/// where `d0`/`d1` are the center-to-center offsets at the start/end of the
/// step. The minimum of `|rel(s)|` is found analytically and the parameter is
/// clamped to the step interval.
pub fn segment_closest_approach(
    a_start: Vector3<f64>,
    a_end: Vector3<f64>,
    b_start: Vector3<f64>,
    b_end: Vector3<f64>,
) -> ClosestApproach {
    let d0 = a_start - b_start;
    let d1 = a_end - b_end;
    let w = d1 - d0; // change in relative position across the step
    let ww = w.dot(&w);

    // If the relative velocity is ~zero, separation is constant over the step.
    let fraction = if ww <= f64::EPSILON {
        0.0
    } else {
        (-d0.dot(&w) / ww).clamp(0.0, 1.0)
    };

    let rel = d0 + fraction * w;
    ClosestApproach {
        separation: rel.norm(),
        fraction,
    }
}

/// A close encounter between two objects during a step.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Encounter {
    pub id_a: u64,
    pub id_b: u64,
    /// Center-to-center distance at closest approach (km).
    pub separation: f64,
    /// `separation - (r_a + r_b)`; negative means the hard bodies overlapped.
    pub surface_gap: f64,
    /// Step fraction in `[0, 1]` at which the closest approach occurred.
    pub fraction: f64,
}

impl Encounter {
    /// Whether this encounter is a collision (hard bodies overlapped).
    pub fn is_collision(&self) -> bool {
        self.surface_gap < 0.0
    }
}

/// The result of scanning one step for collisions and near misses.
#[derive(Clone, Debug, Default)]
pub struct DetectionReport {
    /// All encounters whose closest approach was within the summed radii.
    pub collisions: Vec<Encounter>,
    /// The single closest encounter (by surface gap) across all pairs, if any
    /// pairs were checked.
    pub closest: Option<Encounter>,
    /// Number of object pairs evaluated.
    pub pairs_checked: usize,
}

/// Detect collisions across one propagation step via the naive all-pairs scan.
///
/// `objects` holds the state at the end of the step (used for current position,
/// radius, and id); `previous_positions` holds each object's position at the
/// start of the step, in the same order. Their lengths must match.
pub fn detect_collisions(
    objects: &[Object],
    previous_positions: &[Vector3<f64>],
) -> DetectionReport {
    debug_assert_eq!(
        objects.len(),
        previous_positions.len(),
        "previous_positions must align with objects"
    );

    let mut report = DetectionReport::default();

    for i in 0..objects.len() {
        let a = &objects[i];
        let a_start = previous_positions[i];
        for j in (i + 1)..objects.len() {
            let b = &objects[j];
            let b_start = previous_positions[j];

            let ca = segment_closest_approach(a_start, a.pos, b_start, b.pos);
            let surface_gap = ca.separation - (a.radius + b.radius);

            let encounter = Encounter {
                id_a: a.id,
                id_b: b.id,
                separation: ca.separation,
                surface_gap,
                fraction: ca.fraction,
            };

            report.pairs_checked += 1;
            if encounter.is_collision() {
                report.collisions.push(encounter);
            }
            if report
                .closest
                .is_none_or(|c| encounter.surface_gap < c.surface_gap)
            {
                report.closest = Some(encounter);
            }
        }
    }

    report
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulation::object::ObjectKind;
    use approx::assert_relative_eq;

    fn v(x: f64, y: f64, z: f64) -> Vector3<f64> {
        Vector3::new(x, y, z)
    }

    #[test]
    fn perpendicular_paths_cross_at_midpoint() {
        // A sweeps along x through the origin, B along y through the origin,
        // both reaching it at the same instant (s = 0.5).
        let ca = segment_closest_approach(
            v(-10.0, 0.0, 0.0),
            v(10.0, 0.0, 0.0),
            v(0.0, -10.0, 0.0),
            v(0.0, 10.0, 0.0),
        );
        assert_relative_eq!(ca.separation, 0.0, epsilon = 1e-12);
        assert_relative_eq!(ca.fraction, 0.5, epsilon = 1e-12);
    }

    #[test]
    fn parallel_motion_keeps_constant_separation() {
        // Same velocity, offset by 5 in y: separation constant, no relative motion.
        let ca = segment_closest_approach(
            v(0.0, 0.0, 0.0),
            v(1.0, 0.0, 0.0),
            v(0.0, 5.0, 0.0),
            v(1.0, 5.0, 0.0),
        );
        assert_relative_eq!(ca.separation, 5.0, epsilon = 1e-12);
        assert_eq!(ca.fraction, 0.0);
    }

    #[test]
    fn closest_approach_clamps_to_end_of_step() {
        // A stationary at origin; B approaches from (2,0,0) to (1,0,0) but the
        // nearest point is the end of the step (s = 1).
        let ca = segment_closest_approach(
            v(0.0, 0.0, 0.0),
            v(0.0, 0.0, 0.0),
            v(2.0, 0.0, 0.0),
            v(1.0, 0.0, 0.0),
        );
        assert_relative_eq!(ca.separation, 1.0, epsilon = 1e-12);
        assert_relative_eq!(ca.fraction, 1.0, epsilon = 1e-12);
    }

    fn obj(id: u64, pos: Vector3<f64>, radius: f64) -> Object {
        Object::new(id, pos, v(0.0, 0.0, 0.0), radius, 1.0, ObjectKind::Intact)
    }

    #[test]
    fn detects_collision_when_paths_cross_within_radii() {
        // Two objects whose chords intersect at the origin mid-step; radii sum
        // comfortably exceeds the (zero) closest approach.
        let prev = vec![v(-10.0, 0.0, 0.0), v(0.0, -10.0, 0.0)];
        let objects = vec![obj(1, v(10.0, 0.0, 0.0), 0.5), obj(2, v(0.0, 10.0, 0.0), 0.5)];
        let report = detect_collisions(&objects, &prev);

        assert_eq!(report.pairs_checked, 1);
        assert_eq!(report.collisions.len(), 1);
        let c = report.collisions[0];
        assert_eq!((c.id_a, c.id_b), (1, 2));
        assert!(c.is_collision());
        assert_relative_eq!(c.fraction, 0.5, epsilon = 1e-12);
    }

    #[test]
    fn near_miss_records_stats_without_collision() {
        // Chords cross in x/y but stay 2 km apart in z; radii are tiny.
        let prev = vec![v(-10.0, 0.0, 0.0), v(0.0, -10.0, 2.0)];
        let objects = vec![
            obj(1, v(10.0, 0.0, 0.0), 0.005),
            obj(2, v(0.0, 10.0, 2.0), 0.005),
        ];
        let report = detect_collisions(&objects, &prev);

        assert!(report.collisions.is_empty());
        let closest = report.closest.expect("a pair was checked");
        assert_relative_eq!(closest.separation, 2.0, epsilon = 1e-12);
        assert!(closest.surface_gap > 0.0);
    }

    #[test]
    fn closest_tracks_the_minimum_gap_across_pairs() {
        // Three objects; the (1,3) pair is the closest. All stationary.
        let prev = vec![v(0.0, 0.0, 0.0), v(100.0, 0.0, 0.0), v(1.0, 0.0, 0.0)];
        let objects = vec![
            obj(1, v(0.0, 0.0, 0.0), 0.01),
            obj(2, v(100.0, 0.0, 0.0), 0.01),
            obj(3, v(1.0, 0.0, 0.0), 0.01),
        ];
        let report = detect_collisions(&objects, &prev);

        assert_eq!(report.pairs_checked, 3);
        let closest = report.closest.unwrap();
        assert_eq!((closest.id_a, closest.id_b), (1, 3));
        assert_relative_eq!(closest.separation, 1.0, epsilon = 1e-12);
    }
}
