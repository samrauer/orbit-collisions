use std::collections::{HashMap, HashSet};

use nalgebra::Vector3;
use rand::{Rng, RngExt};

use crate::simulation::cluster::group_collisions_into_clusters;
use crate::simulation::collision::{DetectionReport, detect_collisions};
use crate::simulation::fragment::{FragmentationConfig, fragment_cluster};
use crate::simulation::object::Object;
use crate::simulation::spawn::Spawner;

/// One breakup applied during a step: a set of objects destroyed and replaced.
#[derive(Clone, Debug, PartialEq)]
pub struct Breakup {
    /// The objects destroyed, ascending. Two for an ordinary collision, more
    /// when a pileup merged several collisions into one event.
    pub parent_ids: Vec<u64>,
    /// How many fragments replaced them.
    pub fragment_count: usize,
    /// Where the breakup was anchored (km, inertial).
    pub impact_point: Vector3<f64>,
    /// Step fraction in `[0, 1]` at which it was anchored.
    pub fraction: f64,
}

/// What one [`SimulationState::step`] did.
#[derive(Clone, Debug, Default)]
pub struct StepReport {
    /// Collisions and close-approach statistics for the step.
    pub detection: DetectionReport,
    /// The breakups applied, one per pileup.
    pub breakups: Vec<Breakup>,
    /// Total fragments added to the catalog this step.
    pub fragments_created: usize,
}

/// Holds the state of every active debris object / satellite in the
/// simulation.
///
/// The state also owns the catalog's id counter. Uniqueness is a property of
/// the catalog as a whole, and objects join it from two directions — spawning
/// and fragmentation — so one counter living beside the objects is what keeps
/// those paths from issuing the same id twice.
///
/// Ids are never recycled: an object destroyed in a collision leaves its id
/// retired, so an id names one object for the whole run and the parents
/// recorded in a collision stay unambiguous afterwards.
#[derive(Clone, Debug, Default)]
pub struct SimulationState {
    objects: Vec<Object>,
    next_id: u64,
}

impl SimulationState {
    pub fn new() -> Self {
        Self::default()
    }

    /// Build a catalog from existing objects, advancing the counter past every
    /// id they already carry so later allocations cannot duplicate one.
    pub fn from_objects(objects: impl IntoIterator<Item = Object>) -> Self {
        let mut state = Self::new();
        for object in objects {
            state.push(object);
        }
        state
    }

    /// Add an object, reserving its id against future allocations.
    pub fn push(&mut self, object: Object) {
        // `max`, not assignment: an object carrying a lower id must not rewind
        // the counter, or a later allocation would reissue an id already in use.
        self.next_id = self.next_id.max(object.id.saturating_add(1));
        self.objects.push(object);
    }

    /// The id the catalog will assign next.
    pub fn next_id(&self) -> u64 {
        self.next_id
    }

    /// Spawn `count` random objects into the catalog, assigning them fresh ids.
    ///
    /// The counter is deliberately not exposed for mutation on its own: it is
    /// reachable only through methods that also take ownership of the objects
    /// the ids were issued to, so it cannot be advanced — or rewound — by a
    /// caller that never adds anything.
    pub fn spawn<R: Rng + ?Sized>(&mut self, spawner: &Spawner, count: usize, rng: &mut R) {
        let spawned = spawner.spawn_many(count, &mut self.next_id, rng);
        for object in spawned {
            self.push(object);
        }
    }

    pub fn len(&self) -> usize {
        self.objects.len()
    }

    pub fn is_empty(&self) -> bool {
        self.objects.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &Object> {
        self.objects.iter()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut Object> {
        self.objects.iter_mut()
    }

    /// Advance every object forward by `dt` seconds under two-body motion.
    pub fn propagate(&mut self, dt: f64) {
        for object in &mut self.objects {
            object.propagate(dt);
        }
    }

    /// Advance every object by `dt` seconds and detect any collisions or close
    /// approaches that occurred during the step, using the pre- and post-step
    /// positions to catch pass-throughs the discrete states would miss.
    pub fn step_and_detect(&mut self, dt: f64) -> DetectionReport {
        let previous_positions: Vec<Vector3<f64>> =
            self.objects.iter().map(|o| o.pos).collect();
        self.propagate(dt);
        detect_collisions(&self.objects, &previous_positions)
    }

    /// Advance the simulation by one full step: propagate, detect collisions,
    /// and replace everything that collided with its debris.
    ///
    /// This is the cascade loop — the fragments added here are ordinary catalog
    /// objects, so they collide on later steps like anything else.
    pub fn step<R: Rng + RngExt + ?Sized>(
        &mut self,
        dt: f64,
        config: &FragmentationConfig,
        rng: &mut R,
    ) -> StepReport {
        let detection = self.step_and_detect(dt);
        if detection.collisions.is_empty() {
            return StepReport {
                detection,
                ..StepReport::default()
            };
        }

        let index_by_id: HashMap<u64, usize> = self
            .objects
            .iter()
            .enumerate()
            .map(|(index, object)| (object.id, index))
            .collect();

        let mut breakups = Vec::new();
        let mut fragments = Vec::new();
        let mut destroyed: HashSet<u64> = HashSet::new();

        for cluster in group_collisions_into_clusters(&detection.collisions) {
            // Clone the parents out of the catalog before fragmenting: the
            // breakup needs the id counter, which lives in the same struct as
            // the objects, so it cannot borrow from both at once. Clusters hold
            // a handful of objects, so this is cheap.
            let parents: Vec<Object> = cluster
                .parent_ids
                .iter()
                .map(|id| self.objects[index_by_id[id]].clone())
                .collect();

            let mut cluster_fragments = fragment_cluster(
                &parents,
                cluster.seed.impact_point,
                config,
                &mut self.next_id,
                rng,
            );

            // The breakup happened partway through the step, but every other
            // object is already at the end of it. Carry the fragments through
            // the rest of the step so the catalog stays at one instant.
            let remaining = (1.0 - cluster.seed.fraction) * dt;
            for fragment in &mut cluster_fragments {
                fragment.propagate(remaining);
            }

            destroyed.extend(&cluster.parent_ids);
            breakups.push(Breakup {
                parent_ids: cluster.parent_ids,
                fragment_count: cluster_fragments.len(),
                impact_point: cluster.seed.impact_point,
                fraction: cluster.seed.fraction,
            });
            fragments.append(&mut cluster_fragments);
        }

        self.objects.retain(|object| !destroyed.contains(&object.id));
        let fragments_created = fragments.len();
        for fragment in fragments {
            self.push(fragment);
        }

        StepReport {
            detection,
            breakups,
            fragments_created,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulation::object::ObjectKind;
    use crate::simulation::spawn::SpawnConfig;
    use approx::assert_relative_eq;
    use nalgebra::Vector3;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    fn sample_object(id: u64) -> Object {
        let r = 6778.0;
        let v = (crate::simulation::constants::MU_EARTH / r).sqrt();
        Object::new(
            id,
            Vector3::new(r, 0.0, 0.0),
            Vector3::new(0.0, v, 0.0),
            0.005,
            100.0,
            ObjectKind::Intact,
        )
    }

    #[test]
    fn new_state_is_empty() {
        let state = SimulationState::new();
        assert!(state.is_empty());
        assert_eq!(state.len(), 0);
    }

    #[test]
    fn from_objects_collects() {
        let state = SimulationState::from_objects([sample_object(1), sample_object(2)]);
        assert_eq!(state.len(), 2);
    }

    #[test]
    fn push_adds_object() {
        let mut state = SimulationState::new();
        state.push(sample_object(1));
        assert_eq!(state.len(), 1);
        assert!(!state.is_empty());
    }

    #[test]
    fn ids_continue_past_objects_supplied_to_from_objects() {
        let state = SimulationState::from_objects([sample_object(4), sample_object(9)]);
        assert_eq!(state.next_id(), 10);
    }

    #[test]
    fn ids_continue_past_pushed_objects() {
        let mut state = SimulationState::new();
        assert_eq!(state.next_id(), 0);
        state.push(sample_object(41));
        assert_eq!(state.next_id(), 42);
    }

    #[test]
    fn a_lower_id_never_rewinds_the_counter() {
        // The trap in `push`: reserving with assignment rather than `max` would
        // drop the counter back to 6 here, and the next fragment created would
        // be issued an id object 41 already holds.
        let mut state = SimulationState::new();
        state.push(sample_object(41));
        state.push(sample_object(5));
        assert_eq!(state.next_id(), 42);
    }

    #[test]
    fn the_next_id_is_never_one_already_in_the_catalog() {
        // The invariant the counter exists for: objects created mid-run (as
        // fragments are) must not reuse an id already in the catalog, even when
        // it was seeded with out-of-order ids.
        let state = SimulationState::from_objects([sample_object(7), sample_object(3)]);
        let fresh = state.next_id();
        assert!(
            state.iter().all(|o| o.id != fresh),
            "id {fresh} would collide with an existing object"
        );
    }

    #[test]
    fn reserving_past_the_maximum_id_does_not_overflow() {
        // Exhausting u64 is not a real scenario, but the reservation must not
        // wrap to 0 (or panic in debug builds) if it somehow arises. Note that
        // the counter is pinned at the ceiling here and makes no claim that the
        // id it reports is still free — the catalog itself now holds it.
        let mut state = SimulationState::new();
        state.push(sample_object(u64::MAX));
        assert_ne!(state.next_id(), 0, "the counter must not wrap");
    }

    #[test]
    fn spawning_continues_the_catalogs_ids() {
        // Spawning into a catalog that already holds objects must carry on from
        // its counter, not restart, or the new objects would shadow the old.
        let mut state = SimulationState::from_objects([sample_object(7)]);
        let spawner = Spawner::new(SpawnConfig::default());
        state.spawn(&spawner, 4, &mut StdRng::seed_from_u64(3));

        let ids: Vec<u64> = state.iter().map(|o| o.id).collect();
        assert_eq!(ids, vec![7, 8, 9, 10, 11]);
        assert_eq!(state.next_id(), 12);
    }

    /// [`sample_object`]'s counterpart on the far side of the same orbit.
    fn antipode(id: u64) -> Object {
        let mut object = sample_object(id);
        object.pos = -object.pos;
        object.vel = -object.vel;
        object
    }

    /// An object at a LEO-ish state, sized so the fixtures below overlap.
    fn colliding_object(id: u64, offset: Vector3<f64>, vel: Vector3<f64>, mass: f64) -> Object {
        Object::new(
            id,
            Vector3::new(7000.0, 0.0, 0.0) + offset,
            vel,
            0.003,
            mass,
            ObjectKind::Intact,
        )
    }

    /// Two objects overlapping at the start of the step, closing fast.
    fn crossing_pair() -> [Object; 2] {
        [
            colliding_object(1, Vector3::zeros(), Vector3::new(0.0, 7.5, 0.0), 800.0),
            colliding_object(
                2,
                Vector3::new(0.0, 0.002, 0.0),
                Vector3::new(0.0, -7.0, 1.0),
                400.0,
            ),
        ]
    }

    fn total_mass(state: &SimulationState) -> f64 {
        state.iter().map(|o| o.mass).sum()
    }

    #[test]
    fn a_collision_replaces_its_parents_with_fragments() {
        let mut state = SimulationState::from_objects(crossing_pair());
        let mass_before = total_mass(&state);
        let report = state.step(
            30.0,
            &FragmentationConfig::default(),
            &mut StdRng::seed_from_u64(1),
        );

        assert_eq!(report.breakups.len(), 1);
        assert_eq!(report.breakups[0].parent_ids, vec![1, 2]);
        assert_eq!(report.breakups[0].fragment_count, report.fragments_created);

        // Both parents are gone and every remaining object is debris.
        assert!(state.iter().all(|o| o.id != 1 && o.id != 2));
        assert!(state.iter().all(|o| o.kind == ObjectKind::Fragment));
        assert_eq!(state.len(), report.fragments_created);
        assert_relative_eq!(total_mass(&state), mass_before, epsilon = 1e-9);
    }

    #[test]
    fn a_pileup_becomes_one_breakup_naming_every_parent() {
        // Three mutually overlapping objects produce three pairwise collisions.
        // They must resolve as a single breakup of all three, not as separate
        // events that would destroy a shared object twice and duplicate mass.
        let mut state = SimulationState::from_objects([
            colliding_object(1, Vector3::zeros(), Vector3::new(0.0, 7.5, 0.0), 800.0),
            colliding_object(
                2,
                Vector3::new(0.0, 0.002, 0.0),
                Vector3::new(0.0, -7.0, 1.0),
                400.0,
            ),
            colliding_object(
                3,
                Vector3::new(0.0, 0.001, 0.001),
                Vector3::new(1.0, 0.5, -7.2),
                250.0,
            ),
        ]);
        let mass_before = total_mass(&state);
        let report = state.step(
            30.0,
            &FragmentationConfig::default(),
            &mut StdRng::seed_from_u64(2),
        );

        assert_eq!(report.detection.collisions.len(), 3, "three pairwise hits");
        assert_eq!(report.breakups.len(), 1, "merged into one breakup");
        assert_eq!(report.breakups[0].parent_ids, vec![1, 2, 3]);
        assert_relative_eq!(total_mass(&state), mass_before, epsilon = 1e-9);
    }

    #[test]
    fn a_step_without_collisions_changes_nothing_but_position() {
        // Antipodal on the same circular orbit, so they stay half an orbit
        // apart and never come near each other.
        let mut state = SimulationState::from_objects([sample_object(1), antipode(2)]);
        let next_id_before = state.next_id();
        let report = state.step(
            30.0,
            &FragmentationConfig::default(),
            &mut StdRng::seed_from_u64(3),
        );

        assert!(report.breakups.is_empty());
        assert_eq!(report.fragments_created, 0);
        assert_eq!(state.len(), 2);
        assert_eq!(state.next_id(), next_id_before);
    }

    #[test]
    fn fragments_are_carried_to_the_end_of_the_step() {
        // The breakup is anchored partway through the step, but the catalog is
        // already at the end of it, so fragments must be propagated the rest of
        // the way rather than left at the impact point.
        let dt = 30.0;
        let mut state = SimulationState::from_objects(crossing_pair());
        let report = state.step(
            dt,
            &FragmentationConfig::default(),
            &mut StdRng::seed_from_u64(4),
        );

        let breakup = &report.breakups[0];
        assert!(breakup.fraction < 1.0, "nothing left of the step to travel");

        // Left unpropagated they would all sit exactly cloud_radius_km away.
        let cloud = FragmentationConfig::default().cloud_radius_km;
        let travelled = (1.0 - breakup.fraction) * dt;
        for fragment in state.iter() {
            let drift = (fragment.pos - breakup.impact_point).norm();
            assert!(
                drift > cloud * 10.0,
                "fragment {} only moved {drift} km in {travelled} s",
                fragment.id
            );
        }
    }

    #[test]
    fn fragment_ids_continue_from_the_catalog_counter() {
        // Fragments join the catalog mid-run, so their ids must come from the
        // same counter the parents' did — including ids the parents retired.
        let mut state = SimulationState::from_objects(crossing_pair());
        let mut rng = StdRng::seed_from_u64(5);
        state.step(30.0, &FragmentationConfig::default(), &mut rng);

        let ids: Vec<u64> = state.iter().map(|o| o.id).collect();
        assert!(ids.iter().all(|&id| id >= 2), "reused a retired parent id");

        let mut unique = ids.clone();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(unique.len(), ids.len(), "duplicate fragment ids");
        assert_eq!(state.next_id(), ids.iter().max().unwrap() + 1);
    }

    #[test]
    fn propagate_moves_objects() {
        let mut state = SimulationState::from_objects([sample_object(1)]);
        let before = state.iter().next().unwrap().pos;
        state.propagate(60.0);
        let after = state.iter().next().unwrap().pos;
        assert!((after - before).norm() > 0.0);
    }
}
