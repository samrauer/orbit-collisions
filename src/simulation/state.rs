use nalgebra::Vector3;
use rand::Rng;

use crate::simulation::collision::{DetectionReport, detect_collisions};
use crate::simulation::object::Object;
use crate::simulation::spawn::Spawner;

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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulation::object::ObjectKind;
    use crate::simulation::spawn::SpawnConfig;
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

    #[test]
    fn propagate_moves_objects() {
        let mut state = SimulationState::from_objects([sample_object(1)]);
        let before = state.iter().next().unwrap().pos;
        state.propagate(60.0);
        let after = state.iter().next().unwrap().pos;
        assert!((after - before).norm() > 0.0);
    }
}
