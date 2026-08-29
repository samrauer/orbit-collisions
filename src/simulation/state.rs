use std::collections::{HashMap, HashSet};

use nalgebra::Vector3;
use rand::{Rng, RngExt};

use crate::simulation::cluster::group_collisions_into_clusters;
use crate::simulation::collision::{DetectionReport, detect_collisions};
use crate::simulation::constants::{EARTH_RADIUS_KM, REENTRY_ALTITUDE_KM};
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

/// One object removed after descending below the reentry altitude.
///
/// The whole object is kept, not just its id: its mass is what makes the
/// catalog's mass balance checkable once objects can leave (surviving mass plus
/// re-entered mass is conserved, where surviving mass alone no longer is), and
/// its [`ObjectKind`](crate::simulation::ObjectKind) records whether what came
/// down was debris or an intact satellite.
#[derive(Clone, Debug, PartialEq)]
pub struct Reentry {
    /// The object as it was when it was removed, at the end of the step.
    pub object: Object,
    /// Elapsed simulation time (seconds) at the end of that step.
    pub elapsed_seconds: f64,
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
    /// The objects that re-entered and were removed this step.
    pub reentries: Vec<Reentry>,
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
///
/// # Reentry
///
/// Objects leave the catalog two ways: destroyed in a collision, or removed
/// after descending below [`REENTRY_ALTITUDE_KM`]. The state keeps a log of the
/// latter for the whole run, so a finished run can be asked what came down and
/// when — and so the mass that left the catalog can still be accounted for.
///
/// Removal is by *current position*, not by perigee. An object on a doomed
/// orbit is still physically up in the shell while it descends, and must stay
/// collidable the whole way down — that descent through the traffic is exactly
/// where a cascade does its damage, so culling it the moment its perigee dips
/// would quietly delete the interesting case.
#[derive(Clone, Debug, Default)]
pub struct SimulationState {
    objects: Vec<Object>,
    next_id: u64,
    elapsed_seconds: f64,
    reentries: Vec<Reentry>,
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

    /// Simulation time elapsed (seconds) since the catalog was created.
    pub fn elapsed_seconds(&self) -> f64 {
        self.elapsed_seconds
    }

    /// Every object that has re-entered so far, in the order they came down.
    pub fn reentries(&self) -> &[Reentry] {
        &self.reentries
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
    ///
    /// The clock is advanced here rather than in [`Self::step`] because this is
    /// the one choke point every advance funnels through, so the elapsed time
    /// stays correct whichever entry point a caller uses.
    pub fn propagate(&mut self, dt: f64) {
        for object in &mut self.objects {
            object.propagate(dt);
        }
        self.elapsed_seconds += dt;
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
    /// replace everything that collided with its debris, and remove whatever
    /// has descended into the atmosphere.
    ///
    /// This is the cascade loop — the fragments added here are ordinary catalog
    /// objects, so they collide on later steps like anything else.
    ///
    /// The reentry sweep runs *last*, after detection and fragmentation, for
    /// two reasons. Removing objects before detection would desynchronize the
    /// catalog from the previous-position slice that
    /// [`detect_collisions`] indexes in parallel with it. And sweeping at the
    /// end means an object still collides during the step it comes down on,
    /// and that fragments born below the floor leave again immediately rather
    /// than lingering a step underground.
    pub fn step<R: Rng + RngExt + ?Sized>(
        &mut self,
        dt: f64,
        config: &FragmentationConfig,
        rng: &mut R,
    ) -> StepReport {
        let detection = self.step_and_detect(dt);

        let (breakups, fragments_created) = if detection.collisions.is_empty() {
            (Vec::new(), 0)
        } else {
            self.apply_breakups(&detection, dt, config, rng)
        };

        let reentries = self.remove_reentered_objects();

        StepReport {
            detection,
            breakups,
            fragments_created,
            reentries,
        }
    }

    /// Replace every object caught in this step's collisions with its debris,
    /// returning the breakups applied and the total fragments created.
    fn apply_breakups<R: Rng + RngExt + ?Sized>(
        &mut self,
        detection: &DetectionReport,
        dt: f64,
        config: &FragmentationConfig,
        rng: &mut R,
    ) -> (Vec<Breakup>, usize) {
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

        (breakups, fragments_created)
    }

    /// Remove every object that has descended below the reentry altitude,
    /// appending each to the run log and returning this sweep's reentries.
    ///
    /// Draws no randomness, so it cannot perturb the rng sequence a run
    /// replays from.
    fn remove_reentered_objects(&mut self) -> Vec<Reentry> {
        let floor_radius = EARTH_RADIUS_KM + REENTRY_ALTITUDE_KM;
        let elapsed_seconds = self.elapsed_seconds;
        let mut reentries = Vec::new();

        // `retain` visits in order, so the log stays in catalog order.
        self.objects.retain(|object| {
            let has_reentered = object.pos.norm() < floor_radius;
            if has_reentered {
                reentries.push(Reentry {
                    object: object.clone(),
                    elapsed_seconds,
                });
            }
            !has_reentered
        });

        self.reentries.extend(reentries.iter().cloned());
        reentries
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

    /// The radius below which an object counts as having re-entered.
    fn reentry_floor_radius() -> f64 {
        EARTH_RADIUS_KM + REENTRY_ALTITUDE_KM
    }

    /// An object parked at `radius` on a slow, nearly-radial state, so a short
    /// step cannot move it across the reentry floor on its own. This isolates
    /// the sweep from the propagation.
    fn object_at_radius(id: u64, radius: f64) -> Object {
        Object::new(
            id,
            Vector3::new(radius, 0.0, 0.0),
            Vector3::new(0.0, 0.001, 0.0),
            0.003,
            500.0,
            ObjectKind::Intact,
        )
    }

    fn step_once(state: &mut SimulationState, seed: u64) -> StepReport {
        state.step(1.0, &FragmentationConfig::default(), &mut StdRng::seed_from_u64(seed))
    }

    #[test]
    fn an_object_below_the_reentry_altitude_is_removed_and_recorded() {
        let mut state =
            SimulationState::from_objects([object_at_radius(1, reentry_floor_radius() - 10.0)]);
        let report = step_once(&mut state, 1);

        assert!(state.is_empty(), "the object was not removed");
        assert_eq!(report.reentries.len(), 1);
        assert_eq!(report.reentries[0].object.id, 1);
        assert_eq!(state.reentries().len(), 1, "the run log did not record it");
        assert_eq!(state.reentries()[0], report.reentries[0]);
    }

    #[test]
    fn an_object_above_the_reentry_altitude_survives() {
        let mut state =
            SimulationState::from_objects([object_at_radius(1, reentry_floor_radius() + 10.0)]);
        let report = step_once(&mut state, 1);

        assert_eq!(state.len(), 1, "an object still in orbit was removed");
        assert!(report.reentries.is_empty());
        assert!(state.reentries().is_empty());
    }

    #[test]
    fn reentry_is_swept_on_a_step_with_no_collisions() {
        // The trap this guards: `step` used to return early when nothing
        // collided. A lone object has no possible collision partner, so if the
        // sweep sits behind that early return it never runs at all.
        let mut state =
            SimulationState::from_objects([object_at_radius(1, reentry_floor_radius() - 10.0)]);
        let report = step_once(&mut state, 1);

        assert!(
            report.detection.collisions.is_empty(),
            "the fixture was supposed to have nothing to collide with"
        );
        assert_eq!(report.reentries.len(), 1, "the sweep was skipped");
        assert!(state.is_empty());
    }

    #[test]
    fn mass_leaving_the_catalog_is_preserved_in_the_reentry_log() {
        // Once objects can leave, the catalog's mass alone is no longer
        // conserved. What is conserved is the mass still in orbit plus the mass
        // recorded as having come down.
        let mut state = SimulationState::from_objects([
            object_at_radius(1, reentry_floor_radius() - 10.0),
            object_at_radius(2, reentry_floor_radius() + 10.0),
        ]);
        let mass_before = total_mass(&state);
        step_once(&mut state, 1);

        let mass_reentered: f64 = state.reentries().iter().map(|r| r.object.mass).sum();
        assert_relative_eq!(total_mass(&state) + mass_reentered, mass_before, epsilon = 1e-9);
    }

    #[test]
    fn the_clock_advances_and_stamps_each_reentry() {
        let dt = 30.0;
        let mut state = SimulationState::from_objects([
            object_at_radius(1, reentry_floor_radius() + 10.0),
            object_at_radius(2, reentry_floor_radius() - 10.0),
        ]);
        assert_eq!(state.elapsed_seconds(), 0.0);

        let config = FragmentationConfig::default();
        let mut rng = StdRng::seed_from_u64(1);

        let report = state.step(dt, &config, &mut rng);
        assert_relative_eq!(state.elapsed_seconds(), dt);
        assert_relative_eq!(report.reentries[0].elapsed_seconds, dt);

        state.step(dt, &config, &mut rng);
        assert_relative_eq!(state.elapsed_seconds(), 2.0 * dt);

        // The stamp records when the object came down, not when it is read.
        assert_relative_eq!(state.reentries()[0].elapsed_seconds, dt);
    }

    #[test]
    fn fragments_born_below_the_reentry_altitude_leave_the_same_step() {
        // Fragments are created after detection, so the sweep has to run after
        // fragmentation too — otherwise debris from a low-altitude breakup
        // would spend a step propagating underground before being noticed.
        let low = reentry_floor_radius() - 10.0;
        let mut state = SimulationState::from_objects([
            colliding_object(1, Vector3::new(low - 7000.0, 0.0, 0.0), Vector3::new(0.0, 7.5, 0.0), 800.0),
            colliding_object(
                2,
                Vector3::new(low - 7000.0, 0.002, 0.0),
                Vector3::new(0.0, -7.0, 1.0),
                400.0,
            ),
        ]);
        let mass_before = total_mass(&state);
        let report = state.step(
            1.0,
            &FragmentationConfig::default(),
            &mut StdRng::seed_from_u64(6),
        );

        assert_eq!(report.breakups.len(), 1, "the fixture did not collide");
        assert!(report.fragments_created > 0);
        assert_eq!(
            report.reentries.len(),
            report.fragments_created,
            "fragments born below the floor were left in the catalog"
        );
        assert!(state.is_empty());

        let mass_reentered: f64 = state.reentries().iter().map(|r| r.object.mass).sum();
        assert_relative_eq!(mass_reentered, mass_before, epsilon = 1e-9);
    }
}
