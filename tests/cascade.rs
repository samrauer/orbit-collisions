//! End-to-end check that the cascade loop actually runs.
//!
//! The unit tests cover one step at a time; this drives many steps in sequence
//! and asserts the properties that have to hold across a whole run — mass is
//! accounted for, ids stay unique, debris that decays is removed, and a given
//! seed always produces the same run.

use nalgebra::Vector3;
use orbit_collisions::simulation::{
    EARTH_RADIUS_KM, FragmentationConfig, Object, ObjectKind, REENTRY_ALTITUDE_KM, Reentry,
    SimulationState, SpawnConfig, Spawner,
};
use rand::SeedableRng;
use rand::rngs::StdRng;

const STEP_SECONDS: f64 = 30.0;
const STEPS: usize = 20;

/// A catalog seeded with random objects plus one pair placed on top of each
/// other, so a first collision is guaranteed rather than left to chance —
/// meter-scale objects in a random LEO shell essentially never meet.
fn catalog_with_a_guaranteed_collision(seed: u64) -> SimulationState {
    let mut state = SimulationState::new();
    let spawner = Spawner::new(SpawnConfig::default());
    state.spawn(&spawner, 40, &mut StdRng::seed_from_u64(seed));

    let origin = Vector3::new(7000.0, 0.0, 0.0);
    let doomed = [
        (Vector3::zeros(), Vector3::new(0.0, 7.5, 0.0), 800.0),
        (
            Vector3::new(0.0, 0.002, 0.0),
            Vector3::new(0.0, -7.0, 1.0),
            400.0,
        ),
    ];
    for (offset, vel, mass) in doomed {
        let id = state.next_id();
        state.push(Object::new(
            id,
            origin + offset,
            vel,
            0.003,
            mass,
            ObjectKind::Intact,
        ));
    }
    state
}

/// What one cascade run produced.
struct CascadeOutcome {
    /// The surviving catalog at the end of the run, sorted by id.
    objects: Vec<Object>,
    /// Everything that came down over the run, in the order it did.
    reentries: Vec<Reentry>,
    /// How many breakups were applied across every step.
    breakups: usize,
    /// Total catalog mass before the run started.
    mass_before: f64,
}

/// Run the cascade to completion and collect what it produced.
fn run_cascade(seed: u64) -> CascadeOutcome {
    let mut state = catalog_with_a_guaranteed_collision(seed);
    let mass_before: f64 = state.iter().map(|o| o.mass).sum();

    let config = FragmentationConfig::default();
    let mut rng = StdRng::seed_from_u64(seed);
    let mut breakups = 0;
    for _ in 0..STEPS {
        breakups += state.step(STEP_SECONDS, &config, &mut rng).breakups.len();
    }

    let mut objects: Vec<Object> = state.iter().cloned().collect();
    objects.sort_by_key(|o| o.id);
    CascadeOutcome {
        objects,
        reentries: state.reentries().to_vec(),
        breakups,
        mass_before,
    }
}

/// The radius below which an object counts as having re-entered.
fn reentry_floor_radius() -> f64 {
    EARTH_RADIUS_KM + REENTRY_ALTITUDE_KM
}

#[test]
fn a_cascade_breaks_objects_up_and_accounts_for_all_its_mass() {
    // Mass now leaves the catalog two ways: it is redistributed into fragments,
    // or it re-enters and is removed. Surviving mass alone is therefore no
    // longer conserved — surviving plus re-entered is.
    let outcome = run_cascade(17);

    assert!(outcome.breakups > 0, "no collision occurred over {STEPS} steps");
    assert!(
        outcome.objects.len() > 42,
        "catalog did not grow: {} objects",
        outcome.objects.len()
    );

    let mass_in_orbit: f64 = outcome.objects.iter().map(|o| o.mass).sum();
    let mass_reentered: f64 = outcome.reentries.iter().map(|r| r.object.mass).sum();
    let mass_after = mass_in_orbit + mass_reentered;
    let drift = (mass_after - outcome.mass_before).abs() / outcome.mass_before;
    assert!(
        drift < 1e-9,
        "mass drifted from {} to {mass_after} ({mass_in_orbit} in orbit + {mass_reentered} re-entered)",
        outcome.mass_before
    );

    assert!(
        outcome.objects.iter().any(|o| o.kind == ObjectKind::Fragment),
        "no debris was produced"
    );
    assert!(
        outcome.objects.iter().all(|o| o.mass > 0.0 && o.radius > 0.0),
        "an object came out with no mass or size"
    );
}

#[test]
fn a_cascade_removes_the_debris_that_decays_into_the_atmosphere() {
    // The reason this feature exists: fragmentation throws debris onto orbits
    // that intersect the Earth, and without removal those objects propagate on
    // underground and keep colliding.
    let outcome = run_cascade(17);

    assert!(
        !outcome.reentries.is_empty(),
        "no debris came down over {STEPS} steps"
    );

    let floor = reentry_floor_radius();
    for object in &outcome.objects {
        assert!(
            object.pos.norm() >= floor,
            "object {} survived at radius {} km, below the {floor} km floor",
            object.id,
            object.pos.norm()
        );
    }

    // Every reentry is stamped with the step it happened on, so the log is a
    // time series rather than an unordered bag.
    for reentry in &outcome.reentries {
        assert!(
            reentry.elapsed_seconds > 0.0
                && reentry.elapsed_seconds <= STEPS as f64 * STEP_SECONDS,
            "reentry of object {} stamped at {} s, outside the run",
            reentry.object.id,
            reentry.elapsed_seconds
        );
    }
}

#[test]
fn ids_stay_unique_across_a_cascade() {
    // Objects enter the catalog from two directions and leave it mid-run, so a
    // duplicated id here would mean a breakup reissued one already in use.
    let outcome = run_cascade(23);

    let mut ids: Vec<u64> = outcome.objects.iter().map(|o| o.id).collect();
    let total = ids.len();
    ids.dedup();
    assert_eq!(ids.len(), total, "the catalog contains duplicate ids");
}

#[test]
fn the_same_seed_replays_the_same_cascade() {
    // Fragments are generated cluster by cluster, so any instability in how
    // collisions are grouped would show up here as a divergent run.
    let first = run_cascade(31);
    let second = run_cascade(31);

    assert_eq!(first.breakups, second.breakups);
    assert_eq!(first.objects, second.objects);
    // The reentry sweep draws no randomness, so the same run must bring the
    // same objects down at the same times.
    assert_eq!(first.reentries, second.reentries);
}
