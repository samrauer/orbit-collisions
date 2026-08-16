//! End-to-end check that the cascade loop actually runs.
//!
//! The unit tests cover one step at a time; this drives many steps in sequence
//! and asserts the properties that have to hold across a whole run — mass is
//! conserved, ids stay unique, and a given seed always produces the same run.

use nalgebra::Vector3;
use orbit_collisions::simulation::{
    FragmentationConfig, Object, ObjectKind, SimulationState, SpawnConfig, Spawner,
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

/// Run the cascade and return the final catalog, sorted by id.
fn run_cascade(seed: u64) -> (Vec<Object>, usize, f64) {
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
    (objects, breakups, mass_before)
}

#[test]
fn a_cascade_breaks_objects_up_and_conserves_mass() {
    let (objects, breakups, mass_before) = run_cascade(17);

    assert!(breakups > 0, "no collision occurred over {STEPS} steps");
    assert!(
        objects.len() > 42,
        "catalog did not grow: {} objects",
        objects.len()
    );

    let mass_after: f64 = objects.iter().map(|o| o.mass).sum();
    let drift = (mass_after - mass_before).abs() / mass_before;
    assert!(
        drift < 1e-9,
        "mass drifted from {mass_before} to {mass_after}"
    );

    assert!(
        objects.iter().any(|o| o.kind == ObjectKind::Fragment),
        "no debris was produced"
    );
    assert!(
        objects.iter().all(|o| o.mass > 0.0 && o.radius > 0.0),
        "an object came out with no mass or size"
    );
}

#[test]
fn ids_stay_unique_across_a_cascade() {
    // Objects enter the catalog from two directions and leave it mid-run, so a
    // duplicated id here would mean a breakup reissued one already in use.
    let (objects, _, _) = run_cascade(23);

    let mut ids: Vec<u64> = objects.iter().map(|o| o.id).collect();
    let total = ids.len();
    ids.dedup();
    assert_eq!(ids.len(), total, "the catalog contains duplicate ids");
}

#[test]
fn the_same_seed_replays_the_same_cascade() {
    // Fragments are generated cluster by cluster, so any instability in how
    // collisions are grouped would show up here as a divergent run.
    let (first, first_breakups, _) = run_cascade(31);
    let (second, second_breakups, _) = run_cascade(31);

    assert_eq!(first_breakups, second_breakups);
    assert_eq!(first, second);
}
