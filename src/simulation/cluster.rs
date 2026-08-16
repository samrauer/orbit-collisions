//! Grouping a step's collisions into the breakups they represent.
//!
//! One step can report several collisions that share an object: a satellite hit
//! from two sides in the same step appears in two encounters. Resolving those
//! pairwise would destroy the shared object twice and duplicate its mass, so
//! instead every collision that shares an object with another is merged into a
//! single breakup covering the whole pileup.
//!
//! That is exactly connected components: the nodes are object ids, the edges are
//! the step's collisions, and each component is one breakup. Pileups are rare
//! and tiny, so a flood fill per component is more than enough — there is no
//! need for union-find here.
//!
//! # Determinism
//!
//! The order clusters come out in decides the order fragments are generated in,
//! and therefore the order of draws from the simulation's rng. A run must be
//! reproducible from its seed, so this module never derives ordering from a
//! `HashMap`: components are discovered by walking `collisions` in slice order,
//! and each cluster's ids are sorted. Maps are used only as lookups that are
//! indexed, never iterated.

use std::collections::{HashMap, HashSet};

use crate::simulation::collision::Encounter;

/// A set of objects that break up together as one event.
#[derive(Clone, Debug, PartialEq)]
pub struct Cluster {
    /// Every object destroyed by this breakup, ascending. Two ids for an
    /// ordinary collision, more for a pileup.
    pub parent_ids: Vec<u64>,
    /// The earliest encounter in the cluster, by step fraction. It anchors both
    /// where the breakup happened and when within the step, since the first
    /// contact is what starts it.
    pub seed: Encounter,
}

/// Group a step's collisions into one cluster per pileup.
///
/// Every id named by `collisions` appears in exactly one returned cluster.
pub fn group_collisions_into_clusters(collisions: &[Encounter]) -> Vec<Cluster> {
    // Which collisions touch a given object, as indices into `collisions`.
    let mut collisions_by_id: HashMap<u64, Vec<usize>> = HashMap::new();
    for (index, encounter) in collisions.iter().enumerate() {
        for id in [encounter.id_a, encounter.id_b] {
            collisions_by_id.entry(id).or_default().push(index);
        }
    }

    let mut clusters = Vec::new();
    let mut visited: HashSet<usize> = HashSet::new();

    // Walking in slice order (rather than over the map) is what makes the
    // output order reproducible.
    for start in 0..collisions.len() {
        if !visited.insert(start) {
            continue;
        }

        // Flood fill from this collision through every collision reachable by a
        // shared object, collecting the ids and the earliest encounter.
        let mut parent_ids: HashSet<u64> = HashSet::new();
        let mut seed = collisions[start];
        let mut queue = vec![start];

        while let Some(index) = queue.pop() {
            let encounter = collisions[index];
            if encounter.fraction < seed.fraction {
                seed = encounter;
            }
            for id in [encounter.id_a, encounter.id_b] {
                parent_ids.insert(id);
                for &neighbor in &collisions_by_id[&id] {
                    if visited.insert(neighbor) {
                        queue.push(neighbor);
                    }
                }
            }
        }

        let mut parent_ids: Vec<u64> = parent_ids.into_iter().collect();
        parent_ids.sort_unstable();
        clusters.push(Cluster { parent_ids, seed });
    }

    clusters
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Vector3;

    /// An encounter between `id_a` and `id_b` at step fraction `fraction`.
    fn encounter(id_a: u64, id_b: u64, fraction: f64) -> Encounter {
        Encounter {
            id_a,
            id_b,
            separation: 0.0,
            surface_gap: -1.0,
            fraction,
            // Distinct per fraction, so tests can tell which encounter seeded.
            impact_point: Vector3::new(fraction, 0.0, 0.0),
        }
    }

    #[test]
    fn no_collisions_yield_no_clusters() {
        assert!(group_collisions_into_clusters(&[]).is_empty());
    }

    #[test]
    fn disjoint_collisions_stay_separate() {
        let clusters =
            group_collisions_into_clusters(&[encounter(1, 2, 0.5), encounter(3, 4, 0.5)]);
        assert_eq!(clusters.len(), 2);
        assert_eq!(clusters[0].parent_ids, vec![1, 2]);
        assert_eq!(clusters[1].parent_ids, vec![3, 4]);
    }

    #[test]
    fn collisions_sharing_an_object_merge_into_one_cluster() {
        // Object 2 is hit by both 1 and 3 in the same step; all three break up
        // together rather than 2 being destroyed twice.
        let clusters =
            group_collisions_into_clusters(&[encounter(1, 2, 0.5), encounter(2, 3, 0.7)]);
        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0].parent_ids, vec![1, 2, 3]);
    }

    #[test]
    fn a_chain_of_shared_objects_merges_transitively() {
        // 1-2, 2-3, 3-4: no collision names both 1 and 4, but they are still one
        // connected pileup.
        let clusters = group_collisions_into_clusters(&[
            encounter(1, 2, 0.5),
            encounter(2, 3, 0.5),
            encounter(3, 4, 0.5),
        ]);
        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0].parent_ids, vec![1, 2, 3, 4]);
    }

    #[test]
    fn the_seed_is_the_earliest_encounter_not_the_first_listed() {
        // The later-listed encounter happens earlier in the step, so it is the
        // one that anchors the breakup.
        let clusters =
            group_collisions_into_clusters(&[encounter(1, 2, 0.9), encounter(2, 3, 0.2)]);
        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0].seed.fraction, 0.2);
        assert_eq!(clusters[0].seed.impact_point, Vector3::new(0.2, 0.0, 0.0));
    }

    #[test]
    fn every_collided_object_lands_in_exactly_one_cluster() {
        let collisions = [
            encounter(10, 11, 0.4),
            encounter(20, 21, 0.6),
            encounter(11, 12, 0.3),
        ];
        let clusters = group_collisions_into_clusters(&collisions);

        let mut seen: Vec<u64> = clusters
            .iter()
            .flat_map(|c| c.parent_ids.iter().copied())
            .collect();
        let total = seen.len();
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(seen.len(), total, "an object appeared in two clusters");
        assert_eq!(seen, vec![10, 11, 12, 20, 21]);
    }

    #[test]
    fn cluster_order_is_stable_across_runs() {
        // Fragment generation draws from the rng cluster by cluster, so an
        // unstable order here would make seeded runs irreproducible.
        let collisions = [
            encounter(7, 8, 0.4),
            encounter(1, 2, 0.6),
            encounter(8, 9, 0.1),
            encounter(3, 4, 0.2),
        ];
        let first = group_collisions_into_clusters(&collisions);
        for _ in 0..20 {
            assert_eq!(group_collisions_into_clusters(&collisions), first);
        }
    }
}
