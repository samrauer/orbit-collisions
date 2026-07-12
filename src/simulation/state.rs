use crate::simulation::object::Object;

/// Holds the state of every active debris object / satellite in the
/// simulation.
#[derive(Clone, Debug, Default)]
pub struct SimulationState {
    objects: Vec<Object>,
}

impl SimulationState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_objects(objects: impl IntoIterator<Item = Object>) -> Self {
        Self {
            objects: objects.into_iter().collect(),
        }
    }

    pub fn push(&mut self, object: Object) {
        self.objects.push(object);
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulation::object::ObjectKind;
    use nalgebra::Vector3;

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
    fn propagate_moves_objects() {
        let mut state = SimulationState::from_objects([sample_object(1)]);
        let before = state.iter().next().unwrap().pos;
        state.propagate(60.0);
        let after = state.iter().next().unwrap().pos;
        assert!((after - before).norm() > 0.0);
    }
}
