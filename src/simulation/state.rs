use satkit::TLE;

/// Holds the state of every active debris object / satellite in the
/// simulation.
#[derive(Clone, Debug, Default)]
pub struct SimulationState {
    objects: Vec<TLE>,
}

impl SimulationState {
    pub fn new() -> Self {
        Self::default()
    }

    /// Build a state from a set of TLE lines (2- or 3-line, mixed is fine).
    pub fn from_lines(lines: &[String]) -> satkit::tle::Result<Self> {
        let tles = TLE::from_lines(lines)?;
        Ok(Self::from_tles(tles))
    }

    pub fn from_tles(tles: impl IntoIterator<Item = TLE>) -> Self {
        Self {
            objects: tles.into_iter().collect(),
        }
    }

    pub fn push(&mut self, tle: TLE) {
        self.objects.push(tle);
    }

    pub fn len(&self) -> usize {
        self.objects.len()
    }

    pub fn is_empty(&self) -> bool {
        self.objects.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &TLE> {
        self.objects.iter()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut TLE> {
        self.objects.iter_mut()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_lines() -> Vec<String> {
        vec![
            "0 INTELSAT 902".to_string(),
            "1 26900U 01039A   06106.74503247  .00000045  00000-0  10000-3 0  8290".to_string(),
            "2 26900   0.0164 266.5378 0003319  86.1794 182.2590  1.00273847 16981".to_string(),
        ]
    }

    #[test]
    fn new_state_is_empty() {
        let state = SimulationState::new();
        assert!(state.is_empty());
        assert_eq!(state.len(), 0);
    }

    #[test]
    fn from_lines_parses_tle() {
        let state = SimulationState::from_lines(&sample_lines()).unwrap();
        assert_eq!(state.len(), 1);
        assert_eq!(state.iter().next().unwrap().sat_num, 26900);
    }

    #[test]
    fn push_adds_object() {
        let mut state = SimulationState::new();
        let tle = TLE::from_lines(&sample_lines()).unwrap().remove(0);
        state.push(tle);
        assert_eq!(state.len(), 1);
        assert!(!state.is_empty());
    }

    #[test]
    fn iter_mut_allows_modification() {
        let mut state = SimulationState::from_lines(&sample_lines()).unwrap();
        for tle in state.iter_mut() {
            tle.sat_num = 1;
        }
        assert_eq!(state.iter().next().unwrap().sat_num, 1);
    }
}
