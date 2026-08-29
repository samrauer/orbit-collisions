//! Shared physical constants for the simulation.

/// Earth's standard gravitational parameter, GM (km^3/s^2).
pub const MU_EARTH: f64 = 398_600.441_8;

/// Mean equatorial radius of the Earth (km).
pub const EARTH_RADIUS_KM: f64 = 6378.137;

/// Altitude (km) below which an object is considered to have re-entered and is
/// removed from the catalog. The Karman line, the conventional edge of space.
pub const REENTRY_ALTITUDE_KM: f64 = 100.0;
