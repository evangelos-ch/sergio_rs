pub mod gene;
pub mod grn;
pub mod interaction;
pub mod mrs;
pub mod noise;
pub mod sim;

use pyo3::{prelude::*, wrap_pyfunction};

#[pymodule]
fn sergio_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<gene::Gene>()?;
    m.add_class::<interaction::Interaction>()?;
    m.add_class::<grn::GRN>()?;
    m.add_class::<mrs::MrProfile>()?;
    m.add_class::<sim::Sim>()?;
    m.add_wrapped(wrap_pyfunction!(noise::py_add_technical_noise))?;
    m.add_class::<noise::NoiseSetting>()?;
    Ok(())
}
