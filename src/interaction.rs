use ndarray::Array1;
use pyo3::pyclass;

use crate::gene::{ConcType, GeneHandleWeak};

#[pyclass]
#[derive(Clone)]
pub struct Interaction {
    pub reg: GeneHandleWeak,
    pub k: f64,
    pub h: Option<f64>,
    pub n: i32,
}

impl Interaction {
    pub fn get_hill(&self, regs_conc: &ConcType) -> Array1<f64> {
        let reg_unwrapped = self.reg.upgrade().unwrap();
        let reg = reg_unwrapped.read().unwrap();
        let x = match regs_conc {
            ConcType::SIM => reg.get_last_conc(),
            ConcType::SS => reg.ss_conc.as_ref().unwrap().view(),
        };
        let pow_h = self.h.unwrap().powi(self.n);
        x.map(|x| {
            let pow_x = x.powi(self.n);
            let val = pow_x / (pow_x + pow_h);
            if self.k > 0.0 {
                self.k * val
            } else {
                self.k.abs() * (1.0 - val)
            }
        })
    }
}
