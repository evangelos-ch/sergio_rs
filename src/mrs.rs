use std::collections::HashMap;
use std::ops::Range;

use ndarray::{Array, Array1};
use ndarray_rand::rand::SeedableRng;
use ndarray_rand::rand_distr::{Bernoulli, Distribution, Uniform};
use pyo3::exceptions::PyValueError;
use pyo3::types::PyType;
use pyo3::{pyclass, pymethods, Bound, PyResult};
use rand_pcg::Lcg128Xsl64;

use crate::grn::GRN;

#[pyclass]
#[derive(Clone)]
pub struct MrProfile {
    pub num_cell_types: usize,
    pub mr_prod_rates: HashMap<String, Array1<f64>>,
}

#[pymethods]
impl MrProfile {
    #[classmethod]
    #[pyo3(name = "from_random")]
    pub fn py_from_random(
        _cls: &Bound<'_, PyType>,
        grn: &GRN,
        num_cell_types: usize,
        low_range: (f64, f64),
        high_range: (f64, f64),
        seed: u64,
    ) -> PyResult<Self> {
        if low_range.0 > low_range.1 {
            return Err(PyValueError::new_err("low_range is invalid."));
        } else if high_range.0 > high_range.1 {
            return Err(PyValueError::new_err("high_range is invalid."));
        } else if high_range.0 < low_range.1 {
            return Err(PyValueError::new_err(
                "high_range and low_range shouldn't have overlap.",
            ));
        }

        Ok(MrProfile::from_random(
            &grn,
            num_cell_types,
            low_range.0..low_range.1,
            high_range.0..high_range.1,
            seed,
        ))
    }
}

impl MrProfile {
    pub fn from_random(
        grn: &GRN,
        num_cell_types: usize,
        low_range: Range<f64>,
        high_range: Range<f64>,
        seed: u64,
    ) -> Self {
        assert!(grn.mrs.len() > 0, "the GRN must have MRs.");
        let mut rng = Lcg128Xsl64::seed_from_u64(seed);

        let mut mr_prod_rates: HashMap<String, Array1<f64>> = HashMap::new();
        let low_high_dist = Bernoulli::new(0.5).unwrap();
        let low_dist = Uniform::new(low_range.start, low_range.end);
        let high_dist = Uniform::new(high_range.start, high_range.end);
        for mr in grn.mrs.iter() {
            let cts: Array1<f64> = Array::zeros((num_cell_types,)).map(|x: &f64| {
                if low_high_dist.sample(&mut rng) {
                    x + high_dist.sample(&mut rng)
                } else {
                    x + low_dist.sample(&mut rng)
                }
            });
            mr_prod_rates.insert(mr.read().unwrap().name.clone(), cts);
        }
        Self {
            num_cell_types,
            mr_prod_rates,
        }
    }
}
