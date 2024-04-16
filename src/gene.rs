use ndarray::{s, Array, Array1, Array2, ArrayView1};
use pyo3::{pyclass, pymethods};
use std::sync::{Arc, RwLock, Weak};

use crate::interaction::Interaction;

pub type GeneHandle = Arc<RwLock<Gene>>;
pub type GeneHandleWeak = Weak<RwLock<Gene>>;

#[pyclass]
pub enum ConcType {
    SS,
    SIM,
}

#[pyclass]
#[derive(Clone)]
pub struct Gene {
    pub name: String,
    pub is_mr: bool,
    pub tars: Vec<GeneHandle>,
    pub in_interactions: Vec<Interaction>,
    pub prod_rates: Option<Array1<f64>>,
    pub ss_conc: Option<Array1<f64>>,
    pub sim_conc: Option<Array2<f64>>,
    pub current_iter: usize,
    pub current_iters: Option<Array1<usize>>,
    pub decay: f64,
    pub num_cell_types: usize,
}

#[pymethods]
impl Gene {
    #[new]
    pub fn new(name: String, decay: f64) -> Self {
        Self {
            name,
            is_mr: false,
            tars: vec![],
            in_interactions: vec![],
            prod_rates: None,
            ss_conc: None,
            sim_conc: None,
            current_iter: 0,
            current_iters: None,
            decay,
            num_cell_types: 0,
        }
    }
}

impl Gene {
    pub fn calc_prod(&self, regs_conc: &ConcType) -> Array1<f64> {
        if self.is_mr {
            return self.prod_rates.as_ref().unwrap().clone();
        }

        let mut x = Array::zeros((self.num_cell_types,));
        for i in 0..self.in_interactions.len() {
            x = x + self.in_interactions[i].get_hill(regs_conc);
        }
        return x;
    }

    pub fn get_last_conc(&self) -> ArrayView1<f64> {
        self.sim_conc
            .as_ref()
            .unwrap()
            .slice(s![.., self.current_iter - 1])
    }

    pub fn append_sim_conc(&mut self, conc: &ArrayView1<f64>) {
        self.sim_conc
            .as_mut()
            .unwrap()
            .slice_mut(s![.., self.current_iter])
            .zip_mut_with(conc, |x, y| *x = y.max(0.0));
        self.current_iter += 1;
    }
}
