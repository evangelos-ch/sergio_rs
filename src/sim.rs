use ndarray::{s, Array, Array1, Array2, ArrayView2, Axis, NewAxis};
use ndarray_rand::{
    rand::{Rng, SeedableRng},
    rand_distr::{Distribution, Normal, Uniform},
    RandomExt,
};
use polars::{frame::DataFrame, prelude::NamedFrom, series::Series};
use pyo3::{pyclass, pymethods};
use pyo3_polars::PyDataFrame;
use rand_pcg::Lcg128Xsl64;

use crate::{gene::ConcType, grn::GRN, mrs::MrProfile};

#[pyclass]
pub struct Sim {
    grn: GRN,
    num_cells: usize,
    safety_iter: usize,
    scale_iter: usize,
    dt: f64,
    noise_s: f64,
    seed: u64,
}

#[pymethods]
impl Sim {
    #[new]
    pub fn new(
        grn: GRN,
        num_cells: usize,
        safety_iter: usize,
        scale_iter: usize,
        dt: f64,
        noise_s: f64,
        seed: u64,
    ) -> Self {
        Self {
            grn,
            num_cells,
            safety_iter,
            scale_iter,
            dt,
            noise_s,
            seed,
        }
    }

    pub fn simulate(&mut self, mr_profile: &MrProfile) -> PyDataFrame {
        let mut rng = Lcg128Xsl64::seed_from_u64(self.seed);
        let max_iter = self.num_cells * self.scale_iter + self.safety_iter;
        self.grn.init(mr_profile, max_iter);
        let lambda = Array1::from_iter(self.grn.genes.iter().map(|x| x.read().unwrap().decay));
        let lambda_col = lambda.slice(s![.., NewAxis]);
        let rand_dist = Normal::new(0.0, 1.0).unwrap();
        let computation_shape = (self.grn.genes.len(), self.grn.num_cell_types);

        for _ in 0..max_iter {
            self.iter_ss(&lambda_col, &rand_dist, &computation_shape, &mut rng);
        }

        PyDataFrame(self.get_expr_df(mr_profile, &mut rng))
    }
}

impl Sim {
    fn iter_ss<T: Distribution<f64>, R: Rng>(
        &self,
        lambda: &ArrayView2<f64>,
        rand_dist: &T,
        computation_shape: &(usize, usize),
        rng: &mut R,
    ) {
        let x_vec: Vec<f64> = self
            .grn
            .genes
            .iter()
            .map(|x| x.read().unwrap().get_last_conc().to_owned())
            .flatten()
            .collect();
        let x: Array2<f64> = Array::from_shape_vec(*computation_shape, x_vec).unwrap();
        let p_vec: Vec<f64> = self
            .grn
            .genes
            .iter()
            .map(|x| x.read().unwrap().calc_prod(&ConcType::SIM))
            .flatten()
            .collect();
        let p: Array2<f64> =
            Array::from_shape_vec((self.grn.genes.len(), self.grn.num_cell_types), p_vec)
                .unwrap()
                .map(|x| x.max(0.0));
        let sqrt_p = p.map(|x| x.sqrt());

        let d = lambda * &x;
        let sqrt_d = d.map(|x| x.sqrt());
        let sqrt_dt = self.dt.sqrt();
        let rnd_p = Array::random_using(*computation_shape, rand_dist, rng);
        let rnd_d = Array::random_using(*computation_shape, rand_dist, rng);
        // Eq 3 in the paper
        let new_x = &x
            + (&p - &d).mapv_into(|x| x * self.dt)
            + (&sqrt_p * &rnd_p + &sqrt_d * &rnd_d).mapv_into(|x| x * self.noise_s * sqrt_dt);
        for (gene, row) in self.grn.genes.iter().zip(new_x.axis_iter(Axis(0))) {
            gene.write().unwrap().append_sim_conc(&row);
        }
    }

    fn get_expr_df<T: Rng>(&self, mr_profile: &MrProfile, rng: &mut T) -> DataFrame {
        // Sample random timesteps after safety_iter
        let rnd_dist = Uniform::new(0, self.num_cells * self.scale_iter);
        let rnd_inds: Array2<usize> =
            Array::random_using((mr_profile.num_cell_types, self.num_cells), rnd_dist, rng)
                .map(|x| x + self.safety_iter);
        // DF Data
        let mut gene_names: Vec<String> = self
            .grn
            .genes
            .iter()
            .map(|x| x.read().unwrap().name.clone())
            .collect();
        gene_names.sort();
        let cell_names: Vec<String> = (0..mr_profile.num_cell_types)
            .into_iter()
            .map(|i| {
                (0..self.num_cells)
                    .into_iter()
                    .map(move |j| format!("type_{i}_cell_{j}"))
            })
            .flatten()
            .collect();
        let gene_exprs: Vec<f64> = gene_names
            .iter()
            .map(|x| {
                self.grn
                    .genes
                    .iter()
                    .find(|y| y.read().unwrap().name == *x)
                    .unwrap()
            })
            .map(|x| {
                let gene_binding = x.read().unwrap();
                let sim_conc = gene_binding.sim_conc.as_ref().unwrap();
                let expr: Array1<f64> = rnd_inds
                    .axis_iter(Axis(0))
                    .enumerate()
                    .map(|(ct, inds)| inds.map(|ind| sim_conc[(ct, *ind)]))
                    .flatten()
                    .collect();
                expr
            })
            .flatten()
            .collect();
        let gene_exprs = Array::from_shape_vec(
            (
                self.grn.genes.len(),
                mr_profile.num_cell_types * self.num_cells,
            ),
            gene_exprs,
        )
        .unwrap();
        let mut df_cols: Vec<Series> = vec![Series::new("Genes", gene_names)];
        df_cols.extend(
            gene_exprs
                .axis_iter(Axis(1))
                .zip(cell_names)
                .map(|(col, cell_name)| Series::new(&cell_name, col.to_vec())),
        );
        DataFrame::new(df_cols).expect("dataframe generation shouldn't error")
    }
}

#[cfg(test)]
mod tests {
    use crate::gene::Gene;

    use super::*;

    #[test]
    fn test_sim() {
        // Mostly a smoke test
        let mut grn = GRN::new();
        let g1 = Gene::new(String::from("gene1"), 0.8);
        let g2 = Gene::new(String::from("gene2"), 0.8);
        let g3 = Gene::new(String::from("gene3"), 0.8);
        let g4 = Gene::new(String::from("gene4"), 0.8);
        let g5 = Gene::new(String::from("gene5"), 0.8);
        let g6 = Gene::new(String::from("gene6"), 0.8);
        let g7 = Gene::new(String::from("gene7"), 0.8);

        grn.add_interaction(&g1, &g2, 3.0, None, 2);
        grn.add_interaction(&g4, &g2, 3.0, None, 2);
        grn.add_interaction(&g7, &g5, 3.0, None, 2);
        grn.add_interaction(&g2, &g3, 3.0, None, 2);
        grn.add_interaction(&g5, &g6, 3.0, None, 2);
        grn.add_interaction(&g3, &g5, 3.0, None, 2);

        grn.set_mrs();

        let num_cell_types = 10;
        let num_cells = 200;
        let mr_profile = MrProfile::from_random(&grn, num_cell_types, 1.0..2.5, 3.5..5.0, 42);
        let mut sim = Sim::new(grn, num_cells, 150, 10, 0.01, 1.0, 42);
        let df = sim.simulate(&mr_profile);
        assert!(df.0.get_columns().len() == num_cell_types * num_cells + 1);
        assert!(df.0.get_column_names()[0] == "Genes");
    }
}
