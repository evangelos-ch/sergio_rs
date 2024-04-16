use std::collections::HashSet;
use std::sync::{Arc, RwLock};

use ndarray::{concatenate, s, Array, Axis, NewAxis};
use pyo3::{pyclass, pymethods};

use crate::{
    gene::{ConcType, Gene, GeneHandle},
    interaction::Interaction,
    mrs::MrProfile,
};

#[pyclass]
#[derive(Clone)]
pub struct GRN {
    pub genes: Vec<GeneHandle>,
    pub mrs: Vec<GeneHandle>,
    pub level_to_gene: Vec<Vec<GeneHandle>>,
    pub num_cell_types: usize,
}

#[pymethods]
impl GRN {
    #[new]
    pub fn new() -> Self {
        Self {
            genes: vec![],
            mrs: vec![],
            level_to_gene: vec![],
            num_cell_types: 0,
        }
    }

    #[pyo3(signature = (reg, tar, k, h=None, n=2))]
    pub fn add_interaction(&mut self, reg: &Gene, tar: &Gene, k: f64, h: Option<f64>, n: i32) {
        let reg_in_net = self.add_gene(reg);
        let tar_in_net = self.add_gene(tar);
        tar_in_net
            .write()
            .unwrap()
            .in_interactions
            .push(Interaction {
                reg: Arc::downgrade(&reg_in_net),
                k,
                h,
                n,
            });
        reg_in_net.write().unwrap().tars.push(tar_in_net);
    }

    pub fn set_mrs(&mut self) {
        self.mrs.clear();
        for gene in self.genes.iter() {
            let mut gene_binding = gene.write().unwrap();
            if gene_binding.in_interactions.len() == 0 {
                gene_binding.is_mr = true;
                self.mrs.push(Arc::clone(&gene));
            }
        }
    }

    pub fn ko_perturbation(&self, gene_name: String, mr_profile: &MrProfile) -> (Self, MrProfile) {
        // Find gene in the GRN
        let gene_idx = self
            .genes
            .iter()
            .position(|x| x.read().unwrap().name == gene_name)
            .expect(&format!("Gene {gene_name} is not in the GRN."));
        // Clone self
        let mut clone = self.clone();
        // Remove the gene
        clone.genes.remove(gene_idx);
        let mrs_idx = clone
            .mrs
            .iter()
            .position(|x| x.read().unwrap().name == gene_name);
        // Remove it from MRs
        if mrs_idx.is_some() {
            clone.mrs.remove(mrs_idx.unwrap());
        };
        // Remove refs to it from other genes
        clone.genes.iter().for_each(|x| {
            let tar_idx = x
                .read()
                .unwrap()
                .tars
                .iter()
                .position(|y| y.read().unwrap().name == gene_name);
            if tar_idx.is_some() {
                x.write().unwrap().tars.remove(tar_idx.unwrap());
            };
            let inter_idx = x
                .read()
                .unwrap()
                .in_interactions
                .iter()
                .position(|y| y.reg.upgrade().unwrap().read().unwrap().name == gene_name);
            if inter_idx.is_some() {
                x.write()
                    .unwrap()
                    .in_interactions
                    .remove(inter_idx.unwrap());
            }
        });
        // Adjust MR Profile
        let mut perturbed_mr_profile = mr_profile.clone();
        perturbed_mr_profile.mr_prod_rates.remove(&gene_name);
        return (clone, perturbed_mr_profile);
    }
}

impl GRN {
    pub fn init(&mut self, mr_profile: &MrProfile, max_iter: usize) {
        self.set_levels();
        self.set_mr_profile(&mr_profile);
        self.init_genes(max_iter);
    }

    fn add_gene(&mut self, gene: &Gene) -> GeneHandle {
        if let Some(existing_gene) = self
            .genes
            .iter()
            .find(|x| x.read().unwrap().name == gene.name)
        {
            return existing_gene.clone();
        }

        self.genes.push(Arc::new(RwLock::new(gene.clone())));
        self.genes.last().unwrap().clone()
    }

    fn set_levels(&mut self) {
        let v: HashSet<String> = self
            .genes
            .iter()
            .map(|x| x.read().unwrap().name.clone())
            .collect();
        let mut u: HashSet<String> = HashSet::new();

        while v != u {
            let current_u = u.clone();
            let current_verts = v.difference(&current_u).filter(|x| {
                let gene = self
                    .genes
                    .iter()
                    .find(|y| y.read().unwrap().name == **x)
                    .unwrap();
                let tars_set: HashSet<String> = gene
                    .read()
                    .unwrap()
                    .tars
                    .iter()
                    .map(|y| y.read().unwrap().name.clone())
                    .collect();
                tars_set.is_subset(&current_u)
            });

            let mut current_level_genes: Vec<GeneHandle> = vec![];

            for vert in current_verts {
                let gene = self
                    .genes
                    .iter()
                    .find(|x| x.read().unwrap().name == *vert)
                    .unwrap();
                current_level_genes.push(Arc::clone(&gene));
                u.insert(vert.clone());
            }

            self.level_to_gene.push(current_level_genes);
        }
    }

    fn set_mr_profile(&mut self, mr_profile: &MrProfile) {
        self.num_cell_types = mr_profile.num_cell_types;
        for (mr, prod_rates) in mr_profile.mr_prod_rates.iter() {
            let mr_in_net = self
                .mrs
                .iter()
                .find(|x| x.read().unwrap().name == *mr)
                .expect("mr_data should contain prod_rates for MRs and MRs. must be set");
            assert!(
                prod_rates.len() == mr_profile.num_cell_types,
                "prod_rates data must be of length num_cell_types."
            );

            mr_in_net.write().unwrap().prod_rates = Some(prod_rates.clone());
        }
        for gene in self.genes.iter() {
            let mut gene_binding = gene.write().unwrap();
            gene_binding.num_cell_types = mr_profile.num_cell_types;
        }
    }

    fn estimate_steady_state(&self, level: &[GeneHandle]) {
        for gene in level {
            let mut gene_binding = gene.write().unwrap();
            let prod = gene_binding
                .calc_prod(&ConcType::SS)
                .map(|x| x / gene_binding.decay);
            gene_binding.ss_conc = Some(prod);
        }
    }

    fn estimate_half_response(&self, level: &[GeneHandle]) {
        for gene in level {
            for inter in gene.write().unwrap().in_interactions.iter_mut() {
                let reg_ss_conc = inter.reg.upgrade().unwrap();
                let ss_conc_mean = reg_ss_conc.read().unwrap().ss_conc.as_ref().unwrap().mean();
                inter.h = Some(ss_conc_mean.unwrap());
            }
        }
    }

    fn init_genes(&mut self, max_iter: usize) {
        for (i, level) in self.level_to_gene.iter().enumerate().rev() {
            if i != (self.level_to_gene.len() - 1) {
                self.estimate_half_response(level);
            }
            self.estimate_steady_state(level);
        }
        for gene in self.genes.iter() {
            let mut gene_binding = gene.write().unwrap();
            let ss_conc_to_col = gene_binding
                .ss_conc
                .as_ref()
                .unwrap()
                .slice(s![.., NewAxis]);
            let zeros = Array::zeros((self.num_cell_types, max_iter));
            gene_binding.sim_conc = Some(concatenate![Axis(1), ss_conc_to_col, zeros]);
            gene_binding.current_iters = Some(Array::ones((self.num_cell_types,)));
            gene_binding.current_iter = 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grn_create() {
        let mut grn = GRN::new();
        let g1 = Gene::new(String::from("gene1"), 0.8);
        let g2 = Gene::new(String::from("gene2"), 0.8);
        grn.add_interaction(&g1, &g2, 3.0, None, 2);
        assert!(grn.genes.len() == 2);
        assert!(
            grn.genes
                .last()
                .unwrap()
                .read()
                .unwrap()
                .in_interactions
                .len()
                == 1
        );
        assert!(grn.genes.first().unwrap().read().unwrap().tars.len() == 1);
        assert!(grn.mrs.len() == 0); // Just adding genes doesn't add MRs
    }

    #[test]
    fn test_grn_cycles() {
        let mut grn = GRN::new();
        let g1 = Gene::new(String::from("gene1"), 0.8);
        let g2 = Gene::new(String::from("gene2"), 0.8);
        grn.add_interaction(&g1, &g2, 3.0, None, 2);
        grn.add_interaction(&g2, &g1, -3.0, None, 2);
        assert!(grn.genes.len() == 2);
        let first_gene = grn.genes.first().unwrap().read().unwrap();
        let last_gene = grn.genes.last().unwrap().read().unwrap();
        assert!(last_gene.in_interactions.len() == 1);
        assert!(first_gene.in_interactions.len() == 1);
        assert!(first_gene.tars.len() == 1);
        assert!(last_gene.tars.len() == 1);
    }

    #[test]
    fn test_set_mrs() {
        let mut grn = GRN::new();
        let g1 = Gene::new(String::from("gene1"), 0.8);
        let g2 = Gene::new(String::from("gene2"), 0.8);
        let g3 = Gene::new(String::from("gene3"), 0.8);
        let g4 = Gene::new(String::from("gene4"), 0.8);
        grn.add_interaction(&g1, &g2, 3.0, None, 2);
        grn.add_interaction(&g2, &g3, 3.0, None, 2);
        grn.add_interaction(&g4, &g2, 3.0, None, 2);
        assert!(grn.genes.len() == 4);
        grn.set_mrs();
        assert!(grn.mrs.len() == 2);
        assert!(grn.genes[0].read().unwrap().is_mr);
        assert!(!grn.genes[1].read().unwrap().is_mr);
        assert!(!grn.genes[2].read().unwrap().is_mr);
        assert!(grn.genes[3].read().unwrap().is_mr);
    }

    #[test]
    fn test_set_levels() {
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

        // 1  4  - Level 4
        // \ /
        //  2    - Level 3
        //  |
        //  3  7 - Level 2
        //  | /
        //  5    - Level 1
        //  |
        //  6    - Level 0

        grn.set_levels();

        assert!(grn.level_to_gene.len() == 5);
        assert!(grn.level_to_gene[0].len() == 1);
        assert!(grn.level_to_gene[0][0].read().unwrap().name == "gene6");

        assert!(grn.level_to_gene[1].len() == 1);
        assert!(grn.level_to_gene[1][0].read().unwrap().name == "gene5");

        assert!(grn.level_to_gene[2].len() == 2);
        assert!(
            HashSet::from_iter(
                grn.level_to_gene[2]
                    .iter()
                    .map(|x| x.read().unwrap().name.clone())
            ) == HashSet::from([String::from("gene3"), String::from("gene7")])
        );

        assert!(grn.level_to_gene[3].len() == 1);
        assert!(grn.level_to_gene[3][0].read().unwrap().name == "gene2");

        assert!(grn.level_to_gene[4].len() == 2);
        assert!(
            HashSet::from_iter(
                grn.level_to_gene[4]
                    .iter()
                    .map(|x| x.read().unwrap().name.clone())
            ) == HashSet::from([String::from("gene1"), String::from("gene4")])
        );
    }

    #[test]
    fn test_init() {
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
        let mr_profile = MrProfile::from_random(&grn, num_cell_types, 1.0..2.5, 3.5..5.0, 42);
        let max_iter = 2000;
        grn.init(&mr_profile, max_iter);
        for gene in grn.genes.iter() {
            let gene_binding = gene.read().unwrap();
            for inter in gene_binding.in_interactions.iter() {
                assert!(inter.h.is_some());
            }
            assert!(gene_binding.ss_conc.is_some());
            assert!(gene_binding.ss_conc.as_ref().unwrap().len() == num_cell_types);
            assert!(gene_binding.sim_conc.as_ref().unwrap().ndim() == 2);
            assert!(
                gene_binding.sim_conc.as_ref().unwrap().dim() == (num_cell_types, max_iter + 1)
            );
        }
    }
}
