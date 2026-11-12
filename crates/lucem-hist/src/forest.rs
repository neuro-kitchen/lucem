//! Random Trees: an ensemble of decision trees, each grown on a bootstrap sample with a random
//! subset of the features tried at every split (Gini impurity); a prediction is the mean of
//! the trees' class fractions. Deterministic for a seed, and saved as JSON with its classifier.

use serde::{Deserialize, Serialize};

use crate::error::{invalid, HistResult};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ForestSettings {
    pub trees: usize,
    /// 0: no limit.
    pub max_depth: usize,
    pub min_samples_leaf: usize,
    /// Features tried per split; `None`: the square root of their number.
    pub features_per_split: Option<usize>,
    pub seed: u64,
}

impl Default for ForestSettings {
    fn default() -> Self {
        Self { trees: 50, max_depth: 0, min_samples_leaf: 1, features_per_split: None, seed: 0x5eed }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
enum Node {
    Split { feature: usize, threshold: f64, left: usize, right: usize },
    Leaf { fractions: Vec<f64> },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct Tree {
    nodes: Vec<Node>,
}

impl Tree {
    fn predict(&self, row: &[f64]) -> &[f64] {
        let mut i = 0;
        loop {
            match &self.nodes[i] {
                Node::Leaf { fractions } => return fractions,
                Node::Split { feature, threshold, left, right } => i = if row[*feature] <= *threshold { *left } else { *right },
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RandomForest {
    pub n_classes: usize,
    pub n_features: usize,
    trees: Vec<Tree>,
}

/// SplitMix64: a small deterministic generator.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n.max(1) as u64) as usize
    }
}

fn gini(counts: &[f64], total: f64) -> f64 {
    if total <= 0.0 {
        return 0.0;
    }
    1.0 - counts.iter().map(|c| (c / total).powi(2)).sum::<f64>()
}

struct Grower<'a> {
    x: &'a [Vec<f64>],
    y: &'a [usize],
    n_classes: usize,
    s: &'a ForestSettings,
    per_split: usize,
    rng: Rng,
}

impl Grower<'_> {
    fn leaf(&self, idx: &[usize]) -> Node {
        let mut f = vec![0.0; self.n_classes];
        for &i in idx {
            f[self.y[i]] += 1.0;
        }
        let n = idx.len().max(1) as f64;
        Node::Leaf { fractions: f.iter().map(|c| c / n).collect() }
    }

    fn grow(&mut self, nodes: &mut Vec<Node>, idx: Vec<usize>, depth: usize) -> usize {
        let at = nodes.len();
        nodes.push(Node::Leaf { fractions: Vec::new() });
        let pure = idx.iter().all(|&i| self.y[i] == self.y[idx[0]]);
        if pure || idx.len() < 2 * self.s.min_samples_leaf || (self.s.max_depth > 0 && depth >= self.s.max_depth) {
            nodes[at] = self.leaf(&idx);
            return at;
        }
        let n_features = self.x[0].len();
        let mut best: Option<(f64, usize, f64)> = None;
        let mut total = vec![0.0; self.n_classes];
        for &i in &idx {
            total[self.y[i]] += 1.0;
        }
        let parent = gini(&total, idx.len() as f64);
        let mut sorted = idx.clone();
        for _ in 0..self.per_split {
            let f = self.rng.below(n_features);
            sorted.sort_by(|&a, &b| self.x[a][f].total_cmp(&self.x[b][f]));
            let mut left = vec![0.0; self.n_classes];
            let n = sorted.len();
            for k in 0..n - 1 {
                left[self.y[sorted[k]]] += 1.0;
                let (v, next) = (self.x[sorted[k]][f], self.x[sorted[k + 1]][f]);
                if v == next || k + 1 < self.s.min_samples_leaf || n - k - 1 < self.s.min_samples_leaf {
                    continue;
                }
                let right: Vec<f64> = total.iter().zip(&left).map(|(t, l)| t - l).collect();
                let (nl, nr) = ((k + 1) as f64, (n - k - 1) as f64);
                let score = (nl * gini(&left, nl) + nr * gini(&right, nr)) / n as f64;
                if score < parent - 1e-12 && best.is_none_or(|b| score < b.0) {
                    best = Some((score, f, (v + next) / 2.0));
                }
            }
        }
        let Some((_, feature, threshold)) = best else {
            nodes[at] = self.leaf(&idx);
            return at;
        };
        let (l, r): (Vec<usize>, Vec<usize>) = idx.iter().partition(|&&i| self.x[i][feature] <= threshold);
        let left = self.grow(nodes, l, depth + 1);
        let right = self.grow(nodes, r, depth + 1);
        nodes[at] = Node::Split { feature, threshold, left, right };
        at
    }
}

impl RandomForest {
    /// Trains on rows `x` (all the same length, finite: replace missing values first) and their
    /// classes `y` (`0..n_classes`).
    pub fn train(x: &[Vec<f64>], y: &[usize], n_classes: usize, s: &ForestSettings) -> HistResult<Self> {
        if x.is_empty() || x.len() != y.len() {
            return Err(invalid("training needs examples, one class each"));
        }
        let n_features = x[0].len();
        if n_features == 0 || x.iter().any(|r| r.len() != n_features || r.iter().any(|v| !v.is_finite())) {
            return Err(invalid("every training example needs the same finite features"));
        }
        if y.iter().any(|&c| c >= n_classes) || n_classes < 2 {
            return Err(invalid("training needs examples of at least two classes"));
        }
        let per_split = s.features_per_split.unwrap_or(((n_features as f64).sqrt().round() as usize).max(1)).clamp(1, n_features);
        let mut g = Grower { x, y, n_classes, s, per_split, rng: Rng(s.seed) };
        let mut trees = Vec::with_capacity(s.trees.max(1));
        for _ in 0..s.trees.max(1) {
            let sample: Vec<usize> = (0..x.len()).map(|_| g.rng.below(x.len())).collect();
            let mut nodes = Vec::new();
            g.grow(&mut nodes, sample, 0);
            trees.push(Tree { nodes });
        }
        Ok(Self { n_classes, n_features, trees })
    }

    /// The mean of the trees' class fractions.
    pub fn predict_proba(&self, row: &[f64]) -> Vec<f64> {
        let mut p = vec![0.0; self.n_classes];
        for t in &self.trees {
            for (a, b) in p.iter_mut().zip(t.predict(row)) {
                *a += b;
            }
        }
        let n = self.trees.len().max(1) as f64;
        p.iter().map(|v| v / n).collect()
    }

    /// The most likely class and its probability.
    pub fn predict(&self, row: &[f64]) -> (usize, f64) {
        self.predict_proba(row).into_iter().enumerate().fold((0, f64::MIN), |b, (i, p)| if p > b.1 { (i, p) } else { b })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn learns_a_rule_on_one_of_three_features() {
        let mut rng = Rng(7);
        let mut x = Vec::new();
        let mut y = Vec::new();
        for _ in 0..300 {
            let row: Vec<f64> = (0..3).map(|_| (rng.below(1000) as f64) / 100.0).collect();
            y.push(usize::from(row[1] > 4.0));
            x.push(row);
        }
        let f = RandomForest::train(&x, &y, 2, &ForestSettings { trees: 25, ..ForestSettings::default() }).unwrap();
        assert_eq!(f.predict(&[9.0, 8.0, 1.0]).0, 1);
        assert_eq!(f.predict(&[9.0, 1.0, 9.0]).0, 0);
        let json = serde_json::to_string(&f).unwrap();
        let back: RandomForest = serde_json::from_str(&json).unwrap();
        assert_eq!(back.predict_proba(&[1.0, 6.0, 1.0]), f.predict_proba(&[1.0, 6.0, 1.0]));
        assert!(RandomForest::train(&x, &vec![0; x.len()], 1, &ForestSettings::default()).is_err());
    }
}
