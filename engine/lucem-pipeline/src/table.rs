//! Measurements gathered over a walk: named numeric columns, one row per block or object.

use std::collections::HashMap;
use std::fs;
use std::io::Write as _;
use std::path::Path;

use polars::prelude::{Column, DataFrame, NamedFrom as _, PolarsResult, Series};

/// A table of `f64` columns (missing values are NaN).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Table {
    pub columns: Vec<(String, Vec<f64>)>,
    rows: usize,
}

impl Table {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn is_empty(&self) -> bool {
        self.rows == 0
    }

    /// The column called `name`.
    pub fn column(&self, name: &str) -> Option<&[f64]> {
        self.columns.iter().find(|(n, _)| n == name).map(|(_, v)| v.as_slice())
    }

    /// Adds a row; a column not seen before is NaN in the earlier rows, and a column this row
    /// does not name is NaN in it.
    pub fn push_row(&mut self, row: &[(String, f64)]) {
        let index: HashMap<&str, f64> = row.iter().map(|(n, v)| (n.as_str(), *v)).collect();
        for (name, _) in row {
            if !self.columns.iter().any(|(n, _)| n == name) {
                self.columns.push((name.clone(), vec![f64::NAN; self.rows]));
            }
        }
        for (name, values) in &mut self.columns {
            values.push(index.get(name.as_str()).copied().unwrap_or(f64::NAN));
        }
        self.rows += 1;
    }

    /// As a Polars frame, for analysis.
    pub fn to_frame(&self) -> PolarsResult<DataFrame> {
        let columns: Vec<Column> = self.columns.iter().map(|(n, v)| Column::from(Series::new(n.as_str().into(), v.as_slice()))).collect();
        DataFrame::new(self.rows, columns)
    }

    /// Writes the table as CSV at `path`, which must not exist yet.
    pub fn write_csv(&self, path: &Path) -> std::io::Result<()> {
        let mut f = std::io::BufWriter::new(fs::OpenOptions::new().write(true).create_new(true).open(path)?);
        let header: Vec<String> = self.columns.iter().map(|(n, _)| escape(n)).collect();
        writeln!(f, "{}", header.join(","))?;
        for r in 0..self.rows {
            let row: Vec<String> = self.columns.iter().map(|(_, v)| if v[r].is_nan() { String::new() } else { v[r].to_string() }).collect();
            writeln!(f, "{}", row.join(","))?;
        }
        f.flush()
    }
}

fn escape(name: &str) -> String {
    if name.contains([',', '"', '\n']) { format!("\"{}\"", name.replace('"', "\"\"")) } else { name.to_string() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rows_may_name_different_columns() {
        let mut t = Table::new();
        t.push_row(&[("a".into(), 1.0)]);
        t.push_row(&[("b".into(), 2.0)]);
        assert_eq!(t.rows(), 2);
        assert_eq!(t.column("a").unwrap()[0], 1.0);
        assert!(t.column("a").unwrap()[1].is_nan());
        assert!(t.column("b").unwrap()[0].is_nan());
        assert_eq!(t.to_frame().unwrap().height(), 2);
    }

    #[test]
    fn csv_is_never_written_over_a_file() {
        let path = std::env::temp_dir().join(format!("lucem-table-{}.csv", std::process::id()));
        let _ = fs::remove_file(&path);
        let mut t = Table::new();
        t.push_row(&[("area, px".into(), 3.0)]);
        t.write_csv(&path).unwrap();
        assert_eq!(fs::read_to_string(&path).unwrap(), "\"area, px\"\n3\n");
        assert!(t.write_csv(&path).is_err());
        fs::remove_file(&path).unwrap();
    }
}
