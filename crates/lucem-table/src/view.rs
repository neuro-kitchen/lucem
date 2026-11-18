//! A table as a view reads it: each cell's text and number, filters typed by people (`> 10`,
//! `10..50`, `= Tumor`, text), the search, the sort, and CSV out and in. What the app's Analyze
//! screen shows, computed here so a server answers a page the same way.

use polars::prelude::{AnyValue, Column, DataFrame, NamedFrom as _, Series};

/// The table in a form the filters, the sort and the plots read quickly.
pub struct TableView {
    pub names: Vec<String>,
    /// Per column: each row's text and, for numeric columns, its number.
    pub text: Vec<Vec<String>>,
    pub num: Vec<Vec<Option<f64>>>,
    pub numeric: Vec<bool>,
    /// The annotation of each row (the annotations table, or any table with an `id` column).
    pub ids: Option<Vec<String>>,
    /// The detected object of each row (`c12`, `t5`: the Cells and Tiles tables).
    pub objects: Option<Vec<String>>,
}

fn text_of(v: &AnyValue) -> String {
    match v {
        AnyValue::Null => String::new(),
        AnyValue::String(s) => s.to_string(),
        AnyValue::StringOwned(s) => s.to_string(),
        other => other.to_string().trim_matches('"').to_string(),
    }
}

impl TableView {
    /// The table `df`, its cells as text (numbers shortened) and, for numeric columns, numbers.
    pub fn of(df: &DataFrame) -> TableView {
        let cols = df.columns();
        let n = df.height();
        let mut m = TableView { names: Vec::new(), text: Vec::new(), num: Vec::new(), numeric: Vec::new(), ids: None, objects: None };
        for c in cols {
            let values: Vec<AnyValue> = (0..n).map(|i| c.get(i).unwrap_or(AnyValue::Null)).collect();
            let num: Vec<Option<f64>> = values.iter().map(|v| if v.is_null() { None } else { v.extract::<f64>() }).collect();
            let numeric = values.iter().zip(&num).all(|(v, x)| v.is_null() || x.is_some());
            let text: Vec<String> = values.iter().zip(&num).map(|(v, x)| match (numeric, x) {
                (true, Some(x)) => short_num(*x),
                _ => text_of(v),
            }).collect();
            if c.name().as_str() == "id" {
                m.ids = Some(text.clone());
            }
            if c.name().as_str() == "object" {
                m.objects = Some(text.clone());
            }
            m.names.push(c.name().to_string());
            m.text.push(text);
            m.num.push(num);
            m.numeric.push(numeric);
        }
        m
    }

    pub fn rows(&self) -> usize {
        self.text.first().map_or(0, Vec::len)
    }

    /// Whether row `r` passes column `c`'s filter (`> 10`, `>= 1`, `< 5`, `<= 5`, `10..50`,
    /// `= Tumor`, `!= Tumor`, or text it contains).
    pub fn passes(&self, c: usize, r: usize, f: &str) -> bool {
        let f = f.trim();
        if f.is_empty() {
            return true;
        }
        let text = &self.text[c][r];
        let num = self.num[c][r];
        let number = |s: &str| s.trim().parse::<f64>().ok();
        if let Some((a, b)) = f.split_once("..") {
            return match (num, number(a), number(b)) {
                (Some(x), lo, hi) => lo.is_none_or(|lo| x >= lo) && hi.is_none_or(|hi| x <= hi),
                _ => false,
            };
        }
        for (op, test) in [(">=", 0), ("<=", 1), ("!=", 2), (">", 3), ("<", 4), ("=", 5)] {
            if let Some(rest) = f.strip_prefix(op) {
                let rest = rest.trim();
                return match (test, num, number(rest)) {
                    (0, Some(x), Some(v)) => x >= v,
                    (1, Some(x), Some(v)) => x <= v,
                    (3, Some(x), Some(v)) => x > v,
                    (4, Some(x), Some(v)) => x < v,
                    (2, _, _) => !text.eq_ignore_ascii_case(rest),
                    (5, _, _) => text.eq_ignore_ascii_case(rest),
                    _ => false,
                };
            }
        }
        text.to_lowercase().contains(&f.to_lowercase())
    }

    /// The rows that pass every filter and the search, in the sort's order.
    pub fn order(&self, filters: &[(usize, String)], search: &str, sort: Option<(usize, bool)>) -> Vec<usize> {
        let search = search.trim().to_lowercase();
        let mut rows: Vec<usize> = (0..self.rows())
            .filter(|&r| filters.iter().all(|(c, f)| self.passes(*c, r, f)))
            .filter(|&r| search.is_empty() || self.text.iter().any(|col| col[r].to_lowercase().contains(&search)))
            .collect();
        if let Some((c, desc)) = sort {
            if self.numeric[c] {
                rows.sort_by(|&a, &b| self.num[c][a].unwrap_or(f64::NEG_INFINITY).total_cmp(&self.num[c][b].unwrap_or(f64::NEG_INFINITY)));
            } else {
                rows.sort_by(|&a, &b| self.text[c][a].cmp(&self.text[c][b]));
            }
            if desc {
                rows.reverse();
            }
        }
        rows
    }

    /// The rows in `order` as CSV (with a header).
    pub fn csv(&self, order: &[usize]) -> String {
        let mut out = self.names.join(",");
        out.push('\n');
        for &r in order {
            let row: Vec<String> = (0..self.names.len()).map(|c| if self.numeric[c] { self.num[c][r].map_or(String::new(), |x| x.to_string()) } else { cell(&self.text[c][r]) }).collect();
            out.push_str(&row.join(","));
            out.push('\n');
        }
        out
    }
}

/// A number shortened for the table.
pub fn short_num(v: f64) -> String {
    if v.fract() == 0.0 && v.abs() < 1e12 {
        format!("{}", v as i64)
    } else if v.abs() >= 1000.0 {
        format!("{v:.1}")
    } else {
        format!("{v:.3}")
    }
}

/// A CSV cell (quoted when it holds a comma or quote).
pub fn cell(s: &str) -> String {
    if s.contains(',') || s.contains('"') || s.contains('\n') { format!("\"{}\"", s.replace('"', "\"\"")) } else { s.to_string() }
}

/// Splits a CSV line (quotes, doubled quotes).
pub fn split_csv(line: &str) -> Vec<String> {
    let (mut out, mut cur, mut quoted, mut chars) = (Vec::new(), String::new(), false, line.chars().peekable());
    while let Some(ch) = chars.next() {
        match ch {
            '"' if quoted && chars.peek() == Some(&'"') => {
                cur.push('"');
                chars.next();
            }
            '"' => quoted = !quoted,
            ',' if !quoted => out.push(std::mem::take(&mut cur)),
            c => cur.push(c),
        }
    }
    out.push(cur);
    out
}

/// A CSV text (a header row first) as a frame: a column is numeric when every non-empty cell is
/// a number.
pub fn parse_csv(text: &str) -> Result<DataFrame, String> {
    let mut lines = text.lines().filter(|l| !l.trim().is_empty());
    let names = split_csv(lines.next().ok_or("the result table is empty")?);
    let rows: Vec<Vec<String>> = lines.map(split_csv).collect();
    let n = rows.len();
    let cols: Vec<Column> = names
        .iter()
        .enumerate()
        .map(|(c, name)| {
            let v: Vec<&str> = rows.iter().map(|r| r.get(c).map_or("", |s| s.as_str())).collect();
            let blank = |s: &str| s.is_empty() || s == "null" || s == "NaN" || s == "nan";
            if v.iter().all(|s| blank(s) || s.parse::<f64>().is_ok()) {
                Column::from(Series::new(name.as_str().into(), v.iter().map(|s| if blank(s) { None } else { s.parse::<f64>().ok() }).collect::<Vec<Option<f64>>>()))
            } else {
                Column::from(Series::new(name.as_str().into(), v.iter().map(|s| (!blank(s)).then(|| s.to_string())).collect::<Vec<Option<String>>>()))
            }
        })
        .collect();
    DataFrame::new(n, cols).map_err(|e| e.to_string())
}


#[cfg(test)]
mod tests {
    use super::*;

    fn frame() -> DataFrame {
        parse_csv("id,class,area\na1,Tumor,120\na2,Stroma,8.5\na3,\"Tumor, edge\",40\n").unwrap()
    }

    /// A frame read as a view: numbers shortened, the id column found; filters, search and sort
    /// pick and order the rows; CSV goes out and back the same.
    #[test]
    fn filters_sort_and_csv() {
        let m = TableView::of(&frame());
        assert_eq!(m.rows(), 3);
        assert_eq!(m.names, ["id", "class", "area"]);
        assert!(m.numeric[2] && !m.numeric[1]);
        assert_eq!(m.text[2], ["120", "8.500", "40"]);
        assert_eq!(m.ids.as_deref(), Some(&["a1".to_string(), "a2".into(), "a3".into()][..]));
        assert_eq!(m.order(&[(2, "> 10".into())], "", None), [0, 2]);
        assert_eq!(m.order(&[(2, "10..50".into())], "", None), [2]);
        assert_eq!(m.order(&[(1, "= tumor".into())], "", None), [0]);
        assert_eq!(m.order(&[], "edge", None), [2]);
        assert_eq!(m.order(&[], "", Some((2, true))), [0, 2, 1]);
        let csv = m.csv(&[0, 2]);
        assert_eq!(csv, "id,class,area\na1,Tumor,120\na3,\"Tumor, edge\",40\n");
        assert_eq!(TableView::of(&parse_csv(&csv).unwrap()).rows(), 2);
        assert!(parse_csv("").is_err());
    }
}
