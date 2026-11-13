//! Reading a TorchScript archive (`instanseg.pt`) without PyTorch: a zip whose `data.pkl` (a
//! Python pickle, protocol 2) holds the module tree, and whose `data/<key>` entries hold each
//! tensor's storage (raw little-endian values). Only what `torch.jit.save` writes for a module's
//! parameters is understood: the module objects and their state dicts, `_rebuild_tensor_v2`,
//! persistent storages, numbers, strings, tuples, lists and dicts.
//!
//! The result is flat: every float tensor by its dotted path in the module tree
//! (`fcn.encoder.0.conv0.0.weight`), and every number or boolean attribute
//! (`default_mask_threshold`).

use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap};
use std::io::{Cursor, Read};
use std::rc::Rc;

/// A float tensor: its shape and values (row-major).
#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

/// What an archive holds.
#[derive(Debug, Default)]
pub struct Archive {
    pub tensors: BTreeMap<String, Tensor>,
    /// Numbers and booleans (booleans as 0 / 1).
    pub scalars: BTreeMap<String, f64>,
}

#[derive(Debug, Clone)]
enum Value {
    None,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    Tuple(Vec<Value>),
    List(Rc<RefCell<Vec<Value>>>),
    Dict(Rc<RefCell<Vec<(Value, Value)>>>),
    Global(String, String),
    Object(Rc<RefCell<Object>>),
    /// A storage: its file key and element type (`FloatStorage`, `LongStorage`, …).
    Storage { key: String, kind: String },
    Tensor { key: String, kind: String, offset: usize, size: Vec<usize>, stride: Vec<usize> },
    Mark,
}

#[derive(Debug)]
struct Object {
    state: Value,
}

fn err(what: impl std::fmt::Display) -> String {
    format!("TorchScript archive: {what}")
}

struct Reader<'a> {
    b: &'a [u8],
    at: usize,
}

impl Reader<'_> {
    fn take(&mut self, n: usize) -> Result<&[u8], String> {
        let s = self.b.get(self.at..self.at + n).ok_or_else(|| err("the pickle ends early"))?;
        self.at += n;
        Ok(s)
    }
    fn u8(&mut self) -> Result<u8, String> {
        Ok(self.take(1)?[0])
    }
    fn u16(&mut self) -> Result<u16, String> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().map_err(err)?))
    }
    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().map_err(err)?))
    }
    fn line(&mut self) -> Result<String, String> {
        let rest = &self.b[self.at..];
        let n = rest.iter().position(|&c| c == b'\n').ok_or_else(|| err("a line without its end"))?;
        let s = String::from_utf8(rest[..n].to_vec()).map_err(err)?;
        self.at += n + 1;
        Ok(s)
    }
}

fn int(v: &Value) -> Result<i64, String> {
    match v {
        Value::Int(i) => Ok(*i),
        Value::Bool(b) => Ok(*b as i64),
        other => Err(err(format!("expected an integer, found {other:?}"))),
    }
}

fn usizes(v: &Value) -> Result<Vec<usize>, String> {
    match v {
        Value::Tuple(items) => items.iter().map(|i| int(i).map(|n| n as usize)).collect(),
        Value::List(items) => items.borrow().iter().map(|i| int(i).map(|n| n as usize)).collect(),
        other => Err(err(format!("expected a tuple of sizes, found {other:?}"))),
    }
}

/// Calls a global with `args` (REDUCE): tensors are rebuilt; anything else becomes an object
/// whose state is its arguments (enough for `OrderedDict()` and the like).
fn call(f: Value, args: Value) -> Result<Value, String> {
    let Value::Global(module, name) = &f else { return Err(err(format!("calling {f:?}"))) };
    let items = match args {
        Value::Tuple(items) => items,
        other => return Err(err(format!("arguments {other:?}"))),
    };
    match (module.as_str(), name.as_str()) {
        ("torch._utils", "_rebuild_tensor_v2") | ("torch._utils", "_rebuild_tensor") => {
            let Some(Value::Storage { key, kind }) = items.first() else { return Err(err("a tensor without its storage")) };
            let offset = int(items.get(1).ok_or_else(|| err("a tensor without its offset"))?)? as usize;
            let size = usizes(items.get(2).ok_or_else(|| err("a tensor without its size"))?)?;
            let stride = usizes(items.get(3).ok_or_else(|| err("a tensor without its stride"))?)?;
            Ok(Value::Tensor { key: key.clone(), kind: kind.clone(), offset, size, stride })
        }
        ("collections", "OrderedDict") => Ok(Value::Dict(Rc::new(RefCell::new(Vec::new())))),
        _ => Ok(Value::Object(Rc::new(RefCell::new(Object { state: Value::Tuple(items) })))),
    }
}

/// Runs the pickle machine over `data.pkl` and returns the root value.
fn unpickle(bytes: &[u8]) -> Result<Value, String> {
    let mut r = Reader { b: bytes, at: 0 };
    let mut stack: Vec<Value> = Vec::new();
    let mut memo: HashMap<u32, Value> = HashMap::new();
    let pop = |stack: &mut Vec<Value>| stack.pop().ok_or_else(|| err("the stack is empty"));
    let to_mark = |stack: &mut Vec<Value>| -> Result<Vec<Value>, String> {
        let at = stack.iter().rposition(|v| matches!(v, Value::Mark)).ok_or_else(|| err("no mark"))?;
        let items = stack.split_off(at + 1);
        stack.pop();
        Ok(items)
    };
    loop {
        let op = r.u8()?;
        match op {
            0x80 => {
                r.u8()?;
            } // PROTO
            0x95 => {
                r.take(8)?;
            } // FRAME
            b'.' => return pop(&mut stack), // STOP
            b'(' => stack.push(Value::Mark),
            b')' => stack.push(Value::Tuple(Vec::new())),
            b'}' => stack.push(Value::Dict(Rc::new(RefCell::new(Vec::new())))),
            b']' => stack.push(Value::List(Rc::new(RefCell::new(Vec::new())))),
            b'N' => stack.push(Value::None),
            0x88 => stack.push(Value::Bool(true)),
            0x89 => stack.push(Value::Bool(false)),
            b'K' => stack.push(Value::Int(r.u8()? as i64)),
            b'M' => stack.push(Value::Int(r.u16()? as i64)),
            b'J' => stack.push(Value::Int(r.u32()? as i32 as i64)),
            0x8a => {
                // LONG1: n little-endian bytes, two's complement
                let n = r.u8()? as usize;
                let b = r.take(n)?;
                let mut v: i64 = 0;
                for (i, byte) in b.iter().enumerate().take(8) {
                    v |= (*byte as i64) << (8 * i);
                }
                if n > 0 && n < 8 && b[n - 1] & 0x80 != 0 {
                    v -= 1i64 << (8 * n);
                }
                stack.push(Value::Int(v));
            }
            b'G' => stack.push(Value::Float(f64::from_be_bytes(r.take(8)?.try_into().map_err(err)?))),
            b'X' => {
                let n = r.u32()? as usize;
                stack.push(Value::Str(String::from_utf8(r.take(n)?.to_vec()).map_err(err)?));
            }
            0x8c => {
                let n = r.u8()? as usize;
                stack.push(Value::Str(String::from_utf8(r.take(n)?.to_vec()).map_err(err)?));
            }
            b't' => {
                let items = to_mark(&mut stack)?;
                stack.push(Value::Tuple(items));
            }
            0x85 => {
                let a = pop(&mut stack)?;
                stack.push(Value::Tuple(vec![a]));
            }
            0x86 => {
                let b = pop(&mut stack)?;
                let a = pop(&mut stack)?;
                stack.push(Value::Tuple(vec![a, b]));
            }
            0x87 => {
                let c = pop(&mut stack)?;
                let b = pop(&mut stack)?;
                let a = pop(&mut stack)?;
                stack.push(Value::Tuple(vec![a, b, c]));
            }
            b'q' => {
                let i = r.u8()? as u32;
                memo.insert(i, stack.last().cloned().ok_or_else(|| err("BINPUT on an empty stack"))?);
            }
            b'r' => {
                let i = r.u32()?;
                memo.insert(i, stack.last().cloned().ok_or_else(|| err("LONG_BINPUT on an empty stack"))?);
            }
            0x94 => {
                let i = memo.len() as u32;
                memo.insert(i, stack.last().cloned().ok_or_else(|| err("MEMOIZE on an empty stack"))?);
            }
            b'h' => {
                let i = r.u8()? as u32;
                stack.push(memo.get(&i).cloned().ok_or_else(|| err(format!("memo {i} is empty")))?);
            }
            b'j' => {
                let i = r.u32()?;
                stack.push(memo.get(&i).cloned().ok_or_else(|| err(format!("memo {i} is empty")))?);
            }
            b'c' => {
                let module = r.line()?;
                let name = r.line()?;
                stack.push(Value::Global(module, name));
            }
            b'R' => {
                let args = pop(&mut stack)?;
                let f = pop(&mut stack)?;
                stack.push(call(f, args)?);
            }
            0x81 => {
                // NEWOBJ: a module object, its state set by BUILD
                let _args = pop(&mut stack)?;
                let _cls = pop(&mut stack)?;
                stack.push(Value::Object(Rc::new(RefCell::new(Object { state: Value::None }))));
            }
            b'b' => {
                let state = pop(&mut stack)?;
                match stack.last() {
                    Some(Value::Object(o)) => o.borrow_mut().state = state,
                    Some(Value::Dict(d)) => {
                        if let Value::Dict(s) = state {
                            d.borrow_mut().extend(s.borrow().iter().cloned());
                        }
                    }
                    other => return Err(err(format!("BUILD on {other:?}"))),
                }
            }
            b'Q' => {
                // BINPERSID: ("storage", <class>, key, location, numel)
                let pid = pop(&mut stack)?;
                let Value::Tuple(items) = pid else { return Err(err("a persistent id that is not a tuple")) };
                let kind = match items.get(1) {
                    Some(Value::Global(_, name)) => name.clone(),
                    other => return Err(err(format!("storage type {other:?}"))),
                };
                let key = match items.get(2) {
                    Some(Value::Str(s)) => s.clone(),
                    Some(Value::Int(i)) => i.to_string(),
                    other => return Err(err(format!("storage key {other:?}"))),
                };
                stack.push(Value::Storage { key, kind });
            }
            b's' => {
                let v = pop(&mut stack)?;
                let k = pop(&mut stack)?;
                match stack.last() {
                    Some(Value::Dict(d)) => d.borrow_mut().push((k, v)),
                    other => return Err(err(format!("SETITEM on {other:?}"))),
                }
            }
            b'u' => {
                let items = to_mark(&mut stack)?;
                match stack.last() {
                    Some(Value::Dict(d)) => {
                        let mut d = d.borrow_mut();
                        for pair in items.chunks(2) {
                            if let [k, v] = pair {
                                d.push((k.clone(), v.clone()));
                            }
                        }
                    }
                    other => return Err(err(format!("SETITEMS on {other:?}"))),
                }
            }
            b'a' => {
                let v = pop(&mut stack)?;
                match stack.last() {
                    Some(Value::List(l)) => l.borrow_mut().push(v),
                    other => return Err(err(format!("APPEND on {other:?}"))),
                }
            }
            b'e' => {
                let items = to_mark(&mut stack)?;
                match stack.last() {
                    Some(Value::List(l)) => l.borrow_mut().extend(items),
                    other => return Err(err(format!("APPENDS on {other:?}"))),
                }
            }
            other => return Err(err(format!("pickle opcode 0x{other:02x} at byte {} is not supported", r.at - 1))),
        }
    }
}

/// Every tensor and scalar under `value`, by dotted path.
fn flatten(value: &Value, path: &str, found: &mut Vec<(String, Value)>) {
    let child = |k: &str| if path.is_empty() { k.to_string() } else { format!("{path}.{k}") };
    match value {
        Value::Object(o) => flatten(&o.borrow().state, path, found),
        Value::Dict(d) => {
            for (k, v) in d.borrow().iter() {
                if let Value::Str(k) = k {
                    flatten(v, &child(k), found);
                }
            }
        }
        Value::Tensor { .. } | Value::Int(_) | Value::Float(_) | Value::Bool(_) => found.push((path.to_string(), value.clone())),
        _ => {}
    }
}

/// The values of a float tensor from its storage file (any strides).
fn gather(storage: &[u8], offset: usize, size: &[usize], stride: &[usize]) -> Result<Vec<f32>, String> {
    let n: usize = size.iter().product();
    let mut out = Vec::with_capacity(n);
    let mut idx = vec![0usize; size.len()];
    for _ in 0..n {
        let at = offset + idx.iter().zip(stride).map(|(i, s)| i * s).sum::<usize>();
        let b = storage.get(at * 4..at * 4 + 4).ok_or_else(|| err("a tensor reaches past its storage"))?;
        out.push(f32::from_le_bytes(b.try_into().map_err(err)?));
        for d in (0..size.len()).rev() {
            idx[d] += 1;
            if idx[d] < size[d] {
                break;
            }
            idx[d] = 0;
        }
    }
    Ok(out)
}

/// Reads a TorchScript archive (the bytes of `instanseg.pt`).
pub fn read(bytes: &[u8]) -> Result<Archive, String> {
    let mut zip = zip::ZipArchive::new(Cursor::new(bytes)).map_err(err)?;
    let names: Vec<String> = zip.file_names().filter_map(|n| n.ok().map(|n| n.into_owned())).collect();
    let pkl = names.iter().find(|n| n.ends_with("/data.pkl") && n.matches('/').count() == 1).ok_or_else(|| err("no data.pkl"))?.clone();
    let root_dir = pkl.trim_end_matches("data.pkl").to_string();
    if let Some(order) = names.iter().find(|n| **n == format!("{root_dir}byteorder")) {
        let mut s = String::new();
        zip.by_name(order).map_err(err)?.read_to_string(&mut s).map_err(err)?;
        if s.trim() != "little" {
            return Err(err(format!("byte order {s:?}: only little-endian archives are read")));
        }
    }
    let mut data = Vec::new();
    zip.by_name(&pkl).map_err(err)?.read_to_end(&mut data).map_err(err)?;
    let root = unpickle(&data)?;
    let mut found = Vec::new();
    flatten(&root, "", &mut found);
    let mut archive = Archive::default();
    let mut storages: HashMap<String, Vec<u8>> = HashMap::new();
    for (path, v) in found {
        match v {
            Value::Tensor { key, kind, offset, size, stride } => {
                if kind != "FloatStorage" {
                    continue; // counters (`num_batches_tracked`) and index constants
                }
                if !storages.contains_key(&key) {
                    let mut b = Vec::new();
                    zip.by_name(&format!("{root_dir}data/{key}")).map_err(|e| err(format!("storage {key}: {e}")))?.read_to_end(&mut b).map_err(err)?;
                    storages.insert(key.clone(), b);
                }
                let values = gather(&storages[&key], offset, &size, &stride).map_err(|e| format!("{path}: {e}"))?;
                archive.tensors.insert(path, Tensor { shape: size, data: values });
            }
            Value::Int(i) => {
                archive.scalars.insert(path, i as f64);
            }
            Value::Float(f) => {
                archive.scalars.insert(path, f);
            }
            Value::Bool(b) => {
                archive.scalars.insert(path, b as u8 as f64);
            }
            _ => {}
        }
    }
    if archive.tensors.is_empty() {
        return Err(err("no float tensors"));
    }
    Ok(archive)
}

/// The TorchScript archive (`instanseg.pt`) inside a bioimage.io package (a zip).
pub fn from_package(package: &[u8]) -> Result<Vec<u8>, String> {
    let mut zip = zip::ZipArchive::new(Cursor::new(package)).map_err(|e| format!("the model package: {e}"))?;
    let name = zip.file_names().filter_map(|n| n.ok().map(|n| n.into_owned())).find(|n| n.ends_with(".pt")).ok_or("the model package holds no TorchScript (.pt) file")?;
    let mut out = Vec::new();
    zip.by_name(&name).map_err(|e| format!("{name}: {e}"))?.read_to_end(&mut out).map_err(|e| format!("{name}: {e}"))?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_hand_written_pickle_reads() {
        // {"a": 1, "b": (2.5, True), "c": "x"}, protocol 2
        let mut p = vec![0x80, 2, b'}', b'q', 0, b'('];
        p.extend([0x8c, 1, b'a', b'K', 1]);
        p.extend([0x8c, 1, b'b', b'G']);
        p.extend(2.5f64.to_be_bytes());
        p.extend([0x88, 0x86]);
        p.extend([0x8c, 1, b'c', 0x8c, 1, b'x', b'u', b'.']);
        let v = unpickle(&p).unwrap();
        let mut found = Vec::new();
        flatten(&v, "", &mut found);
        let names: Vec<&str> = found.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names, ["a"]);
        let Value::Dict(d) = v else { panic!("{v:?}") };
        assert_eq!(d.borrow().len(), 3);
    }

    #[test]
    fn strided_tensors_gather_in_row_major_order() {
        // A 2 × 3 storage read transposed (3 × 2, strides 1, 3)
        let storage: Vec<u8> = (0..6).flat_map(|i| (i as f32).to_le_bytes()).collect();
        assert_eq!(gather(&storage, 0, &[3, 2], &[1, 3]).unwrap(), [0.0, 3.0, 1.0, 4.0, 2.0, 5.0]);
    }
}
