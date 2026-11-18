//! The preferences a user edits by hand: `settings.toml` beside `prefs.json` (in
//! `$XDG_CONFIG_HOME/lucem/`). The Settings screen writes the same file, keeping its comments and
//! order; the app applies an edit made outside while it runs (the compute runtime at the next
//! start). A missing file is written with every default, so the user sees what can be changed. A
//! value that cannot be read keeps its default and is named in a notice. What the app remembers
//! by itself (recent files, panels) stays in `prefs.json`.

use toml_edit::{DocumentMut, Item, Table, Value};

use crate::data::prefs::Prefs;

/// The preferences kept in `settings.toml`.
#[derive(Debug, Clone, PartialEq)]
pub struct Settings {
    /// `Some(true)`: dark, `Some(false)`: light, `None`: follow the system.
    pub theme: Option<bool>,
    /// `None`: lucem's own accent.
    pub accent: Option<kitchen_ui::Rgb>,
    pub font_size: f32,
    /// 0 grey, 1 dark, 2 light.
    pub pasteboard: u8,
    /// `None`: automatic.
    pub runtime: Option<String>,
    /// 0: the machine's cores (at most 8).
    pub tile_threads: usize,
    /// `None`: the plugin's default.
    pub segment_model: Option<String>,
}

impl Default for Settings {
    fn default() -> Self {
        Self::from_prefs(&Prefs::default())
    }
}

const TEMPLATE: &str = r##"# lucem-app settings. The Settings screen writes this file; edits made here apply while the app runs.
# Delete a line to go back to its default.

[appearance]
theme = "dark"          # "system" | "dark" | "light"
accent = "app"          # "app" or a colour as "#rrggbb"
font_size = 15.0        # interface text size, px
pasteboard = "grey"     # around the image: "grey" | "dark" | "light"

[compute]
runtime = "automatic"   # "automatic" | "wgpu" | "cuda" | "cpu"; takes effect at the next start
tile_threads = 0        # threads reading tiles; 0 is the machine's cores, at most 8

[models]
segment = ""            # the Segment tool's model (a plugin's model id); "" is the plugin's default
"##;

/// What `settings.toml` may hold: (section, keys).
const KNOWN: &[(&str, &[&str])] = &[("appearance", &["theme", "accent", "font_size", "pasteboard"]), ("compute", &["runtime", "tile_threads"]), ("models", &["segment"])];

const PASTEBOARDS: [&str; 3] = ["grey", "dark", "light"];

impl Settings {
    pub fn from_prefs(p: &Prefs) -> Self {
        Self { theme: p.theme, accent: p.accent, font_size: p.font_size, pasteboard: p.pasteboard.min(2), runtime: p.runtime.clone().filter(|r| !r.is_empty()), tile_threads: p.tile_threads, segment_model: p.ml_model.clone().filter(|m| !m.is_empty()) }
    }

    pub fn apply_to(&self, p: &mut Prefs) {
        p.theme = self.theme;
        p.accent = self.accent;
        p.font_size = self.font_size;
        p.pasteboard = self.pasteboard;
        p.runtime = self.runtime.clone();
        p.tile_threads = self.tile_threads;
        p.ml_model = self.segment_model.clone();
    }

    /// The settings in `text` and a notice per line that was not understood (which keeps its
    /// default). `Err`: the file is not TOML, nothing is applied.
    pub fn parse(text: &str) -> Result<(Self, Vec<String>), String> {
        let doc: DocumentMut = text.parse().map_err(|e| format!("settings.toml: {e}"))?;
        let mut notices = Vec::new();
        for (section, item) in doc.iter() {
            let Some(keys) = KNOWN.iter().find(|(s, _)| *s == section).map(|(_, k)| *k) else {
                notices.push(format!("settings.toml: unknown section [{section}]"));
                continue;
            };
            for (key, _) in item.as_table_like().into_iter().flat_map(|t| t.iter()) {
                if !keys.contains(&key) {
                    notices.push(format!("settings.toml: unknown setting {section}.{key}"));
                }
            }
        }
        let get = |section: &str, key: &str| doc.get(section).and_then(|s| s.get(key)).and_then(Item::as_value);
        let mut out = Self::default();
        let text_of = |section: &str, key: &str, notices: &mut Vec<String>| match get(section, key) {
            None => None,
            Some(v) => match v.as_str() {
                Some(t) => Some(t.trim().to_string()),
                None => {
                    notices.push(format!("settings.toml: {section}.{key} must be text in quotes"));
                    None
                }
            },
        };
        match text_of("appearance", "theme", &mut notices).as_deref() {
            None | Some("dark") => {}
            Some("system") => out.theme = None,
            Some("light") => out.theme = Some(false),
            Some(other) => notices.push(format!("settings.toml: appearance.theme {other:?} is not \"system\", \"dark\" or \"light\"")),
        }
        match text_of("appearance", "accent", &mut notices).as_deref() {
            None | Some("app") | Some("") => {}
            Some(colour) => match kitchen_ui::Rgb::parse(colour) {
                Some(rgb) => out.accent = Some(rgb),
                None => notices.push(format!("settings.toml: appearance.accent {colour:?} is not \"app\" or \"#rrggbb\"")),
            },
        }
        if let Some(v) = get("appearance", "font_size") {
            match v.as_float().or_else(|| v.as_integer().map(|i| i as f64)).filter(|f| (9.0..=28.0).contains(f)) {
                Some(f) => out.font_size = f as f32,
                None => notices.push("settings.toml: appearance.font_size must be a number from 9 to 28".into()),
            }
        }
        match text_of("appearance", "pasteboard", &mut notices).as_deref() {
            None => {}
            Some(name) => match PASTEBOARDS.iter().position(|p| *p == name) {
                Some(i) => out.pasteboard = i as u8,
                None => notices.push(format!("settings.toml: appearance.pasteboard {name:?} is not \"grey\", \"dark\" or \"light\"")),
            },
        }
        match text_of("compute", "runtime", &mut notices).as_deref() {
            None | Some("automatic") | Some("") => {}
            Some(name) => out.runtime = Some(name.to_string()),
        }
        if let Some(v) = get("compute", "tile_threads") {
            match v.as_integer().filter(|n| (0..=64).contains(n)) {
                Some(n) => out.tile_threads = n as usize,
                None => notices.push("settings.toml: compute.tile_threads must be a whole number from 0 to 64".into()),
            }
        }
        match text_of("models", "segment", &mut notices).as_deref() {
            None | Some("") => {}
            Some(id) => out.segment_model = Some(id.to_string()),
        }
        Ok((out, notices))
    }

    /// `current` (the file as it is; empty or unreadable: the commented template) with these
    /// values, comments and order kept.
    pub fn write_into(&self, current: &str) -> String {
        let mut doc: DocumentMut = current.parse().ok().filter(|_| !current.trim().is_empty()).unwrap_or_else(|| TEMPLATE.parse().expect("the template is TOML"));
        let theme = match self.theme {
            None => "system",
            Some(true) => "dark",
            Some(false) => "light",
        };
        set(&mut doc, "appearance", "theme", theme.into());
        set(&mut doc, "appearance", "accent", self.accent.map(|c| format!("#{:06x}", c.0)).unwrap_or_else(|| "app".into()).into());
        set(&mut doc, "appearance", "font_size", (self.font_size as f64).into());
        set(&mut doc, "appearance", "pasteboard", PASTEBOARDS[self.pasteboard.min(2) as usize].into());
        set(&mut doc, "compute", "runtime", self.runtime.clone().unwrap_or_else(|| "automatic".into()).into());
        set(&mut doc, "compute", "tile_threads", (self.tile_threads as i64).into());
        set(&mut doc, "models", "segment", self.segment_model.clone().unwrap_or_default().into());
        doc.to_string()
    }
}

/// What reading `settings.toml` said (each notice is given once).
static NOTICES: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());

pub(crate) fn notice(text: String) {
    NOTICES.lock().unwrap_or_else(|e| e.into_inner()).push(text);
}

/// The notices collected so far.
pub fn take_notices() -> Vec<String> {
    std::mem::take(&mut *NOTICES.lock().unwrap_or_else(|e| e.into_inner()))
}

/// Sets `section.key`, keeping the comment after the value.
fn set(doc: &mut DocumentMut, section: &str, key: &str, mut new: Value) {
    if doc.get(section).and_then(Item::as_table_like).is_none() {
        doc.insert(section, Item::Table(Table::new()));
    }
    let Some(table) = doc.get_mut(section).and_then(Item::as_table_like_mut) else { return };
    match table.get_mut(key).and_then(Item::as_value_mut) {
        Some(old) => {
            *new.decor_mut() = old.decor().clone();
            *old = new;
        }
        None => {
            table.insert(key, Item::Value(new));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn settings_round_trip_and_keep_the_comments() {
        let s = Settings { theme: Some(false), accent: Some(kitchen_ui::Rgb(0x2b7f86)), font_size: 14.0, pasteboard: 2, runtime: Some("cuda".into()), tile_threads: 4, segment_model: Some("sam2-tiny".into()) };
        let text = s.write_into("");
        // The value changes; the comment after it stays
        assert!(text.contains("theme = \"light\"") && text.contains("# \"system\" | \"dark\" | \"light\""), "{text}");
        assert!(text.contains("accent = \"#2b7f86\""), "{text}");
        assert_eq!(Settings::parse(&text).unwrap(), (s.clone(), vec![]));
        let edited = "# mine\n[compute]\nruntime = \"cpu\" # slow\n";
        let (read, notices) = Settings::parse(edited).unwrap();
        assert_eq!((read.runtime.as_deref(), notices.len()), (Some("cpu"), 0));
        let written = Settings { runtime: Some("wgpu".into()), ..read }.write_into(edited);
        assert!(written.starts_with("# mine\n[compute]\nruntime = \"wgpu\" # slow\n"), "{written}");
    }

    #[test]
    fn the_defaults_are_the_template() {
        assert_eq!(Settings::default().write_into(""), TEMPLATE);
        assert_eq!(Settings::parse("").unwrap(), (Settings::default(), vec![]));
        assert_eq!(Settings::parse(TEMPLATE).unwrap(), (Settings::default(), vec![]));
    }

    #[test]
    fn what_is_not_understood_keeps_its_default_and_is_named() {
        let (s, notices) = Settings::parse("[appearance]\ntheme = \"blue\"\nfont_size = 90\npasteboard = \"pink\"\nfont = 3\n[colours]\n").unwrap();
        assert_eq!(s, Settings::default());
        assert_eq!(notices.len(), 5, "{notices:?}");
        assert!(Settings::parse("[appearance\n").is_err());
    }
}
