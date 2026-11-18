//! The desktop's side of [`crate::platform`].

/// Where images come from, for the title bar.
pub fn host_label() -> String {
    "This computer".into()
}

/// Where the session is kept between runs: the config folder.
const SESSION_FILE: &str = "current-session.lucem.json";

/// Keeps `json` (the session as it is now) for the next start; what went wrong, if anything.
pub fn keep_session(json: &str) -> Result<(), String> {
    let dir = crate::data::prefs::Prefs::dir().ok_or("no config folder")?;
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let file = dir.join(SESSION_FILE);
    let tmp = file.with_extension("json.tmp");
    std::fs::write(&tmp, json).and_then(|_| std::fs::rename(&tmp, &file)).map_err(|e| e.to_string())
}

/// The session kept by the last run, if any.
pub fn kept_session() -> Option<String> {
    std::fs::read_to_string(crate::data::prefs::Prefs::dir()?.join(SESSION_FILE)).ok()
}
