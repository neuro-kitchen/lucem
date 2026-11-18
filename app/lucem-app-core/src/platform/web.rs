//! A browser page's side of [`crate::platform`].

/// Where images come from, for the title bar: the server that sent the page, or "This browser"
/// for a page that runs alone (the loader found no `lucem serve` behind it).
pub fn host_label() -> String {
    let window = web_sys::window();
    let served = window
        .as_ref()
        .and_then(|w| js_sys::Reflect::get(w, &"lucemWeb".into()).ok())
        .and_then(|web| js_sys::Reflect::get(&web, &"server".into()).ok())
        .is_some_and(|s| !s.is_null() && !s.is_undefined());
    if !served {
        return "This browser".into();
    }
    window.and_then(|w| w.location().host().ok()).filter(|h| !h.is_empty()).unwrap_or_else(|| "This browser".into())
}

/// Where the session is kept between visits: the browser's storage for this site (a served
/// page: next, on its server under its sign-in).
const SESSION_KEY: &str = "lucem-session";

/// Keeps `json` (the session as it is now) for the next visit; what went wrong, if anything.
pub fn keep_session(json: &str) -> Result<(), String> {
    let storage = web_sys::window().and_then(|w| w.local_storage().ok().flatten()).ok_or("no browser storage")?;
    storage.set_item(SESSION_KEY, json).map_err(|e| format!("the browser did not keep it: {e:?}"))
}

/// The session kept by the last visit, if any.
pub fn kept_session() -> Option<String> {
    web_sys::window()?.local_storage().ok()??.get_item(SESSION_KEY).ok()?
}
