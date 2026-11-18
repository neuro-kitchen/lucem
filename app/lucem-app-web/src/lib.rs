//! lucem in a browser tab: the entry point only. Everything the app does is `lucem-app-core`,
//! the same as the desktop's; what differs in a browser is where images come from (the page's
//! server, `lucem serve --web`) and that the page computes nothing itself (the server does).
//!
//! Built by `build.sh` (nightly: GPUI's web platform needs it) into `dist/`: `index.html` (a
//! loader that checks the browser and holds the WebTransport shim) and `mt/` (the threaded app).
#![cfg(target_family = "wasm")]

use std::sync::Arc;

use gpui_kit::component::TitleBar;
use gpui_kit::{App, AppContext as _, TitlebarOptions, WindowOptions};
use lucem_app_core::app::LucemApp;
use lucem_app_core::data::prefs::Prefs;
use lucem_app_core::data::store::Store;
use lucem_app_core::navigation::actions;
use std::rc::Rc;

use futures_util::StreamExt as _;
use lucem_app_core::screens::sign_in::SignIn;
use lucem_stream::web_client::{self, Hello};
use wasm_bindgen::prelude::*;

/// The font the page brings (a browser gives it none).
const FONT: &str = "Adwaita Sans";
/// Where the page keeps its token (from pairing with the server).
const TOKEN_KEY: &str = "lucem-token";

unsafe extern "C" {
    /// Runs the static constructors (GPUI's actions, CubeCL's IR). Named so the linker does not
    /// wrap every export in a call to them (dsp-kitchen measured half a frame spent there).
    fn __wasm_call_ctors();
}

/// The value of `name` in the page's address (`?name=value`), decoded.
fn link_parameter(name: &str) -> Option<String> {
    let search = web_sys::window()?.location().search().ok()?;
    let value = search.trim_start_matches('?').split('&').find_map(|pair| pair.strip_prefix(name)?.strip_prefix('='))?;
    js_sys::decode_uri_component(value).ok().map(String::from).filter(|v| !v.is_empty())
}

/// What the loader chose to draw with (`window.lucemWeb.drawing`).
fn backend() -> gpui_kit::web::WebBackendPreference {
    use gpui_kit::web::WebBackendPreference as B;
    let chosen = web_sys::window().and_then(|w| js_sys::Reflect::get(&w, &"lucemWeb".into()).ok()).and_then(|web| js_sys::Reflect::get(&web, &"drawing".into()).ok()).and_then(|d| d.as_string());
    match chosen.as_deref() {
        Some("webgpu") => B::WebGpu,
        Some("webgl") => B::WebGl,
        _ => B::Auto,
    }
}

fn storage() -> Option<web_sys::Storage> {
    web_sys::window()?.local_storage().ok()?
}

/// The pairing code of the link the page was opened with (`#pair=<code>`), taken out of the
/// address bar.
fn link_code() -> Option<String> {
    let window = web_sys::window()?;
    let hash = window.location().hash().ok()?;
    let code = hash.trim_start_matches('#').split('&').find_map(|p| p.strip_prefix("pair="))?.to_string();
    if let Ok(history) = window.history() {
        let clean = format!("{}{}", window.location().pathname().unwrap_or_default(), window.location().search().unwrap_or_default());
        let _ = history.replace_state_with_url(&wasm_bindgen::JsValue::NULL, "", Some(&clean));
    }
    (!code.is_empty()).then_some(code)
}

fn kept_token() -> Option<String> {
    storage().and_then(|s| s.get_item(TOKEN_KEY).ok().flatten())
}

/// A name of this browser's own for the server's list of paired clients.
fn browser_name() -> String {
    format!("browser-{:06x}", (js_sys::Math::random() * f64::from(0xff_ffff)) as u32)
}

/// Signs the page in to its server: with the kept token; else (or when the server no longer
/// accepts it: revoked) with a pairing code typed into the app's sign-in form, tried until one
/// works. The form says why a code did not.
async fn sign_in(info: &lucem_stream::ServeInfo, (window, app): (gpui_kit::AnyWindowHandle, gpui_kit::Entity<LucemApp>), cx: &mut gpui_kit::AsyncApp) -> Result<Arc<web_client::PageLink>, String> {
    // The link `lucem serve` printed (`…/#pair=<code>`): signs in with nothing to type. The code
    // leaves the address bar at once (it is spent anyway)
    if let Some(code) = link_code() {
        let name = browser_name();
        match web_client::connect(info, || Hello::Code { code: code.clone(), name: name.clone() }).await {
            Ok((link, token)) => {
                if let (Some(token), Some(s)) = (token, storage()) {
                    let _ = s.set_item(TOKEN_KEY, &token);
                }
                return Ok(link);
            }
            Err(e) => log::info!("lucem: the link's code did not sign in ({e})"),
        }
    }
    if let Some(token) = kept_token() {
        match web_client::connect(info, || Hello::Token(token.clone())).await {
            Ok((link, _)) => return Ok(link),
            Err(e) => {
                log::info!("lucem: the kept sign-in was refused ({e}): pairing again");
                if let Some(s) = storage() {
                    let _ = s.remove_item(TOKEN_KEY);
                }
            }
        }
    }
    let (codes, mut typed) = futures_channel::mpsc::unbounded::<String>();
    let server = info.server.clone().unwrap_or_else(|| "the server".into());
    let on_code: lucem_app_core::screens::sign_in::OnCode = Rc::new(move |code, _, _| {
        let _ = codes.unbounded_send(code);
    });
    let shell = app.read_with(cx, |a, _| a.shell.clone());
    let form = cx
        .update_window(window, |_, window, cx| {
            let form = cx.new(|cx| SignIn::new(server, on_code, window, cx));
            shell.update(cx, |s, cx| s.set_overlay(Some(form.clone().into()), window, cx));
            form
        })
        .map_err(|e| e.to_string())?;
    let name = browser_name();
    while let Some(code) = typed.next().await {
        let (c, n) = (code.clone(), name.clone());
        match web_client::connect(info, move || Hello::Code { code: c.clone(), name: n.clone() }).await {
            Ok((link, token)) => {
                if let (Some(token), Some(s)) = (token, storage()) {
                    let _ = s.set_item(TOKEN_KEY, &token);
                }
                let _ = cx.update_window(window, |_, window, cx| shell.update(cx, |s, cx| s.set_overlay(None, window, cx)));
                return Ok(link);
            }
            Err(e) => {
                let _ = form.update(cx, |f, cx| f.refused(format!("Not paired: {e}"), cx));
            }
        }
    }
    Err("the sign-in form closed".into())
}

#[wasm_bindgen(start)]
pub fn start() {
    // A panic in a page aborts with "unreachable": say what it was first, on the console and on
    // the loader's note
    std::panic::set_hook(Box::new(|info| {
        let text = format!("lucem panicked: {info}");
        console_error(&text);
        // The loader shows it (its own catch would only see "unreachable")
        if let Some(w) = web_sys::window() {
            let _ = js_sys::Reflect::set(&w, &"lucemLastPanic".into(), &text.clone().into());
        }
    }));
    console_log("lucem: start");
    static CONSTRUCTED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if !CONSTRUCTED.swap(true, std::sync::atomic::Ordering::SeqCst) {
        // SAFETY: the linker's own function; called once, before anything reads a registry
        unsafe { __wasm_call_ctors() };
    }
    console_log("lucem: constructors ran");
    gpui_kit::platform::web_init();
    console_log("lucem: web platform ready");
    // With threads in the build, GPUI's background work (tile reads, which wait for the server)
    // runs on web workers
    let threads = cfg!(target_feature = "atomics");
    let platform = std::rc::Rc::new(gpui_kit::web::WebPlatform::new_with_backend(threads, backend()));
    let application = gpui_kit::Application::with_platform(platform);
    log::info!("lucem: threads in this build: {threads}, drawing: {:?}", backend());
    application.with_assets(lucem_app_core::ui::assets::app_assets()).run(|cx: &mut App| {
        cx.text_system().add_fonts(vec![std::borrow::Cow::Borrowed(include_bytes!("AdwaitaSans-Regular.ttf").as_slice())]).expect("the font");
        gpui_kit::init(cx);
        gpui_kit::component::Theme::global_mut(cx).font_family = FONT.into();
        actions::bind_keys(cx);
        console_log("lucem: application running");
        let store = cx.new(|_| Store::new(Prefs::default(), None));
        let options = WindowOptions { titlebar: Some(TitlebarOptions { title: Some("Lucem".into()), ..TitleBar::title_bar_options() }), ..Default::default() };
        let s = store.clone();
        let window = gpui_kit::open_window(options, cx, |window, cx| cx.new(|cx| LucemApp::new(s, window, cx))).expect("the browser window");
        // Served by `lucem serve --web`: sign in, then the server menu lists its images; a link
        // can name one (`?open=<path under the data folder>`)
        let s = store.clone();
        cx.spawn(async move |cx| {
            let said = match web_client::serve_info().await {
                Ok(Some(info)) => match sign_in(&info, window, cx).await {
                    Ok(link) => {
                        let w = link.welcome().clone();
                        let host = link.host().to_string();
                        let transport = link.transport();
                        lucem_app_core::data::remote::set_page_link(link);
                        // The work of the last visit, as it was left (its images are the server's)
                        let _ = s.update(cx, |s, cx| {
                            s.restore_session(cx);
                        });
                        if let Some(path) = link_parameter("open") {
                            let name = lucem_stream::remote_name(&host, &path);
                            let _ = s.update(cx, |s, cx| s.open(std::path::PathBuf::from(name), cx));
                        }
                        let runtime = if w.runtime.is_empty() { "no processing on the server".to_string() } else { format!("processing on its {}", w.runtime) };
                        format!("Connected to {} as {} ({}, {transport:?}, {runtime}): Connect to a lucem server… lists its images", w.server_name, w.peer_name, w.permission)
                    }
                    Err(e) => format!("Not connected to the server: {e}"),
                },
                Ok(None) => "This page shows images from lucem serve --web: open it from there".to_string(),
                Err(e) => format!("The page could not ask its server: {e}"),
            };
            let _ = s.update(cx, |s, cx| s.set_status(said, cx));
        })
        .detach();
    });
}

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console, js_name = error)]
    fn console_error(s: &str);
    #[wasm_bindgen(js_namespace = console, js_name = log)]
    fn console_log(s: &str);
}
