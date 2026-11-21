//! Icons: gpui-kit's component icons plus the Lucide icons lucem adds (tools, workspaces, views,
//! the frame's meters), embedded on their own and shipped through kitchen-ui's `AppAssets`, the
//! same on the desktop, in a page and in the container's page (as dsp-kitchen's). In a browser
//! only embedded icons draw: an icon not listed here is an empty square there.

gpui_kit::assets::icon_assets!(
    pub ExtraIcons,
    [AppWindow, Bell, Box, BrainCircuit, Brush, Camera, ChartBar, ChartCandlestick, ChartColumn, ChartGantt, ChartScatter, Circle, Clock, Code, Columns2, Columns3, Combine, Contrast, Crosshair, Download, Eraser, Expand, FileCode, FileInput, FileOutput, FilePlus, Flame, Gauge, GitCompare, Gpu, Grid2x2, Grid3x3, Grip, Hand, Image, ImagePlus, Keyboard, Lasso, Layers, Layers2, LayoutGrid, Link, MapPin, Microscope, Monitor, MousePointer2, Package, PaintBucket, PanelLeft, Pencil, Pentagon, Pipette, Power, RotateCcw, Rows2, Ruler, Save, Scan, ScanSearch, Server, Shapes, Sheet, Shrink, SlidersHorizontal, Sparkles, SquareDashed, SquareMousePointer, SquarePen, SquarePlus, SquaresExclude, SquaresIntersect, SquaresSubtract, SquaresUnite, StickyNote, Table2, Tag, Trash, Type, Unlink, WandSparkles, Workflow, Wrench, X, ZoomIn, ZoomOut]
);

/// Brand wordmark font (used strictly for the "Lucem" title/wordmark).
pub const BRAND_FONT: &str = "Matemasie";

/// Asset paths for the dark and light (cream-background) cell logos.
pub const LOGO_DARK_PATH: &str = "brand/logo-dark.svg";
pub const LOGO_LIGHT_PATH: &str = "brand/logo-light.svg";

/// Returns the cell logo path matching the active theme (dark or light cream).
pub fn logo_path() -> &'static str {
    if crate::ui::theme::is_dark() { LOGO_DARK_PATH } else { LOGO_LIGHT_PATH }
}

/// Registers the embedded brand wordmark font (`Matemasie`) with GPUI's text system.
pub fn register_brand_fonts(cx: &gpui_kit::App) {
    let _ = cx.text_system().add_fonts(vec![std::borrow::Cow::Borrowed(
        include_bytes!("../../assets/fonts/Matemasie-Regular.ttf").as_slice(),
    )]);
}

/// What the app ships: gpui-kit's component icons, [`ExtraIcons`], and the dark/light brand logos.
pub fn app_assets() -> kitchen_ui::icons::AppAssets {
    kitchen_ui::icons::AppAssets::new()
        .icons(ExtraIcons)
        .file(LOGO_DARK_PATH, include_bytes!("../../assets/brand/logo-dark.svg"))
        .file(LOGO_LIGHT_PATH, include_bytes!("../../assets/brand/logo-light.svg"))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every icon lucem's code names (`IconName::Name`) is in the assets it ships, on the
    /// desktop and in a page alike: a name missing from [`ExtraIcons`] draws nothing.
    #[test]
    fn every_icon_used_is_bundled() {
        let roots = [concat!(env!("CARGO_MANIFEST_DIR"), "/src"), concat!(env!("CARGO_MANIFEST_DIR"), "/../lucem-app-web/src")];
        let used = kitchen_ui::icons::icon_names_used(&roots);
        assert!(used.len() > 20, "icons found in the sources: {used:?}");
        kitchen_ui::icons::assert_icons_bundled(&app_assets(), &roots);
        // The frame's own icons (kitchen-shell: Settings, the search, the theme, the meters, the
        // server button, people)
        let frame: std::collections::BTreeSet<String> = ["Palette", "Search", "Settings", "Sun", "Moon", "Close", "Cpu", "MemoryStick", "Server", "Monitor", "CircleUser", "Delete", "Plus"].iter().map(|s| s.to_string()).collect();
        assert_eq!(kitchen_ui::icons::missing_icons(&app_assets(), &frame), Vec::<String>::new(), "the frame's icons are bundled");
    }
}
