//! Paths as the interface shows them: never the whole path (it names the user and their folders
//! in screenshots and shared logs), only the last two parts.

use std::path::Path;

/// `…/slides/case-12.ome.zarr` for `/home/<user>/data/slides/case-12.ome.zarr`; a path of two
/// parts or fewer, or a `sample:`, as it is.
pub fn short_path(path: &Path) -> String {
    let s = path.to_string_lossy();
    if s.starts_with("sample:") {
        return s.into_owned();
    }
    let parts: Vec<String> = path.components().filter(|c| matches!(c, std::path::Component::Normal(_))).map(|c| c.as_os_str().to_string_lossy().into_owned()).collect();
    if parts.len() <= 2 && !path.is_absolute() {
        return parts.join("/");
    }
    let tail = &parts[parts.len().saturating_sub(2)..];
    format!("…/{}", tail.join("/"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_the_last_two_parts_show() {
        assert_eq!(short_path(Path::new("/home/someone/data/slides/case-12.ome.zarr")), "…/slides/case-12.ome.zarr");
        assert_eq!(short_path(Path::new("/project")), "…/project");
        assert_eq!(short_path(Path::new("slides/a.tif")), "slides/a.tif");
        assert_eq!(short_path(Path::new("sample:volume")), "sample:volume");
    }
}
